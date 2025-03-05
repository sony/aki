from typing import Optional

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor, AutoModel, AutoProcessor
import open_clip

from .aki import AKI

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    cache_dir: Optional[str] = None,
    gradient_checkpointing: bool = False,
    verbose: bool = True,
    **model_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
        freeze_lm_embeddings (bool, optional): whether to freeze LM input embeddings when configuring Perceiver.
        cache_dir (str, optional): path to cache directory for downloading OpenClip/HF weights.
        gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        verbose (bool, optional): whether to print model info. Defaults to True.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """

    # load vision encoder
    if clip_vision_encoder_pretrained == 'openai':
        vision_encoder = CLIPVisionModel.from_pretrained(clip_vision_encoder_path)
        hf_processor = CLIPImageProcessor.from_pretrained(clip_vision_encoder_path)
        n_px = hf_processor.crop_size['height']
        # Use torchvision processor to be consistent with other vision encoders.
        # https://github.com/openai/CLIP/blob/main/clip/clip.py
        image_processor = Compose([
                                Resize((n_px, n_px), interpolation=BICUBIC),
                                CenterCrop(n_px),
                                _convert_image_to_rgb,
                                ToTensor(),
                                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                            ])
        vis_hidden_dim = vision_encoder.config.hidden_size
    elif clip_vision_encoder_pretrained == 'google':
        # "google/siglip-so400m-patch14-384"
        model = AutoModel.from_pretrained(f"{clip_vision_encoder_pretrained}/{clip_vision_encoder_path}")
        hf_processor = AutoProcessor.from_pretrained(f"{clip_vision_encoder_pretrained}/{clip_vision_encoder_path}")
        n_px = hf_processor.image_processor.size['height']
        vision_encoder = model.vision_model
        vis_hidden_dim = vision_encoder.config.hidden_size
        
        # Define the transformation sequence
        image_processor = Compose([
            Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC, antialias=True),
            Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    else:
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            clip_vision_encoder_path,
            pretrained=clip_vision_encoder_pretrained,
        )
        vision_encoder.visual.output_tokens = True
        vision_encoder = vision_encoder.visual
        vision_encoder_config = open_clip.get_model_config(clip_vision_encoder_path)
        if "SigLIP" in clip_vision_encoder_path or "EVA" in clip_vision_encoder_path: # SigLIP models have a different config format
            vis_hidden_dim = vision_encoder_config["embed_dim"]
        else:    
            vis_hidden_dim = vision_encoder_config["vision_cfg"]["width"]

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
        use_fast=False,
    )

    # enable adding bos and eos tokens
    text_tokenizer.add_bos_token = True
    text_tokenizer.add_eos_token = True

    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir, 
    )
    check_embedding_fns(lang_encoder)

    if text_tokenizer.pad_token is None or text_tokenizer.pad_token == text_tokenizer.eos_token:
        # add a pad token if it doesn't exist
        text_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        added_pad_token = True
    else:
        added_pad_token = False

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)

    model = AKI(
        vision_encoder=vision_encoder,
        lang_model=lang_encoder,
        vis_feature_dim=vis_hidden_dim,
        initial_tokenizer_len=len(text_tokenizer),
        gradient_checkpointing=gradient_checkpointing,
        decoder_layers_attr_name=decoder_layers_attr_name,
        pad_token_id=text_tokenizer.pad_token_id,
        **model_kwargs,
    )

    # add special tokens to the tokenizer and language models
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": list(model.special_tokens.values())}
    )
    # model.lang_model.resize_token_embeddings(len(text_tokenizer))
    model.lang_model.config.vocab_size = len(text_tokenizer)
    model.set_special_token_ids(
        {
            v: text_tokenizer.convert_tokens_to_ids(v)
            for v in model.special_tokens.values()
        }
    )

    # freeze appropriate parameters
    model.set_trainable()

    if verbose:
        print(f"==========Model initialized with {model.num_trainable_params:,} trainable parameters")
        print(f"==========Trainable Parameters\n{model.num_trainable_params_per_module}")
        print(f"==========Total Parameters\n{model.num_params_per_module}\n==========")
    return model, image_processor, text_tokenizer


def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
    "gemma": "model.layers",
    "phi": "model.layers",
    "minicpm": "model.layers",
    "stablelm": "model.layers",
    "qwen": "model.layers",
    "mistral": "model.layers"
}


def check_embedding_fns(lang_model):
    """Checks for and attempts to set {get/set}_{input/output}_embeddings functions to the model"""
    if not has_fn(lang_model, "get_input_embeddings"):
        if hasattr_recursive(lang_model, "transformer.wte"):  # MPT
            lang_model.get_input_embeddings = lambda: lang_model.transformer.wte
        elif hasattr_recursive(lang_model, "model.decoder.embed_tokens"):  # OPT
            lang_model.get_input_embeddings = lambda: lang_model.decoder.embed_tokens
        else:
            raise ValueError(
                "We require the language encoder to have a get_input_embeddings method but we couldn't determine the name of the input embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "set_input_embeddings"):
        if hasattr_recursive(lang_model, "transformer.wte"):  # MPT
            lang_model.set_input_embeddings = lambda x: setattr_recursive(
                lang_model, "transformer.wte", x
            )
        elif hasattr_recursive(lang_model, "model.decoder.embed_tokens"):  # OPT
            lang_model.set_input_embeddings = lambda x: setattr_recursive(
                lang_model, "model.decoder.embed_tokens", x
            )
        else:
            raise ValueError(
                "We require the language encoder to have a set_input_embeddings method but we couldn't determine the name of the input embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "get_output_embeddings"):
        if hasattr_recursive(lang_model, "lm_head"):
            lang_model.get_output_embeddings = lambda: lang_model.lm_head
        else:
            raise ValueError(
                "We require the language encoder to have a get_output_embeddings method but we couldn't determine the name of the output embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "set_output_embeddings"):
        if hasattr_recursive(lang_model, "lm_head"):
            lang_model.set_output_embeddings = lambda x: setattr_recursive(
                lang_model, "lm_head", x
            )
        else:
            raise ValueError(
                "We require the language encoder to have a set_output_embeddings method but we couldn't determine the name of the output embeddings attribute. Please supply this manually in factory.py."
            )


def has_fn(model, fn_name):
    """Check if model has a function fn_name"""
    return callable(getattr(model, fn_name, None))