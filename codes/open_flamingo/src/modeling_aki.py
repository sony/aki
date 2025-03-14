import torch
from einops import rearrange
from torch import nn
from typing import List, Optional, Tuple, Union
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from .helpers import PerceiverResampler
from .vlm import VLMWithLanguageStream

class AKI(VLMWithLanguageStream, PyTorchModelHubMixin):
    def __init__(
        self,
        vision_encoder_path: str,
        lang_model_path: str,
        pad_token_id: int,
        initial_tokenizer_len: Optional[int] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
        base_img_size: Optional[int] = None,
        num_vision_tokens: int = 144,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            initial_tokenizer_len (int): size of the tokenizer vocab
            padding_token_id (int): id of the padding token. None if no padding token; then a padding token
                will be inserted into self.special_tokens, which factory.py fills after creating new tokens
            decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        """

        # load the vision model
        model = AutoModel.from_pretrained(vision_encoder_path)
        vision_encoder = model.vision_model
        vis_feature_dim = vision_encoder.config.hidden_size

        # load the language model
        lang_model = AutoModelForCausalLM.from_pretrained(
            lang_model_path,
            local_files_only=False,
            trust_remote_code=True,
        )

        self._special_tokens = {
            "media_token": "<image>",
            "end_of_trunk_token": "<|endofchunk|>",
        }
        lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=PerceiverResampler(
                dim=vis_feature_dim, dim_inner=lang_embedding_dim,
                num_latents=num_vision_tokens,
            ),
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            gradient_checkpointing=gradient_checkpointing,
            base_img_size=base_img_size,
            decoder_layers_attr_name=decoder_layers_attr_name,
            pad_token_id=pad_token_id,
        )

        if tokenizer is not None:
            self.lang_model.config.vocab_size = len(tokenizer)
            self.set_special_token_ids(
                {
                    v: tokenizer.convert_tokens_to_ids(v)
                    for v in self.special_tokens.values()
                }
            )

    def set_trainable(self):
        """
        Unfreeze everything except the vision_encoder
        """
        self.requires_grad_(True)
        self.vision_encoder.requires_grad_(False)
    
    def forward(
        self,
        vision_x: Optional[torch.Tensor],
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W) with F=1
                only F = 1 is supported (single-frame videos)
                if T_img > the number of media tokens in the corresponding input_ids (lang_x),
                only the first number of media tokens in lang_x are used
            lang_x: Language input ids, with media tokens denoting where
                visual media should be inserted.
                shape (B, T_txt)
            attention_mask: Attention mask. Defaults to None.
            labels: Labels. Defaults to None.
                shape (B, T_txt)
            past_key_values (Tuple[torch.Tensor]], optional): Past key value pairs for each of the T_txt previous tokens in the language model. Defaults to None.
                list of length = number of decoder layers in the LM
                exact implementation depends on LM, see Hugging Face docs
            past_media_locations (torch.Tensor, optional): boolean mask denoting which of the previous T_txt tokens were media tokens. Defaults to None.
                shape (B, T_txt)
            past_vision_tokens (torch.Tensor, optional): Previous vision tokens. Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to False.
                If True, includes key_values, media_locations, and vision_tokens in the output.
        """
        assert not (past_vision_tokens is None) ^ (
            past_media_locations is None
        ), "past_vision_tokens and past_media_locations must both be None or both be not None"

        # convert pixels to vision tokens
        vision_attention_mask = None
        if vision_x is not None:
            vision_tokens = self.vision_tokenizer(self._encode_vision_x(vision_x=vision_x))
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            padding_side="right",
            past_vision_tokens=past_vision_tokens,
        )
        output = self.lang_model(
            **new_inputs,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )

        # postforward hooks
        self._post_forward_hook()
        return output
    
    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                see documentation for forward
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            **kwargs: see generate documentation in Hugging Face CausalLM models.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)

        # convert pixels to vision tokens
        vision_attention_mask = None
        if vision_x is not None:
            vision_tokens = self.vision_tokenizer(self._encode_vision_x(vision_x=vision_x))
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            padding_side="left",
            num_beams=num_beams,
        )

        # customize handling of position_ids since attention mask is already formulated as 4D
        if len(new_inputs["attention_mask"].shape) == 4:
            position_ids = new_inputs.get("position_ids", None)
            if position_ids is None:
                seq_length = new_inputs["inputs_embeds"].shape[1]
                position_ids = torch.arange(seq_length, dtype=torch.long, device=new_inputs["inputs_embeds"].device)
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
                new_inputs["position_ids"] = position_ids

        if past_key_values is not None:
            output = self.lang_model.generate(
                **new_inputs,
                past_key_values=past_key_values,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        else:
            output = self.lang_model.generate(
                **new_inputs,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        self._post_forward_hook()
        return output

