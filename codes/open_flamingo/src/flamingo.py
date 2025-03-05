import torch
from einops import rearrange
from torch import nn
from .helpers import PerceiverResampler
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from .utils import apply_with_stopping_condition, num_params, getattr_recursive


class Flamingo(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        vis_dim: int,
        decoder_layers_attr_name: str = None,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
        """
        super().__init__()
        self._special_tokens = {
            "media_token": "<image>",
            "end_of_trunk_token": "<|endofchunk|>",
        }

        self.eoc_token_id = self._special_tokens["end_of_trunk_token"]
        self.media_token_id = self._special_tokens["media_token"]
        self.vis_dim = vis_dim
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size

        self.vision_encoder = vision_encoder
        self.vision_tokenizer = PerceiverResampler(dim=self.vis_dim)
        self.lang_encoder = lang_encoder
        self.lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        self.decoder_layers_attr_name = decoder_layers_attr_name
        self.lang_encoder.init_flamingo(
            media_token_id=self.media_token_id,
            lang_hidden_size=self.lang_dim,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            gradient_checkpointing=gradient_checkpointing,
        )
        self._use_gradient_checkpointing = gradient_checkpointing
        self.vision_tokenizer._use_gradient_checkpointing = gradient_checkpointing

    def set_trainable(self):
        """
        Unfreeze everything except the vision_encoder
        """
        self.requires_grad_(True)
        self.vision_encoder.requires_grad_(False)

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (
            self.lang_encoder.initialized_flamingo
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        assert (
            self.lang_encoder._use_cached_vision_x or vision_x is not None
        ), "Must provide either vision_x or have precached media using cache_media()."

        if self.lang_encoder._use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_vision_x(vision_x=vision_x)
            self._condition_media_locations(input_ids=lang_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            **kwargs: see generate documentation in Hugging Face CausalLM models. Some notable kwargs:
                max_length (int, optional): Maximum length of the output. Defaults to None.
                attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
                num_beams (int, optional): Number of beams. Defaults to 1.
                max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
                temperature (float, optional): Temperature. Defaults to 1.0.
                top_k (int, optional): Top k. Defaults to 50.
                top_p (float, optional): Top p. Defaults to 1.0.
                no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
                length_penalty (float, optional): Length penalty. Defaults to 1.0.
                num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
                do_sample (bool, optional): Do sample. Defaults to False.
                early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        self.lang_encoder._use_cached_vision_x = True
        self._encode_vision_x(vision_x=vision_x)

        eos_token_id = kwargs.pop("eos_token_id", self.eoc_token_id)
        output = self.lang_encoder.generate(
            input_ids=lang_x,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            num_beams=num_beams,
            **kwargs,
        )

        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
        return output

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            if self.vision_encoder.__class__.__name__ == "TimmModel":
                vision_x = self.vision_encoder.trunk.forward_features(vision_x)
            elif self.vision_encoder.__class__.__name__ in ['CLIPVisionModel', 'SiglipVisionTransformer']:
                vision_x = self.vision_encoder(vision_x).last_hidden_state
            else:
                vision_x = self.vision_encoder(vision_x)[1]  # OpenCLIP returns tuples
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = self.vision_tokenizer(vision_x)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

    def get_fsdp_lambda_fn(self):
        """
        Returns the lambda function used to decide how to perform FSDP wrapping.
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )
        from .helpers import GatedCrossAttentionBlock

        decoder_block_class = getattr_recursive(
            self.lang_encoder, self.decoder_layers_attr_name
        )[0].__class__

        def lambda_fn(module: nn.Module):
            # we want FSDP(ckpt(module)), not ckpt(FSDP(module))
            if getattr(module, "_use_gradient_checkpointing", False) and not isinstance(
                module, CheckpointWrapper
            ):
                return False
            if module is self.vision_tokenizer:
                return True
            if isinstance(module, GatedCrossAttentionBlock):
                return True
            if isinstance(module, decoder_block_class):
                return True

        return lambda_fn

    def _condition_media_locations(self, input_ids: torch.Tensor):
        """
        Compute the media token locations from lang_x and condition the language model on these.
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
        """
        media_locations = input_ids == self.media_token_id

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_media_locations(media_locations)

    def cache_media(self, input_ids: torch.Tensor, vision_x: torch.Tensor):
        """
        Pre-cache a prompt/sequence of images / text for log-likelihood evaluations.
        All subsequent calls to forward() will generate attending to the LAST
        image in vision_x.
        This is not meant to be used to cache things for generate().
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
        """
        self._encode_vision_x(vision_x=vision_x)
        self._condition_media_locations(input_ids=input_ids)
        self.lang_encoder._use_cached_vision_x = True

    def uncache_media(self):
        """
        Clear all conditioning.
        """
        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False

    @property
    def special_tokens(self):
        """
        Returns a dict mapping from the attribute name of a special token to its string format,
         e.g. "media_token": "<image>"
        """
        assert (
            "media_token" in self._special_tokens
        ), "VLMs need to request that the tokenizer add a media_token and call set_special_token_ids to set self.media_token_id"
        return self._special_tokens

    @property
    def special_token_ids(self):
        """
        Returns a list of the special token ids
        """
        return [getattr(self, f"{att_name}_id") for att_name in self.special_tokens]

    def set_special_token_ids(self, string_to_ids):
        """
        Args:
            string_to_ids (dict): mapping from token string to id
        """
        assert set(self.special_tokens.values()).issubset(set(string_to_ids.keys()))
        for att_name, token_str in self.special_tokens.items():
            token_id = string_to_ids[token_str]
            setattr(self, f"{att_name}_id", token_id)
            setattr(self.lang_encoder, f"{att_name}_id", token_id)

    @property
    def num_trainable_params(self):
        """Print the number of trainable parameters"""
        return num_params(self, filter_to_trainable=True)

    @property
    def num_params_per_module(self):
        """Print the number of parameters per module in the model"""
        num_xattn_params = num_params(self.lang_encoder.gated_cross_attn_layers)
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder):,} parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer):,} parameters",
                f"Cross attention: {num_xattn_params:,} parameters",
                f"Language model: {num_params(self.lang_encoder) - num_xattn_params:,} parameters",
            ]
        )

    @property
    def num_trainable_params_per_module(self):
        """Print the number of trainable parameters per module in the model"""
        num_xattn_params = num_params(
            self.lang_encoder.gated_cross_attn_layers, filter_to_trainable=True
        )
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder, filter_to_trainable=True):,} trainable parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer, filter_to_trainable=True):,} trainable parameters",
                f"Cross attention: {num_xattn_params:,} trainable parameters",
                f"Language model: {num_params(self.lang_encoder, filter_to_trainable=True) - num_xattn_params:,} trainable parameters",
            ]
        )
