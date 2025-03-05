import torch
from einops import rearrange
from torch import nn
from typing import List, Optional, Tuple, Union
from .utils import extend_instance, stack_with_padding, stack_with_padding_2D_attention, num_params, getattr_recursive
from .helpers import DecoupledEmbedding, DecoupledLinear, VLMOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import CLIPVisionModel
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer


class VLM(nn.Module):
    """
    Generic vision-language model (VLM) class.
    A VLM consists of four components:
        1. A vision encoder that extracts features from pixels, e.g. CLIP
            input: (B, T_img, F, C, H, W)
            output: (B, T_img, F, v, d)
        2. A vision tokenizer that converts these features to visual token-like embeddings, e.g. Perceiver, or a linear projection head
            input: (B, T_img, F, v, d)
            output: (B, T_img, n, d)
        3. A fusion method that allows the language model to attend to these tokens, e.g. cross-attention, or placing the tokens directly in the language model's input sequence
        4. A language model
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        gradient_checkpointing: bool = False,
        base_img_size: Optional[int] = None,
    ):
        """
        Args:
            vision_encoder (nn.Module): e.g. CLIP
            vision_tokenizer (nn.Module): e.g. PerceiverResampler
            lang_model (nn.Module): e.g. MPT
            initial_tokenizer_len (int): size of the original tokenizer vocab
            pad_token_id (int): id of the pad token
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()

        # save dimension information
        self.lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        if hasattr(lang_model.config, "d_model"):
            self.lang_hidden_dim = lang_model.config.d_model  # mpt uses d_model
        else:
            self.lang_hidden_dim = lang_model.config.hidden_size
        self.vis_embedding_dim = vision_tokenizer.dim_media
        self.num_tokens_per_vis = vision_tokenizer.num_tokens_per_media

        # core components
        self.vision_encoder = vision_encoder
        self.vision_tokenizer = vision_tokenizer
        self.lang_model = lang_model

        if base_img_size is None:
            if isinstance(self.vision_encoder, CLIPVisionModel) or isinstance(self.vision_encoder, SiglipVisionTransformer):
                base_img_size = self.vision_encoder.config.image_size
            else:
                base_img_size = self.vision_encoder.image_size[0]
        self.base_img_size = base_img_size

        # lm embeddings
        self.pad_token_id = pad_token_id
        self.initial_tokenizer_len = initial_tokenizer_len
        input_embeds = DecoupledEmbedding(
            max_original_id=initial_tokenizer_len - 1,
            num_additional_embeddings=len(self.special_tokens),
            _weight=self.lang_model.get_input_embeddings().weight,
            pad_token_id=self.pad_token_id,
        )
        if hasattr(input_embeds, "additional_embedding"):
            input_embeds.additional_embedding.weight.data.normal_(
                mean=0.0,
                std=self.lang_model.config.initializer_range
                if hasattr(self.lang_model.config, "initializer_range")
                else 0.02,
            )
        self.lang_model.set_input_embeddings(input_embeds)

        out_embeds = DecoupledLinear(
            max_original_id=initial_tokenizer_len - 1,
            additional_out_features=len(self.special_tokens),
            _weight=self.lang_model.get_output_embeddings().weight,
            _bias=self.lang_model.get_output_embeddings().bias if hasattr(self.lang_model.get_output_embeddings(), "bias") else None,
        )
        if hasattr(out_embeds, "additional_fc"):
            out_embeds.additional_fc.weight.data.normal_(
                mean=0.0,
                std=self.lang_model.config.initializer_range
                if hasattr(self.lang_model.config, "initializer_range")
                else 0.02,
            )
        self.lang_model.set_output_embeddings(out_embeds)

        # gradient checkpointing
        self.vision_tokenizer._use_gradient_checkpointing = gradient_checkpointing

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
        if vision_x is not None:
            vision_features = self._encode_vision_x(vision_x=vision_x)
            vision_tokens = self.vision_tokenizer(vision_features)
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
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

        # postprocessing may be needed, e.g. to remove extra tokens from logits that were inserted into the language stream
        # or to add the past_vision_tokens and past_media_locations to the output
        output = self._postprocess_outputs_from_forward(
            output=output,
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            use_cache=use_cache,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
        )

        # postforward hooks
        self._post_forward_hook()
        return output

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        
        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            if self.vision_encoder.__class__.__name__ == "TimmModel":
                vision_x = self.vision_encoder.trunk.forward_features(vision_x)
            elif self.vision_encoder.__class__.__name__ in ['CLIPVisionModel', 'SiglipVisionTransformer']:
                vision_x = self.vision_encoder(vision_x).last_hidden_state
            else:
                vision_x = self.vision_encoder(vision_x)[1]  # OpenCLIP returns tuples
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        return vision_x

    def _concat_vision_cache(
        self, lang_x, vision_tokens, past_vision_tokens, past_media_locations, use_cache
    ):
        """
        Helper function to include the past vision tokens and past media locations in the output.
        """
        if use_cache:
            if past_media_locations is not None and past_vision_tokens is not None:
                if vision_tokens is not None:
                    updated_vision_tokens = torch.cat(
                        [
                            past_vision_tokens,
                            vision_tokens,
                        ],
                        dim=1,
                    )
                else:
                    updated_vision_tokens = past_vision_tokens
                updated_media_locations = torch.cat(
                    [
                        past_media_locations,
                        lang_x == self.media_token_id,
                    ],
                    dim=1,
                )
            else:
                updated_vision_tokens = vision_tokens
                updated_media_locations = lang_x == self.media_token_id

        else:
            updated_vision_tokens = None
            updated_media_locations = None

        return updated_vision_tokens, updated_media_locations

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
        if vision_x is not None:
            vision_features = self._encode_vision_x(vision_x=vision_x)
            vision_tokens = self.vision_tokenizer(vision_features)
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        # for xattn, vision_x and media_location are repeat_interleaved s.t.
        # the total batch size is B * num_beams
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            padding_side="left",
            num_beams=num_beams,
        )
        output = self.lang_model.generate(
            **new_inputs,
            past_key_values=past_key_values,
            num_beams=num_beams,
            use_cache=True,
            **kwargs,
        )
        self._post_forward_hook()
        return output

    @property
    def num_trainable_params(self):
        """Print the number of trainable parameters"""
        return num_params(self, filter_to_trainable=True)

    def set_trainable(self):
        """
        Freeze appropriate parameters in the model.
        """
        raise NotImplementedError

    def group_params_by_weight_decay(self):
        """
        Return a tuple of (params to optimize w/ weight decay, params to optimize w/o weight decay)
        """
        params_with_wd, params_without_wd = [], []
        for n, p in self.named_parameters():
            if p.requires_grad:    
                if self._should_apply_weight_decay(n):
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)
        return params_with_wd, params_without_wd

    def _should_apply_weight_decay(self, parameter_name):
        """
        Return whether weight decay should be applied to a parameter.
        """
        raise NotImplementedError

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
            setattr(self.lang_model, f"{att_name}_id", token_id)

    def init_gradient_checkpointing(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointWrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )
        from functools import partial

        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            self,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
            and not isinstance(m, CheckpointWrapper),
        )


class VLMWithLanguageStream(VLM):
    """
    VLM that fuses modalities by inserting vision tokens directly into the language stream.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
        base_img_size: Optional[int] = None,
    ):
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=vision_tokenizer,
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            pad_token_id=pad_token_id,
            base_img_size=base_img_size,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.decoder_layers_attr_name = decoder_layers_attr_name
        for block in getattr_recursive(self.lang_model, self.decoder_layers_attr_name):
            block._use_gradient_checkpointing = gradient_checkpointing

    @staticmethod
    def _make_modality_mutual_mask(
        attention_mask_2d: torch.Tensor,
        image_start_idx: int,
        text_start_idx: int,
        text_end_idx: int,              # the end of the question in the SFT stage
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Make non-causal mask between modalities.
        """
        tgt_len = input_ids_shape[0]
        mask = torch.full((tgt_len, tgt_len), 0, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)

        # enable vision tokens to attend to text tokens
        mask[image_start_idx:text_start_idx, text_start_idx:text_end_idx] = 1

        mask = mask.to(dtype)
        mask = mask[None, :, :].expand(1, tgt_len, tgt_len)

        expanded_mask = attention_mask_2d[None, None, :].expand(1, tgt_len, tgt_len).to(torch.float32)
        inverted_mask = 1.0 - expanded_mask
        expanded_attn_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(torch.float32).min)

        expanded_attn_mask = mask.masked_fill(expanded_attn_mask.bool(), 0)

        # expanded_attn_mask + causal_4d_mask can cause some overflow
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask
    
    def _prepare_inputs_for_forward(
        self,
        vision_tokens: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        past_key_values=None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        past_media_locations: torch.Tensor = None,
        past_vision_tokens: torch.Tensor = None,
        padding_side: str = "left",
        num_beams: int = 1,
    ):
        """
        Insert the vision tokens directly into the language stream/
        This requires us to modify the input_ids, attention_mask, and labels.
        [NOTE]: This function can be changed to fit the ablation setting of putting text before images.
        """
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2]
            assert attention_mask.shape[1] == past_len + lang_x.shape[1], (
                "Attention_mask must be as long as the entire past len (including image tokens) and current input IDs. "
                + "Check that you've expanded the attention mask to account for past image tokens."
            )
        
        if vision_tokens is None:
            return {
                "input_ids": lang_x,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        # get the language embeddings
        lang_embeds = self.lang_model.get_input_embeddings()(lang_x)

        # build up the multimodal embeddings
        B = lang_x.shape[0]
        has_labels = labels is not None
        multimodal_embeds = []
        multimodal_attention_mask = []
        multimodal_labels = [] if has_labels else None
        for i in range(B):
            # get index of <image> tokens in lang_x[i]
            image_token_idxs = torch.where(lang_x[i] == self.media_token_id)[0]

            # get the <|assistant|> token index, hardcode for now but can easily get from tokenizer's special tokens
            # assume only one <|assistant|> token, i.e., single-turn
            question_token_idx = torch.where(lang_x[i] == 32001)[0]
            if len(question_token_idx) != 0:
                question_token_idx = question_token_idx[0]
            else:
                question_token_idx = 0

            if len(image_token_idxs) == 0:
                multimodal_embeds.append(lang_embeds[i].clone())
                new_attention_mask = self._make_modality_mutual_mask(
                                        attention_mask_2d=attention_mask[i].clone(),
                                        image_start_idx=0,
                                        text_start_idx=0,
                                        text_end_idx=question_token_idx,
                                        input_ids_shape=attention_mask[i].shape,
                                        dtype=attention_mask[i].dtype,
                                        device=attention_mask[i].device,
                                    )
                multimodal_attention_mask.append(new_attention_mask)
                if has_labels:
                    multimodal_labels.append(labels[i].clone())
                continue

            # since an image is represented by self.num_tokens_per_vis tokens, we need to offset the image_token_idxs
            # loop through the image_token_idxs and insert the vision tokens
            new_embed = lang_embeds[i].clone()
            new_attention_mask = (
                attention_mask[i].clone() if attention_mask is not None else None
            )
            if has_labels:
                new_label = labels[i].clone()

            for img_num in range(len(image_token_idxs)):
                img_idx = image_token_idxs[img_num]
                assert (
                    vision_tokens[i][img_num].shape[0] == self.num_tokens_per_vis
                ), f"vision token number mismatch: image embedding ({vision_tokens[i][img_num].shape[0]}) \
                        vs. model.num_tokens_per_vis ({self.num_tokens_per_vis})"
                # By default, vision tokens are not padded.
                num_vis_tokens = self.num_tokens_per_vis
                vis_attention_mask = torch.ones(
                    num_vis_tokens, dtype=torch.long
                ).to(attention_mask.device)
                
                # Offset the rest of image tokens with current num_vis_tokens
                for j in range(img_num+1, len(image_token_idxs)):
                    image_token_idxs[j] += num_vis_tokens

                new_embed = torch.cat(
                    (
                        new_embed[:img_idx],
                        vision_tokens[i][img_num],
                        new_embed[img_idx + 1 :],
                    ),
                    dim=0,
                )
                new_attention_mask = torch.cat(
                    (
                        new_attention_mask[:img_idx],
                        vis_attention_mask,
                        new_attention_mask[img_idx + 1 :],
                    ),
                    dim=0,
                )

                new_attention_mask = self._make_modality_mutual_mask(
                                attention_mask_2d=new_attention_mask,
                                image_start_idx=img_idx,
                                text_start_idx=img_idx+len(vis_attention_mask),                                      # 1+128 -> start position of text
                                text_end_idx=question_token_idx+len(vis_attention_mask),
                                input_ids_shape=new_attention_mask.shape,                                             # (252)
                                dtype=new_attention_mask.dtype,
                                device=new_attention_mask.device,
                            )

                if has_labels:
                    new_label = torch.cat(
                        (
                            new_label[:img_idx],
                            torch.ones(num_vis_tokens, dtype=torch.long).to(
                                labels.device
                            )
                            * -100,
                            new_label[img_idx + 1 :],
                        ),
                        dim=0,
                    )
            multimodal_embeds.append(new_embed)
            multimodal_attention_mask.append(new_attention_mask)
            if has_labels:
                multimodal_labels.append(new_label)

        # stack
        multimodal_embeds = stack_with_padding(
            multimodal_embeds,
            padding_value=self.pad_token_id,
            padding_side=padding_side,
        )
        multimodal_attention_mask = stack_with_padding_2D_attention(
            multimodal_attention_mask,
        )
        if has_labels:
            multimodal_labels = stack_with_padding(
                multimodal_labels,
                padding_value=-100,
                padding_side=padding_side,
            )

        return {
            "inputs_embeds": multimodal_embeds,
            "attention_mask": multimodal_attention_mask,
            "labels": multimodal_labels,
        }

    def _postprocess_outputs_from_forward(
        self,
        output: CausalLMOutputWithPast,
        lang_x: torch.Tensor,
        vision_tokens: torch.Tensor,
        past_vision_tokens: torch.Tensor,
        past_media_locations: torch.Tensor,
        use_cache: bool = False,
    ):
        # Include the past vision tokens and past media locations in the output
        updated_vision_tokens, updated_media_locations = self._concat_vision_cache(
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
            use_cache=use_cache,
        )

        # return logits that are the same shape as the original input_ids
        logits = output.logits
        batch_logits = []
        B, T_txt = lang_x.shape
        for i in range(B):
            sequence_logits = []
            logits_j = 0
            img_id = 0
            for j in range(T_txt):
                if lang_x[i, j] != self.media_token_id:
                    sequence_logits.append(logits[i, logits_j])
                    logits_j += 1
                else:
                    # append the logit for the first image token, then skip over the rest
                    # note: the model actually learns to predict <im_patch>, not <image>
                    sequence_logits.append(logits[i, logits_j])
                    # logits_j += self.num_tokens_per_vis
                    # Offset in account of dynamic num_vis_tokens.
                    logits_j += vision_tokens[i][img_id].shape[0]
                    img_id += 1
            sequence_logits = torch.stack(sequence_logits, dim=0)  # (B, vocab_size)
            batch_logits.append(sequence_logits)

        batch_logits = torch.stack(batch_logits, dim=0)  # (B, T_txt, vocab_size)
        # The final logits shape should be the same as the original input_ids shape
        assert batch_logits.shape[:2] == (B, T_txt)

        # assemble the output
        output = VLMOutputWithPast(
            loss=output.loss,
            logits=batch_logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            past_media_locations=updated_media_locations,
            past_vision_tokens=updated_vision_tokens,
        )

        return output

    def _post_forward_hook(self):
        pass

    def get_fsdp_lambda_fn(self):
        """
        Returns the lambda function used to decide how to perform FSDP wrapping.
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        decoder_block_class = getattr_recursive(
            self.lang_model, self.decoder_layers_attr_name
        )[0].__class__

        def lambda_fn(module: nn.Module):
            if getattr(module, "_use_gradient_checkpointing", False) and not isinstance(
                module, CheckpointWrapper
            ):
                return False
            if module is self.vision_tokenizer:
                return True
            if isinstance(module, decoder_block_class):
                return True

        return lambda_fn
    
    def get_fsdp_wrapping_policy(self):
        """
        Returns the policy used to decide how to perform FSDP wrapping.
        """
        from torch.distributed.fsdp.wrap import _or_policy, _module_wrap_policy, transformer_auto_wrap_policy
        from open_clip.transformer import VisionTransformer, ResidualAttentionBlock
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from transformers.models.phi.modeling_phi import PhiDecoderLayer
        # for Phi-3 hot fiix
        try:
            import importlib
            commit_hash = str(type(self.lang_model)).split('instruct.')[1].split('.modeling')[0]
            module_name = f"transformers_modules.microsoft.Phi-3-mini-128k-instruct.{commit_hash}.modeling_phi3"
            module = importlib.import_module(module_name)
            Phi3DecoderLayer = module.Phi3DecoderLayer
            import_phi3 = True
        except IndexError:
            import_phi3 = False
            
                
        # hard code the wrap module name
        # vision
        if isinstance(self.vision_encoder, SiglipVisionModel):
            from transformers import SiglipVisionModel
            vit_wrap_policy = functools.partial(_module_wrap_policy, module_classes={SiglipVisionModel})
            from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer, SiglipVisionTransformer, SiglipVisionEmbeddings, SiglipMultiheadAttentionPoolingHead
            # import torch.nn.LayerNorm as LayerNorm
            transformer_layer_cls_vit = {SiglipEncoderLayer, SiglipVisionTransformer, SiglipVisionEmbeddings, SiglipMultiheadAttentionPoolingHead}
            vision_transformer_block_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls_vit)
            vision_wrap_policy = functools.partial(_or_policy, policies=[vit_wrap_policy, vision_transformer_block_policy])

        else:
            vit_wrap_policy = functools.partial(_module_wrap_policy, module_classes={VisionTransformer, TimmModel})
            # vit_wrap_policy = functools.partial(_module_wrap_policy, module_classes={VisionTransformer})
            # transformer_layer_cls_vit = {ResidualAttentionBlock}
            transformer_layer_cls_vit = {ResidualAttentionBlock, Block}
            # transformer_layer_cls_vit = {Block}
            vision_transformer_block_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls_vit)
            vision_wrap_policy = functools.partial(_or_policy, policies=[vit_wrap_policy, vision_transformer_block_policy])
        # llm
        transformer_layer_cls={LlamaDecoderLayer, PhiDecoderLayer}
        if import_phi3:
            transformer_layer_cls.add(Phi3DecoderLayer)
        llm_transformer_block_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls)
        # vision_tokenizer
        vis_tokenizer_policy = functools.partial(_module_wrap_policy, module_classes={LinearPatchProjection, PerceiverResampler})
        return functools.partial(
            _or_policy, 
            policies = [
                vision_wrap_policy, 
                llm_transformer_block_policy,
                vis_tokenizer_policy
            ])    

    def group_params_by_weight_decay(self):
        """
        Return a tuple of (params to optimize w/ weight decay, params to optimize w/o weight decay)
        """
        params_with_wd, params_without_wd = [], []
        for n, p in self.named_parameters():
            if p.requires_grad:
                if "lang_model.model.embed_tokens" in n:
                    params_without_wd.append(p)
                else:
                    params_with_wd.append(p)
        return params_with_wd, params_without_wd

    @property
    def num_params_per_module(self):
        """Print the number of parameters per module in the model"""
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder):,} parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer):,} parameters",
                f"Language model: {num_params(self.lang_model):,} parameters",
            ]
        )

    @property
    def num_trainable_params_per_module(self):
        """Print the number of trainable parameters per module in the model"""
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder, filter_to_trainable=True):,} trainable parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer, filter_to_trainable=True):,} trainable parameters",
                f"Language model: {num_params(self.lang_model, filter_to_trainable=True):,} trainable parameters",
            ]
        )
