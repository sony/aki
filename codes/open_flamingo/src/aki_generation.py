import torch

from transformers.utils import ModelOutput
from typing import Any, Dict


def update_causal_attention_mask(attention_mask, cache=False):
    """
    Updates a causal attention mask by expanding it to (n+1, n+1) during generation.

    Parameters:
        attention_mask (torch.Tensor): Current causal attention mask of shape (1, 1, n, n).

    Returns:
        torch.Tensor: Updated causal attention mask of shape (1, 1, n+1, n+1).
    """
    # Get the current size `n`
    _, _, n, _ = attention_mask.shape
    
    # Create a new row and column with -inf values
    new_row = torch.full((1, 1, 1, n), 1, device=attention_mask.device)
    new_col = torch.full((1, 1, n+1, 1), 0, device=attention_mask.device)

    new_col[0, 0, -1, -1] = 1
    
    # Concatenate the new row and column to the existing mask
    attention_mask = torch.cat([attention_mask, new_row], dim=2)  # Add the new row
    attention_mask = torch.cat([attention_mask, new_col], dim=3)  # Add the new column
    
    if cache:
        return attention_mask[:, :, -1:, :]
    else:
        return attention_mask


def _aki_update_model_kwargs_for_generation(
    self,
    outputs: ModelOutput,
    model_kwargs: Dict[str, Any],
    is_encoder_decoder: bool = False,
    standardize_cache_format: bool = False,
    num_new_tokens: int = 1,
) -> Dict[str, Any]:
    # update past_key_values
    model_kwargs["past_key_values"] = self._extract_past_from_model_output(
        outputs, standardize_cache_format=standardize_cache_format
    )
    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            # modify the update mechanism to incorporate 4D attention mask
            attention_mask = model_kwargs["attention_mask"]
            # after the first computation, roll back to the original attention 2D design to fit Huggingface logistics
            model_kwargs["attention_mask"] = torch.full((1, attention_mask.shape[-1]+1), 1, device=attention_mask.device)
    else:
        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                dim=-1,
            )

    if (
        model_kwargs.get("use_cache", True)
        and "cache_position" in model_kwargs
        and model_kwargs["cache_position"] is not None
    ):
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

        # update position_ids and keep only the last one
        position_ids = torch.arange(model_kwargs["past_key_values"][0][0].shape[2]+1, device=model_kwargs["attention_mask"].device).unsqueeze(0)          # +1 for the new token
        if model_kwargs.get("past_key_values", None) is not None:
            position_ids = position_ids[:, -1:]

        model_kwargs["position_ids"] = position_ids

    return model_kwargs