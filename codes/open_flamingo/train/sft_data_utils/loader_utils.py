import torch
import numpy as np
import random
from .templates.templates import IGNORE_INDEX


_INFINITE = int(1e12)  # infinite token length for no-truncation
_IMG_SIZE = 384


def _pad_trunc(
    x: list[list[int]],
    padding: str,
    padding_side: str,
    pad_value: int,
    max_length: int,
) -> torch.LongTensor:
    """Pad and truncate sequences to the same length

    Args:
        x (list[list[int]])
        padding ("longest" or "max_length")
        padding_side ("left" or "right")
        pad_value (int)
        max_length (int or None): if padding == "max_length", max_length should be given.
    """
    assert padding in ["longest", "max_length"]
    assert padding_side in ["left", "right"]

    lengths = [len(sample) for sample in x]
    if padding == "longest":
        max_length = max(lengths)

    new_x = []
    for sample, length in zip(x, lengths):
        if torch.is_tensor(sample):
            sample = sample.tolist()

        if length >= max_length:
            new_x.append(sample[:max_length])
            continue

        padding_size = max_length - length
        pads = [pad_value] * padding_size
        if padding_side == "right":
            new_x.append(sample + pads)
        else:
            new_x.append(pads + sample)

    return torch.as_tensor(new_x, dtype=torch.long)


def batch_collate_pad(
    batch: list,
    padding: str,
    padding_side: str,
    pad_token_id: int,
    max_length: int | None,
) -> dict[str, torch.LongTensor]:
    """Collate batch and pad/truncate to the same length

    Args:
        batch
        padding ("longest" or "max_length")
        padding_side ("left" or "right")
        pad_value (int)
        max_length (int or None): if padding == "max_length", max_length should be given
    """
    if padding == "max_length":
        assert max_length is not None, "max_length should be given if padding == 'max_length'"
    else:
        # if padding == 'longest' and max_length is None, set to infinite for no-truncation
        max_length = max_length or _INFINITE

    input_ids = [sample["input_ids"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    attention_mask = [sample["attention_mask"] for sample in batch]

    # max_length + 1 for bos_token
    input_ids = _pad_trunc(input_ids, padding, padding_side, pad_token_id, max_length+1)
    labels = _pad_trunc(labels, padding, padding_side, IGNORE_INDEX, max_length+1)
    attention_mask = _pad_trunc(attention_mask, padding, padding_side, 0, max_length+1)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def batchify(batch, tokenizer, max_length: int, use_trunc=False):
    """collate_fn
    Args:
        batch
        tokenizer
        max_length (int)
        use_trunc (bool)

    """

    # Collate for text
    text_batch = [data["text"] for data in batch]
    padding = "longest" if use_trunc else "max_length"
    text_batch = batch_collate_pad(
        text_batch,
        padding=padding,
        padding_side="right",
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Collate for image
    # pad non-image samples with all zeros (black)
    images = torch.empty(0, 3, _IMG_SIZE, _IMG_SIZE)
    for data in batch:
        if data["image"] is None:
            images = torch.cat([images, torch.zeros(1, 3, _IMG_SIZE, _IMG_SIZE)], axis=0)
        else:
            images = torch.cat([images, data["image"]], axis=0)

    return (images, text_batch)


def seed_worker(worker_id):
    """
    Copied and slightly modified from https://github.com/Lightning-AI/lightning/blob/984f49f7195ddc67e961c7c498ee6e19fc0cecb5/src/lightning/fabric/utilities/seed.py#L81-L104
    Helper function to set worker seed during Dataloader initialization.
    """
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    global_rank = torch.distributed.get_rank()
    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # use 128 bits (4 x 32-bit words)
    np_seed = ss.generate_state(4)
    np.random.seed(np_seed)
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)