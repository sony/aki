"""
Preprocess and load datasets for training.
"""

import functools
import io
import json
import math
import re
import random
import numpy as np
import torch
import torchvision
import webdataset as wds
from PIL import Image
import base64
from scipy.optimize import linear_sum_assignment
from ast import literal_eval

from data_utils import *

Image.MAX_IMAGE_PIXELS = 1000000000
N_CHANNELS = 3
MIN_KB = 3
_SHARD_SHUFFLE_SIZE = 20
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000
INTERLEAVED_DATASETS = [
    # "mmc4"
]
SINGLE_PAIR_DATASETS = [
    # "cc3m", 
    # "cc12m", 
    # "blip3_grounding_50m", 
    # "blip3_ocr_200m",
    "blip3_kale"
]
SUPPORTED_DATASETS = INTERLEAVED_DATASETS + SINGLE_PAIR_DATASETS

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def preprocess_image(sample, image_processor):
    """
    Convert images to tensors for training.
    Augmentations: random horizontal flip.
    Normalization handled by wds.
    """
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    return image


def filter_no_caption_or_no_image(sample):
    """
    Filter out LAION samples with no caption or no image.
    """
    return ("txt" in sample) and (
        "png" in sample or "jpg" in sample or "jpeg" in sample
    )


def preprocess_paired_text(sample, tokenizer, dataset_name, max_tokens=128):
    """
    Preprocess text for paired data.
    Captions are truncated to 64 tokens by default.
    """
    tokenizer.padding_side = "right"

    # Due to the composition of blip3_grounding_50m and blip3_ocr_200m, we need to manually select the corret information
    # -1 means the most fine-grained grounding information, but select 0 to provide rough locations
    # 0: caption 1: grangularity level (0-2) 2: include_datacomp_raw_cap
    # blip3_ocr_200m somehow cannot be parsed possibly due to the non-standard format
    if dataset_name == "cc3m" or dataset_name == "cc12m" or dataset_name == "blip3_kale":
        sample = [
            (f"<image>{s.strip()}<|endofchunk|>") for s in sample
        ]
    elif dataset_name == "blip3_grounding_50m":
        sample = [
            (f"<image>{literal_eval(s)[0][0].strip()}<|endofchunk|>") for s in sample
        ]
    elif dataset_name == "blip3_ocr_200m":
        sample = [
            (f"<image>{json.loads(s)[1]['text'].strip()}<|endofchunk|>") for s in sample
        ]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")

    text = tokenizer(
        sample,
        max_length=max_tokens,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"]


def preprocess_interleaved(
    sample,
    tokenizer,
    clip_processor,
    sim_threshold,
    min_num_images,
    max_num_images,
    max_tokens=256,
):
    """
    Preprocess an interleaved image-text sequence, either by calling preprocess_gpt_interleaved (if the sequence
    is ChatGPT-generated) or by preprocessing in this function (if the sequences is from MMC4).
    """
    info = json.loads(sample[0])
    if "is_gpt" in info:
        return preprocess_gpt_interleaved(
            info, tokenizer, clip_processor, min_num_images, max_num_images, max_tokens
        )

    sentences = info["text_list"]
    sim_matrix = info["similarity_matrix"]

    # load images first to find which ones are valid
    valid_images, valid_image_indices = [], []
    for i, sample_image in enumerate(info["image_info"]):
        if "image_base64" not in sample_image:
            continue
        image_base64 = sample_image["image_base64"]
        rawbytes = base64.b64decode(image_base64)

        # filter to images >= 10KB
        if len(rawbytes) // 1000 <= MIN_KB:
            continue

        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
        valid_images.append(image)
        valid_image_indices.append(i)

    if len(valid_image_indices) == 0:
        raise ValueError("No images in sample first")

    sim_matrix = np.array(sim_matrix)  # of shape images x sentences
    sim_matrix = sim_matrix[valid_image_indices]

    # negate the similarities to turn then into costs
    cost_matrix = -sim_matrix
    # find one to one assignements
    image_indices, sentence_indices = linear_sum_assignment(cost_matrix)

    images, sentence_ixs = [], []
    for i, sim_ix in zip(image_indices, sentence_indices):
        sim_score = sim_matrix[i][sim_ix]

        if sim_score < sim_threshold:
            continue

        images.append(valid_images[i])
        sentence_ixs.append(sim_ix)

    if len(images) == 0:
        raise ValueError("No images in sample second")

    # preprocess and pad images
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), max_num_images))
    images_tensors = images_tensors[keep_ixs]
    sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (
                max_num_images - len(images_tensors),
                N_CHANNELS,
                images_tensors[0].shape[1],
                images_tensors[0].shape[2],
            ),
            dtype=torch.float,
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    # add in <image> and <eoc> tokens
    for ix in sentence_ixs:
        sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"
    text = " ".join(sentences)
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )

    # if num_images < min_num_images:
    #     raise ValueError(f"Fewer than {min_num_images} images in sample")
    # elif (
    #     num_images == 1 and random.random() <= 0.5
    # ):  # 50% chance of keeping single image samples
    #     raise ValueError("Only one image in sample")

    # avoid the situation where there's one <image> token and it's at the end
    if (
        num_images == 1
        and text_tensor["input_ids"][:, -1]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    ):
        raise ValueError(
            "Only one image at the end of sample, so labels will all be -100"
        )

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )


def get_interleaved_dataset(args, image_processor, tokenizer, dataset_name, epoch=0, floor=False):
    """
    Initialize webdataset for interleaved sequences
    """
    input_shards = getattr(args, f"{dataset_name}_shards")
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = getattr(args, f"train_num_samples_{dataset_name}")
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    preprocess_fn = functools.partial(
        preprocess_interleaved,
        clip_processor=image_processor,
        tokenizer=tokenizer,
        sim_threshold=args.mmc4_textsim_threshold,
        min_num_images=args.mmc4_min_num_images,
        max_num_images=args.mmc4_max_num_images,
    )

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.to_tuple("json", handler=log_and_continue),
            wds.map(preprocess_fn, handler=log_and_continue),
            wds.batched(getattr(args, f"batch_size_{dataset_name}"), partial=False),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = getattr(args, f"batch_size_{dataset_name}") * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(name=dataset_name, dataloader=dataloader, shared_epoch=shared_epoch)


def get_paired_dataset(args, image_processor, tokenizer, dataset_name, epoch=0, floor=False):
    """
    Initialize webdataset for single text-image-pair data
    """
    input_shards = getattr(args, f"{dataset_name}_shards")
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = getattr(args, f"train_num_samples_{dataset_name}")
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # create two preprocess functions that take in the passed in image_processor and tokenizer
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_paired_text, tokenizer=tokenizer, dataset_name=dataset_name)

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.to_tuple("jpg;png;jpeg", "txt", handler=log_and_continue),
            wds.batched(getattr(args, f"batch_size_{dataset_name}"), partial=False),
            wds.map_tuple(
                preprocess_image_fn, preprocess_text_fn, handler=log_and_continue
            ),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = getattr(args, f"batch_size_{dataset_name}") * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(name=dataset_name, dataloader=dataloader, shared_epoch=shared_epoch)


def get_dataset_fn(dataset_type):
    """
    Helper function to get the dataset function based on the dataset type
    """
    if dataset_type in SINGLE_PAIR_DATASETS:
        return get_paired_dataset
    elif dataset_type in INTERLEAVED_DATASETS:
        return get_interleaved_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, image_processor, tokenizer, dataset_name, epoch=0):
    """
    Interface for getting the webdatasets
    """
    return get_dataset_fn(dataset_name)(
        args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer, dataset_name=dataset_name
    )

    # return get_interleaved_dataset(
    #     args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer
    # )
