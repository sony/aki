""" Main training script """

import argparse
import os
import random
import functools
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from data import get_data, SUPPORTED_DATASETS
from distributed import (
    init_distributed_device, 
    world_info_from_env,
    get_fsdp_config,
    get_fsdp_checkpoint_config,
)
from losses import (
    SUPPORTED_LOSSES,
    get_loss_fn,
    get_cosine_schedule_with_warmup
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from train_utils import (
    train_one_epoch,
    save_checkpoint,
    find_most_recent_checkpoint,
    load_checkpoint,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from open_flamingo import create_model_and_transforms

import logging

# Set up basic configuration for logging
logging.basicConfig()
# logging.getLogger().setLevel(logging.ERROR)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int)

    # model configuration args
    parser.add_argument(
        "--model_family", default="AKI", type=str
    )
    parser.add_argument("--vision_encoder_path", default="ViT-SO400M-14-SigLIP-384", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="webli", type=str)
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-30b",
        type=str,
        help="path to tokenizer",
    )

    # training args
    parser.add_argument(
        "--loss", type=str, choices=SUPPORTED_LOSSES, default="next_token_prediction"
    )
    parser.add_argument(
        "--min_lr", default=1e-5, type=float
    )
    parser.add_argument(
        "--checkpoint_steps", type=int, default=1000, help="save model every n steps"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="openflamingo3B",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states. if there exists a checkpoint in the dir named run_name, we will resume from that checkpoint by default",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )

    # data args
    for dataset_name in SUPPORTED_DATASETS:
        parser.add_argument(f"--batch_size_{dataset_name}", type=int, help="Batch per GPU", default=128)
        parser.add_argument(
            f"--loss_multiplier_{dataset_name}", type=float, default=1.0
        )
        parser.add_argument(
            f"--train_num_samples_{dataset_name}",
            type=int,
            default=10000,
            help="Number of samples in an 'epoch' for this dataset. Note that train_num_samples/batch_size must be the same for all datasets.",
        )
        parser.add_argument(
            f"--{dataset_name}_shards",
            type=str,
            default=None,
            help="Should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar. If None, we will not train on this dataset.",
        )

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="whether to train with gradient/activation checkpointing",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="we define an 'epoch' as a fixed number of examples (train_num_samples_mmc4, train_num_samples_laion), not a pass through the entire dataset",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--freeze_lm_embeddings",
        action="store_true",
        help="if True, we freeze the LM embeddings during training. Otherwise, we train the <image> and <|endofchunk|> embeddings.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )

    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset_resampled", action="store_true")

    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        "--fsdp",
        default=False,
        action="store_true",
        help="Use FullyShardedDataParallel for distributed training.",
    )
    parser.add_argument(
        "--fsdp_use_orig_params",
        default=False,
        action="store_true",
        help="Passed into the FSDP constructor. Enables param_groups and gradient masking for weight_decay. Does not work with OPT.",
    )
    parser.add_argument(
        "--fsdp_sharding_strategy", default="full", type=str, choices=["full", "hybrid"]
    )

    # tensorboard args
    parser.add_argument(
        "--report_to_tensorboard", default=False, action="store_true"
    )
    parser.add_argument(
        "--log_dir", default="logs", type=str, help="directory to save logs"
    )

    parser.add_argument(
        '--cpu_offload_gradients',
        default=False,  action='store_true',
        help='This specifies whether to offload parameters to CPU when not involved in computation. If True, then this offloads gradients to CPU as well, meaning that the optimizer step runs on CPU.'
    )

    args = parser.parse_args()

    # Parse which datasets to train on and which to exclude
    datasets_to_train_on = []
    for dataset_name in SUPPORTED_DATASETS:
        datasets_to_train_on.append(dataset_name)
        if getattr(args, f"{dataset_name}_shards") is None:
            print(f"Excluding {dataset_name} from training")
            setattr(args, f"train_num_samples_{dataset_name}", 0)
            setattr(args, f"batch_size_{dataset_name}", 0)
        else:
            datasets_to_train_on.append(dataset_name)
            shards_path = getattr(args, f"{dataset_name}_shards")
            if shards_path.startswith("s3"):
                setattr(
                    args,
                    f"{dataset_name}_shards",
                    f"pipe:aws s3 cp {shards_path} -",
                )
    assert len(datasets_to_train_on) > 0, "Must train on at least one dataset"

    # Validate args
    for i in range(len(datasets_to_train_on) - 1):
        current_dataset_number_batches = getattr(args, f"train_num_samples_{datasets_to_train_on[i]}") // getattr(
            args, f"batch_size_{datasets_to_train_on[i]}"
        )
        next_dataset_number_batches = getattr(args, f"train_num_samples_{datasets_to_train_on[i+1]}") // getattr(
            args, f"batch_size_{datasets_to_train_on[i + 1]}"
        )
        assert current_dataset_number_batches == next_dataset_number_batches, f'Number of batches in each dataloader must be the same, but have {current_dataset_number_batches} and {next_dataset_number_batches}'

    # Set up distributed training
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    if args.rank == 0:
        print(f"Initializing distributed training with {args.world_size} GPUs.")
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device_id = init_distributed_device(args)
    random_seed(args.seed)

    # Initialize model
    additional_kwargs = (
        {"cross_attn_every_n_layers": args.cross_attn_every_n_layers}
        if args.model_family == "flamingo"
        else {}
    )
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=args.vision_encoder_path,
        clip_vision_encoder_pretrained=args.vision_encoder_pretrained,
        lang_encoder_path=args.lm_path,
        tokenizer_path=args.tokenizer_path if args.tokenizer_path else args.lm_path,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        verbose=(args.rank == 0),
        **additional_kwargs,
    )
    random_seed(args.seed, args.rank)

    # set dtype of model
    if args.precision == "bf16":
        model.to(torch.bfloat16)
    else:
        pass
        # model.to(torch.float16)

    # Initialize logging
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_tensorboard:
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    # Load model checkpoint (on CPU)
    if args.fsdp:
        args.fsdp_checkpoint_config = get_fsdp_checkpoint_config(args)

    # if args do not specify a checkpoint to resume from, resume from most recent checkpoint
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        args.resume_from_checkpoint = find_most_recent_checkpoint(args)

    if (
        args.resume_from_checkpoint is not None
    ): 
        resume_from_epoch, resume_from_step, checkpoint = load_checkpoint(args, model)
        if args.rank == 0:
            print(f"Resuming from epoch {resume_from_epoch}, step {resume_from_step}")
    else:
        resume_from_epoch, resume_from_step = 0, 0

    # Initialize FSDP / DDP, and ensure the model is on GPU
    print(f"Initializing distributed training with {args.world_size} GPUs.")
    if args.fsdp:
        auto_wrap_policy = functools.partial(
            lambda_auto_wrap_policy, lambda_fn=model.get_fsdp_lambda_fn()
        )
        wrapper_kwargs = get_fsdp_config(args, device_id)
        ddp_model = FSDP(
            model, auto_wrap_policy=auto_wrap_policy, **wrapper_kwargs
        )
    else:
        model = model.to(device_id)
        ddp_model = DDP(model, device_ids=[device_id])

    # Initialize gradient checkpointing
    if args.gradient_checkpointing:
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            offload_to_cpu=True,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            ddp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
            and not isinstance(m, FSDP)
            and not isinstance(m, CheckpointWrapper),
        )

    # Initialize optimizer
    params_with_wd, params_without_wd = ddp_model.group_params_by_weight_decay()
    optimizer = torch.optim.AdamW(
        [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )

    # load optimizer checkpoint
    if args.resume_from_checkpoint is not None:
        optim_state_dict = checkpoint["optimizer_state_dict"]
        if args.fsdp:
            optim_state_dict = FSDP.optim_state_dict_to_load(optim_state_dict=optim_state_dict, model=ddp_model, optim=optimizer)
        optimizer.load_state_dict(optim_state_dict)

    # Initialize data loaders
    datasets = [
        get_data(args, image_processor, tokenizer, dataset_name)
        for dataset_name in datasets_to_train_on
    ]
    
    total_training_steps = (
        getattr(args, f"train_num_samples_{datasets_to_train_on[0]}")
        // (getattr(args, f"batch_size_{datasets_to_train_on[0]}") * args.world_size)
    ) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    # Initialize lr scheduler
    if args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            lr=args.learning_rate,
            min_lr=args.min_lr,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        raise NotImplementedError("Only cosine lr scheduler is supported")

    # load lr scheduler checkpoint
    if args.resume_from_checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # Initialize the loss fn
    loss_fn = get_loss_fn(args.loss)

    # Start training!
    ddp_model.train()

    for epoch in range(resume_from_epoch, args.num_epochs):
        for dataset in datasets:
            dataset.set_epoch(epoch)

        train_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            datasets=datasets,
            compute_loss_fn=loss_fn,
            device_id=device_id,
            writer=writer,
            previous_step=resume_from_step,
        )
        save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)

    # save final checkpoint
    save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)

    # Close TensorBoard writer
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
