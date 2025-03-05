""" Main training script """

import os
import random
import hydra

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
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
    finetune_one_epoch,
    save_checkpoint,
    find_most_recent_checkpoint,
    load_checkpoint,
)
from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from types import SimpleNamespace
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
import functools

from open_flamingo import create_model_and_transforms
from sft_data_utils import dataset_provider

import logging

# Set up basic configuration for logging
logger = logging.getLogger(__name__)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


@hydra.main(version_base=None, config_path="../configs", config_name="sft")
def main(args: DictConfig):
    OmegaConf.resolve(args)
    args = SimpleNamespace(**args)

    # change the saved path to the same as Hydra
    hydra_runtime_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    args.run_name = os.path.join(hydra_runtime_dir, args.run_name)

    if args.training_mode == "sft" and args.resume_from_checkpoint is None:
        logger.info("===Warning: training mode is set to sft but no checkpoint is provided to resume from. Training will start from scratch.===")

    # Set up distributed training
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    if args.rank == 0:
        logger.info(f"Initializing distributed training with {args.world_size} GPUs.")
    device_id = init_distributed_device(args)
    random_seed(args.seed)

    # Initialize model
    additional_kwargs = (
        {}
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

    # Initialize logging
    logger.info(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_tensorboard:
        writer = SummaryWriter(log_dir=f"{hydra_runtime_dir}/{args.log_dir}")
    else:
        writer = None

    # Load model checkpoint (on CPU)
    if args.fsdp:
        args.fsdp_checkpoint_config = get_fsdp_checkpoint_config(args)

    # if args do not specify a checkpoint to resume from, resume from most recent checkpoint
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        args.resume_from_checkpoint = find_most_recent_checkpoint(args)

    if (args.resume_from_checkpoint is not None): 
        resume_from_epoch, resume_from_step, checkpoint = load_checkpoint(args, model)
        if args.training_mode == "sft_scratch":
            resume_from_epoch, resume_from_step = 0, 0       # set to 0 since this is the SFT-scratch stage
        if args.rank == 0:
            logger.info(f"Resuming from epoch {resume_from_epoch}, step {resume_from_step}")
    else:
        resume_from_epoch, resume_from_step = 0, 0

    # Initialize FSDP / DDP, and ensure the model is on GPU
    logger.info(f"Initializing distributed training with {args.world_size} GPUs.")
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
    if args.resume_from_checkpoint is not None and args.training_mode == "sft_resume":
        optim_state_dict = checkpoint["optimizer_state_dict"]
        if args.fsdp:
            optim_state_dict = FSDP.optim_state_dict_to_load(optim_state_dict=optim_state_dict, model=ddp_model, optim=optimizer)
        optimizer.load_state_dict(optim_state_dict)

    # Prepare SFT datasets and initialize data loaders
    mixed_dataset = dataset_provider(data_config=args.sft_datasets, training_config=args.training_config, tokenizer=tokenizer, processor=image_processor, num_gpus=args.world_size)

    if args.training_config.total_training_steps != -1:
        total_training_steps = args.training_config.total_training_steps
    else:
        total_training_steps = mixed_dataset.dataloader.num_samples

    if args.rank == 0:
        logger.info(f"Total training steps: {total_training_steps}")

    # Initialize lr scheduler
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            lr=args.learning_rate,
            min_lr=args.min_lr,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    # load lr scheduler checkpoint
    if args.resume_from_checkpoint is not None and args.training_mode == "sft_resume":
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # Initialize the loss fn
    loss_fn = get_loss_fn(args.loss)

    # Start training!
    ddp_model.train()

    for epoch in range(resume_from_epoch, args.num_epochs):
        mixed_dataset.set_epoch(epoch)

        finetune_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            datasets=mixed_dataset,
            compute_loss_fn=loss_fn,
            device_id=device_id,
            writer=writer,
            previous_step=resume_from_step,
            total_training_steps=total_training_steps if args.training_config.total_training_steps != -1 else None,
        )
        save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)

    # save final checkpoint
    save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)

    # Close TensorBoard writer
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
