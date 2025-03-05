import time
from contextlib import suppress
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os
import glob
import logging
from einops import rearrange

# Set up basic configuration for logging
logger = logging.getLogger(__name__)


# modify from utils.py since each element in the list is a batch
def stack_with_batch_padding(list_of_tensors, padding_value=0, padding_side="right"):
    """
    Stack a list of tensors with padding on one side
    Args:
        list_of_tensors (list[torch.Tensor]): List of tensors to stack
        padding_value (int, optional): Value to pad with. Defaults to 0.
        padding_side (str, optional): Side to pad on. Defaults to "right".
    Returns:
        torch.Tensor: Stacked tensors
    """
    max_tokens = max(tensor.size(1) for tensor in list_of_tensors)
    padded_tensors = []
    for tensor in list_of_tensors:
        # dim 0 is batch
        num_tokens = tensor.size(1)
        padding = torch.full(
            (tensor.size(0), max_tokens - num_tokens),
            padding_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded_tensor = (
            torch.cat((tensor, padding), dim=1)
            if padding_side == "right"
            else torch.cat((padding, tensor), dim=1)
        )
        padded_tensors.append(padded_tensor)
    return torch.cat(padded_tensors, dim=0)


def get_mp_policy_dtype(precision: str):
    if "bfloat16" in precision or "bf16" in precision:
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    else:
        return torch.float32


def get_autocast(precision, cache_enabled=True):
    if precision == "amp":
        return torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.amp.autocast(
            'cuda', dtype=torch.bfloat16, cache_enabled=cache_enabled
        )
    else:
        return suppress


def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model
    

def finetune_one_epoch(
    args,
    model,
    epoch,
    datasets,
    compute_loss_fn: callable,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    previous_step,
    writer=None,
    total_training_steps=None,
):
    # setup loaders
    num_batches_per_epoch = len(datasets.dataloader)
    if total_training_steps is None:
        total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )  # if fsdp, disable cache to save memory

    # setup model
    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # loop through dataloader
    for num_steps, batch in tqdm(
        enumerate(datasets.dataloader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        if num_steps < previous_step:
            continue

        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        images = batch[0].to(device_id, non_blocking=True)
        input_ids = batch[1]["input_ids"].to(device_id, non_blocking=True)
        attention_mask = batch[1]["attention_mask"].to(
            device_id, non_blocking=True
        )
        labels = batch[1]["labels"].to(device_id, non_blocking=True)

        ### for single pair
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)

        # forward pass
        dataset_loss = compute_loss_fn(
            model=model,
            tokenizer=tokenizer,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            autocast=autocast
        )

        dataset_loss = dataset_loss / args.gradient_accumulation_steps
        dataset_loss.backward()

        # clip gradient norm
        if args.fsdp:
            model.clip_grad_norm_(1.0, norm_type=2.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            if ((num_steps + 1) % 1000 == 0) and args.rank == 0:
                logger.info(
                    f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. || Loss: {dataset_loss.item():.3f}"
                )

            # rank 0 logging
            if args.rank == 0 and args.report_to_tensorboard:
                writer.add_scalar('training_loss', dataset_loss, global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], global_step)
                step_time_m.reset()
                data_time_m.reset()

        if ((num_steps + 1) % args.checkpoint_steps == 0):
            save_checkpoint(model, optimizer, lr_scheduler, epoch, args, step=num_steps)

        if num_steps > total_training_steps:
            break


def train_one_epoch(
    args,
    model,
    epoch,
    datasets,
    compute_loss_fn: callable,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    previous_step,
    writer=None,
):
    # setup loaders
    num_batches_per_epoch = datasets[0].dataloader.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )  # if fsdp, disable cache to save memory

    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # loop through dataloader
    for num_steps, batches in tqdm(
        enumerate(zip(*[dataset.dataloader for dataset in datasets])),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        # if num_steps < previous_step:
        #     continue

        for dataset_ix, batch in enumerate(batches):
            images = batch[0].to(device_id, non_blocking=True)
            input_ids = batch[1][0].to(device_id, non_blocking=True)
            attention_mask = batch[1][1].to(
                device_id, non_blocking=True
            )

            ### for single pair
            images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)

            ### for interleaved
            # images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1)       
            # input_ids = torch.stack([x[0] for x in batch[1]]).squeeze(1)
            # attention_mask = torch.stack([x[1] for x in batch[1]]).squeeze(1)

            # forward pass
            dataset_loss = compute_loss_fn(
                model=model,
                tokenizer=tokenizer,
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                autocast=autocast
            )

            dataset_loss = dataset_loss / args.gradient_accumulation_steps
            dataset_loss.backward()

        # clip gradient norm
        if args.fsdp:
            model.clip_grad_norm_(1.0, norm_type=2.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            if ((num_steps + 1) % 1000 == 0) and args.rank == 0:
                print(
                    f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. || Loss: {dataset_loss.item():.3f}"
                )

            # rank 0 logging
            if args.rank == 0 and args.report_to_tensorboard:
                writer.add_scalar('training_loss', dataset_loss, global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], global_step)
                step_time_m.reset()
                data_time_m.reset()

        if ((num_steps + 1) % args.checkpoint_steps == 0):
            save_checkpoint(model, optimizer, lr_scheduler, epoch, args, step=num_steps)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


####################################################
# Helper functions for checkpoint loading / saving #
####################################################


def find_most_recent_checkpoint(args):
    """
    Returns the path of the most recent checkpoint for a given run name.
    """
    checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
    if len(checkpoint_list) == 0:
        print(f"Found no checkpoints for run {args.run_name}.")
        resume_from_checkpoint = None
    else:
        resume_from_checkpoint = sorted(
            checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )[-1]
        print(f"Found checkpoint {resume_from_checkpoint} for run {args.run_name}.")
    return resume_from_checkpoint


def load_checkpoint(args, model, pretrained=False):
    """
    Loads a checkpoint into the model and returns the checkpoint + epoch to resume from.
    Does not load the optimizer or learning rate checkpoints, but these are included in the returned checkpoint dict.
    """
    if pretrained:
        ckpt_path = args.pretrained
    else:
        ckpt_path = args.resume_from_checkpoint

    if args.rank == 0:
        print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    msd = checkpoint.pop("model_state_dict")
    msd = {k.replace("module.", ""): v for k, v in msd.items()}

    if 'vision_tokenizer.latents' in msd.keys():
        msd_current = model.state_dict()
        if msd_current['vision_tokenizer.latents'].shape != msd['vision_tokenizer.latents'].shape:
            msd["vision_tokenizer.latents"] = msd_current['vision_tokenizer.latents'] # Random re-init.

    if not pretrained:
        resume_from_epoch = checkpoint["epoch"] + 1
    else:
        resume_from_epoch = None
    
    if 'step' in checkpoint and checkpoint["step"] is not None:
        resume_from_step = checkpoint["step"] + 1
        resume_from_epoch = checkpoint["epoch"] # Resume from prev epoch at the given step.
    else:
        resume_from_step = 0

    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            **args.fsdp_checkpoint_config,
        )
    result = model.load_state_dict(msd, strict=False)
    # # Print missing and unexpected keys
    # if args.rank == 0:
    #     print("Missing keys:", result.missing_keys)
    #     print("Unexpected keys:", result.unexpected_keys)
    
    return resume_from_epoch, resume_from_step, checkpoint


def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This is because we need the new <image> <|endofchunk|> tokens to
    be consistent across initializations.
    """
    for (
        name,
        p,
    ) in model.named_parameters():  # won't work for fsdp + use_orig_params=False
        if "fsdp" in name:
            continue
        if "embed" in name or isinstance(p, torch.nn.Embedding):
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")

    # also remove the keys in state_dict generated from
    # lang_encoder.old_decoder_blocks and lang_encoder.gated_cross_attn_layers
    # because these are already saved in lang_encoder.model...
    to_delete = [
        n
        for n in state_dict.keys()
        if ("lang_encoder.old_decoder_blocks" in n)
        or ("lang_encoder.gated_cross_attn_layers" in n)
        or ("vision_encoder" in n)
    ]
    for name in to_delete:
        del state_dict[name]
    return state_dict


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args, step=None):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """
    torch.cuda.empty_cache() # (Sometimes this is necessary to avoid OOM errors when saving checkpoints)

    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            **args.fsdp_checkpoint_config,
        )
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if args.rank == 0:
        model_state = filter_state_dict_to_trainable(model, model_state)

        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

        checkpoint_dict = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }

        if step is not None:
            save_name = f"{args.run_name}/checkpoint_{epoch}_{step}.pt"
        else:
            save_name = f"{args.run_name}/checkpoint_{epoch}.pt"
        print(f"Saving checkpoint to {save_name}")
        torch.save(checkpoint_dict, save_name)

        if args.delete_previous_checkpoint:
            if epoch > 0:
                os.remove(f"{args.run_name}/checkpoint_{epoch-1}.pt")
            else:
                checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
                if len(checkpoint_list) > 1:
                    last_checkpoint = sorted(
                        checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
                    )[0]
                    os.remove(f"{last_checkpoint}")
    torch.distributed.barrier()