import torch
import math
from typing import List, Optional
from torch.optim.lr_scheduler import LambdaLR

SUPPORTED_LOSSES = ["next_token_prediction",
                    "supervised_finetune"]


def get_cosine_schedule_with_warmup(
    optimizer,
    lr,
    min_lr,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """Cosine schedule with warmup & min_lr"""
    delta_min_lr = (lr - min_lr) / lr

    def cvt_mult_with_minlr(mult):
        """Convert multiplier considering min_lr"""
        return (1 - delta_min_lr) + delta_min_lr * mult

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Do not consider min_lr when warmup
            progress = float(current_step) / float(max(1, num_warmup_steps))
            return cvt_mult_with_minlr(progress)

        # [0, 1] progress
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # [0, 1] cosine multiplier
        cos_mult = max(0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return cvt_mult_with_minlr(cos_mult)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_loss_fn(loss_name):
    if loss_name == "next_token_prediction":
        return NextTokenPrediction()
    elif loss_name == "supervised_finetune":
        return SupervisedPrediction()
    else:
        raise ValueError(
            f"Loss {loss_name} not supported. Supported losses: {SUPPORTED_LOSSES}"
        )

class Loss:
    @property
    def name(self):
        raise NotImplementedError

    def __call__(
        self,
        model,
        tokenizer,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        autocast: callable,
    ):
        """
        Args:
            model: VLM model
            images: images tensor, already moved to device and cast to appropriate dtype
                shape (B, T_img, F, C, H, W)
            input_ids: input ids tensor, already moved to device and cast to appropriate dtype
                shape (B, T_text)
            attention_mask: attention mask tensor, already moved to device and cast to appropriate dtype
                shape (B, T_text)
            autocast: autocast context manager
        Return:
            loss: scalar loss
        """
        raise NotImplementedError


class NextTokenPrediction(Loss):
    @property
    def name(self):
        return "next_token_prediction"

    def __call__(
        self,
        model,
        tokenizer,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        autocast: callable,
    ):
        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        # special_token_ids = torch.Tensor(unwrap_model(model).special_token_ids).to(
        #     labels.device
        # )
        # labels[torch.isin(labels, special_token_ids)] = -100

        labels = labels.to(input_ids.device)

        # call forward
        with autocast():
            loss = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
        return loss


class SupervisedPrediction(Loss):
    @property
    def name(self):
        return "supervised_finetune"

    def __call__(
        self,
        model,
        tokenizer,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        autocast: callable,
        image_size: Optional[torch.Tensor] = None,
    ):
        # set up labels; language model is expected to handle shifting
        labels[labels == tokenizer.pad_token_id] = -100
        special_token_ids = torch.Tensor(unwrap_model(model).special_token_ids).to(
            labels.device
        )
        labels[torch.isin(labels, special_token_ids)] = -100 # TODO: dont want to remove loss on <|endofchunk|> tokens

        # call forward
        with autocast():
            loss = model(
                vision_x=images,
                image_size=image_size,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
        return loss


def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    ):
        return model.module
    else:
        return model
