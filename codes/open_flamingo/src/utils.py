import torch


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    )  # mixin needs to go first for our forward() logic to work


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


def apply_with_stopping_condition(
    module, apply_fn, apply_condition=None, stopping_condition=None, **other_args
):
    if stopping_condition(module):
        return
    if apply_condition(module):
        apply_fn(module, **other_args)
    for child in module.children():
        apply_with_stopping_condition(
            child,
            apply_fn,
            apply_condition=apply_condition,
            stopping_condition=stopping_condition,
            **other_args
        )


def num_params(module, filter_to_trainable=False):
    """Returns the number of parameters in the module, or optionally only the trainable parameters"""
    if filter_to_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


def stack_with_padding(list_of_tensors, padding_value=0, padding_side="right"):
    """
    Stack a list of tensors with padding on one side
    Args:
        list_of_tensors (list[torch.Tensor]): List of tensors to stack
        padding_value (int, optional): Value to pad with. Defaults to 0.
        padding_side (str, optional): Side to pad on. Defaults to "right".
    Returns:
        torch.Tensor: Stacked tensors
    """
    max_tokens = max(tensor.size(0) for tensor in list_of_tensors)
    padded_tensors = []
    for tensor in list_of_tensors:
        num_tokens = tensor.size(0)
        if len(tensor.size()) == 1:
            padding = torch.full(
                (max_tokens - num_tokens,),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            padding = torch.full(
                (max_tokens - num_tokens, tensor.size(1)),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        padded_tensor = (
            torch.cat((tensor, padding), dim=0)
            if padding_side == "right"
            else torch.cat((padding, tensor), dim=0)
        )
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors)


def stack_with_padding_2D_attention(list_of_tensors):
    max_size = max(tensor.size(1) for tensor in list_of_tensors)
    # Initialize a padded tensor of zeros with the target shape
    padded_tensors = []
    for tensor in list_of_tensors:
        a = tensor.shape[-1]
        padding = (0, max_size - a, 0, max_size - a)  # (left, right, top, bottom)
        padded_tensor = torch.nn.functional.pad(tensor, padding)
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors)