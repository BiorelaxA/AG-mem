"""Single-process decoder layer placement for Reader training and generation.

This uses ordinary differentiable device transfers, not inference dispatch or
replicated data parallelism. Checkpoint parameter names and optimizer ordering
are unchanged, allowing existing single-GPU checkpoints to resume.
"""
from functools import partial


def _move_tensors(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    # DynamicCache is shared by all decoder layers. Each layer initializes and
    # updates its own cache on its own device; never move the whole cache.
    return value


def _inputs_on_device(module, args, kwargs, *, device):
    return _move_tensors(args, device), _move_tensors(kwargs, device)


def place_reader_layers(model, devices):
    """Keep embeddings/head on the first device; split whole decoder layers.

    Hooks also run during gradient-checkpoint recomputation. The final norm
    moves the hidden states back to the first device for tied embeddings/head,
    generation bookkeeping, targets, and loss calculation.
    """
    import torch
    devices = [torch.device(device) for device in devices]
    text = model.model.language_model
    if not devices or len(text.layers) < len(devices):
        raise ValueError("Reader placement needs at least one decoder layer per device")
    if getattr(model, "_agmem_placement_hooks", None):
        raise ValueError("Reader placement is already installed")
    # Models arrive on CPU. Place each subtree directly, avoiding a temporary
    # full-model allocation on the first GPU.
    for name, child in model.named_children():
        if name != "model":
            child.to(devices[0])
    for name, child in model.model.named_children():
        if name != "language_model":
            child.to(devices[0])
    for name, child in text.named_children():
        if name != "layers":
            child.to(devices[0])
    hooks, layer_devices = [], []
    for index, layer in enumerate(text.layers):
        device = devices[index * len(devices) // len(text.layers)]
        layer.to(device)
        hooks.append(layer.register_forward_pre_hook(
            partial(_inputs_on_device, device=device), with_kwargs=True))
        layer_devices.append(str(device))
    hooks.append(text.norm.register_forward_pre_hook(
        partial(_inputs_on_device, device=devices[0]), with_kwargs=True))
    model.tie_weights()
    model._agmem_placement_hooks = hooks
    model._agmem_layer_devices = layer_devices
    return model


def reader_devices(device, environ):
    """Placement is a runtime option, separate from training hyperparameters."""
    import torch
    device = torch.device(device)
    count = int(environ.get("AGMEM_READER_MODEL_PARALLEL_SIZE", "1"))
    if count not in (1, 2):
        raise ValueError("AGMEM_READER_MODEL_PARALLEL_SIZE must be 1 or 2")
    # The frozen reference is explicitly loaded on CPU and managed separately.
    if count == 1 or device.type != "cuda":
        return [device]
    if device.index not in (None, 0) or torch.cuda.device_count() != count:
        raise ValueError("Two-GPU Reader requires exactly two visible GPUs and device cuda:0")
    return [torch.device("cuda", index) for index in range(count)]
