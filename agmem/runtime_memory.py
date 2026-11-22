"""Runtime memory controls, independent of frozen training/cache configuration."""
from contextlib import nullcontext
import os


def configure_cuda_allocator(environ=None):
    """Call before importing torch; preserve explicit modern or legacy settings."""
    environ = os.environ if environ is None else environ
    if "PYTORCH_ALLOC_CONF" not in environ and "PYTORCH_CUDA_ALLOC_CONF" not in environ:
        environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    return environ.get("PYTORCH_ALLOC_CONF", environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))


def memory_settings(environ=None):
    environ = os.environ if environ is None else environ
    value = environ.get("AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS", "16384")
    try:
        threshold = int(value)
    except (TypeError, ValueError):
        raise ValueError("AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS must be a nonnegative integer") from None
    if threshold < 0:
        raise ValueError("AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS must be a nonnegative integer")
    return {"activation_offload_min_tokens": threshold,
            # Pageable host storage avoids pinning several GB alongside CPU Adam.
            "activation_offload_pin_memory": False,
            "allocator_config": environ.get("PYTORCH_ALLOC_CONF", environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))}


def use_activation_offload(input_tokens, device, settings):
    threshold = settings["activation_offload_min_tokens"]
    return str(device).split(":")[0] == "cuda" and threshold > 0 and input_tokens >= threshold


def activation_offload_context(input_tokens, device, settings):
    if not use_activation_offload(input_tokens, device, settings):
        return nullcontext()
    import torch
    # Autograd restores each saved tensor to its ORIGINAL device. This is
    # essential for Reader layers split across cuda:0 and cuda:1. It also
    # offloads saved inputs at non-reentrant checkpoint boundaries; temporary
    # tensors required by recomputation/backward still occupy GPU memory.
    return torch.autograd.graph.save_on_cpu(pin_memory=False, device_type="cuda")
