from __future__ import annotations

from pathlib import Path
import inspect
import json

from .core import CapacityError, Completion, digest, file_digest


def model_fingerprint(path: str | Path, full_weights: bool = False) -> dict:
    root = Path(path)
    files = {}
    for item in sorted(root.iterdir()):
        if item.is_file() and (item.suffix == ".json" or item.suffix == ".safetensors"):
            stat = item.stat()
            files[item.name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if item.suffix == ".json" or full_weights:
                files[item.name]["sha256"] = file_digest(item)
    if "config.json" not in files or not any(name.endswith(".safetensors") for name in files):
        raise ValueError(f"Incomplete local model: {root}")
    return {"fingerprint": digest(files), "full_weight_hashes": full_weights, "files": files}


def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def chat_token_ids(tokenizer, messages) -> list[int]:
    """Request one flat ID sequence explicitly; Transformers 5 defaults to BatchEncoding."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_dict=False, return_tensors=None,
                                        padding=False, truncation=False)
    if not isinstance(ids, list) or not ids or any(type(token) is not int for token in ids):
        raise TypeError("Chat tokenization must return a nonempty flat list of integer input_ids")
    return ids


def load_model(path, device, dtype="bfloat16", checkpointing=False):
    import torch
    try:
        from transformers import Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("Use the provided new Conda environment: current Transformers lacks Qwen3.5") from exc
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                          dtype=getattr(torch, dtype), attn_implementation="sdpa")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model.to(device).eval()


def sequence_logprobs(model, prompt_ids, token_ids, temperature: float, device):
    """Only completion targets receive loss; causal logits use each token's true prompt."""
    import torch
    import torch.nn.functional as F
    if not prompt_ids or not token_ids or temperature <= 0:
        raise ValueError("Nonempty prompt/completion and positive behavior temperature required")
    # The final sampled token is only a target, never an input needed to score this completion.
    ids = torch.tensor([prompt_ids + token_ids[:-1]], dtype=torch.long, device=device)
    n = len(token_ids)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=n)
    logits = output.logits[:, -n:, :].float() / temperature
    targets = torch.tensor(token_ids, dtype=torch.long, device=device)
    return -F.cross_entropy(logits[0], targets, reduction="none")


class HFPolicy:
    def __init__(self, model, tokenizer, config: dict, temperature=None):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.device = next(model.parameters()).device
        self.temperature = config["rollout_temperature"] if temperature is None else temperature
        self.policy_version = 0

    def encode(self, messages):
        return chat_token_ids(self.tokenizer, messages)

    def check_capacity(self, messages):
        ids = self.encode(messages)
        limit = self.config.get("max_context_tokens")
        if limit is None:
            limit = self.model.config.text_config.max_position_embeddings
        if len(ids) + self.config["max_new_tokens_per_decision"] > limit:
            raise CapacityError(f"Reader requires {len(ids)} prompt tokens plus output; limit={limit}. No truncation performed")

    def complete(self, messages, seed: int):
        import torch
        self.check_capacity(messages)
        prompt_ids = self.encode(messages)
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        self.model.eval()
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            kwargs = dict(max_new_tokens=self.config["max_new_tokens_per_decision"], do_sample=self.temperature > 0,
                          top_p=1.0, top_k=0, repetition_penalty=1.0, use_cache=self.config.get("rollout_use_cache", True),
                          return_dict_in_generate=True, output_scores=self.temperature > 0,
                          pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
            if self.temperature > 0:
                kwargs["temperature"] = self.temperature
            with torch.inference_mode():
                output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs)
                generated = output.sequences[0, len(prompt_ids):].tolist()
                logprobs = []
                if self.temperature > 0:
                    if len(output.scores) != len(generated):
                        raise ValueError("Generation scores do not align with generated tokens")
                    # These are the actual processed behavior logits used during sampling,
                    # not a post-hoc assumption about the sampling distribution.
                    logprobs = [torch.log_softmax(score[0].float(), -1)[token].item() for score, token in zip(output.scores, generated)]
        return Completion(self.tokenizer.decode(generated, skip_special_tokens=True), prompt_ids, generated, logprobs)
