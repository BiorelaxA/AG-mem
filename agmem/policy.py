from __future__ import annotations

from pathlib import Path
import inspect
import json

from .core import CapacityError, Completion, digest, file_digest, seeded_int


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
    import os
    import torch
    from .model_parallel import place_reader_layers, reader_devices
    devices = reader_devices(device, os.environ)
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
    if len(devices) > 1:
        model = place_reader_layers(model, devices)
        print(json.dumps({"event": "reader_model_parallel", "devices": list(map(str, devices)),
                          "layer_devices": model._agmem_layer_devices}), flush=True)
        return model.eval()
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
        self._check_ids_capacity(self.encode(messages))

    def _check_ids_capacity(self, ids):
        limit = self.config.get("max_context_tokens")
        if limit is None:
            limit = self.model.config.text_config.max_position_embeddings
        if len(ids) + self.config["max_new_tokens_per_decision"] > limit:
            raise CapacityError(f"Reader requires {len(ids)} prompt tokens plus output; limit={limit}. No truncation performed")

    def complete(self, messages, seed: int):
        prompt_ids = self.encode(messages)
        self._check_ids_capacity(prompt_ids)
        return self._generate([prompt_ids], [seed])[0]

    def complete_batch(self, messages, seeds, max_batch_tokens=16384):
        if len(messages) != len(seeds):
            raise ValueError("Each batched prompt must have a seed")
        if type(max_batch_tokens) is not int or max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be a positive integer")
        prompts = [self.encode(item) for item in messages]
        for ids in prompts:
            self._check_ids_capacity(ids)
        results, start = [], 0
        while start < len(prompts):
            end, width = start + 1, len(prompts[start])
            while end < len(prompts):
                candidate_width = max(width, len(prompts[end]))
                if (candidate_width + self.config["max_new_tokens_per_decision"]) * (end - start + 1) > max_batch_tokens:
                    break
                width, end = candidate_width, end + 1
            # A long single prompt is still allowed by the model context limit;
            # the batch budget only restricts grouping, never truncates a history.
            results.extend(self._generate(prompts[start:end], seeds[start:end]))
            start = end
        return results

    def _generate(self, prompts, seeds):
        import torch
        width = max(map(len, prompts))
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Reader tokenizer needs a pad or EOS token")
        ids = torch.tensor([[pad_id] * (width - len(row)) + row for row in prompts], dtype=torch.long, device=self.device)
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in prompts], dtype=torch.long, device=self.device)
        # A batch uses a reproducible joint RNG stream. Changing the batch size
        # can change samples; record the actual seed as well as each rollout seed.
        seed = seeds[0] if len(seeds) == 1 else seeded_int("reader_batch_v1", seeds) % (2 ** 31)
        self.model.eval()
        devices = sorted({p.device.index for p in self.model.parameters() if p.device.type == "cuda"})
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            kwargs = dict(max_new_tokens=self.config["max_new_tokens_per_decision"], do_sample=self.temperature > 0,
                          top_p=1.0, top_k=0, repetition_penalty=1.0, use_cache=self.config.get("rollout_use_cache", True),
                          return_dict_in_generate=True, output_scores=self.temperature > 0,
                          pad_token_id=pad_id)
            if self.temperature > 0:
                kwargs["temperature"] = self.temperature
            with torch.inference_mode():
                output = self.model.generate(input_ids=ids, attention_mask=mask, **kwargs)
                generated_ids = output.sequences[:, width:]
                sampled = generated_ids.tolist()
                eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
                if eos is None:
                    eos = self.tokenizer.eos_token_id
                eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
                if self.temperature > 0 and len(output.scores) != len(sampled[0]):
                    raise ValueError("Generation scores do not align with generated tokens")
                behavior = None
                if self.temperature > 0 and output.scores:
                    # Transfer selected probabilities once per batch, instead of
                    # synchronizing the GPU for every generated token via item().
                    behavior = torch.stack([
                        torch.log_softmax(score.float(), -1).gather(1, generated_ids[:, t:t + 1]).squeeze(1)
                        for t, score in enumerate(output.scores)
                    ], dim=1).tolist()
                results = []
                for row_index, (prompt_ids, generated) in enumerate(zip(prompts, sampled)):
                    # Include the sampled EOS in loss; discard only subsequent
                    # forced padding while longer rows continue generating.
                    for index, token in enumerate(generated):
                        if token in eos:
                            generated = generated[:index + 1]
                            break
                    logprobs = []
                    if behavior is not None:
                        logprobs = behavior[row_index][:len(generated)]
                    results.append(Completion(self.tokenizer.decode(generated, skip_special_tokens=True),
                                              prompt_ids, generated, logprobs, seed, len(prompts)))
        return results
