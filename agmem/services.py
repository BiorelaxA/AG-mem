from __future__ import annotations

import json
import os
import time

from .core import InfrastructureError, canonical, digest
from .data import ADAPTER_VERSION
from .prompts import writer_messages


class ChatService:
    """OpenAI-compatible local HTTP protocol; no OpenAI SDK or remote model dependency."""

    def __init__(self, config: dict, store, role: str):
        self.config, self.store, self.role = config, store, role
        self.identity = {key: config[key] for key in ("model", "model_revision", "base_url")}
        if not config["model_revision"] or config["model_revision"].startswith("REQUIRED"):
            raise ValueError("Set service.model_revision to the local checkpoint fingerprint/version")
        import httpx
        self.client = httpx.Client(timeout=config.get("timeout_seconds", 600))

    def complete(self, messages: list[dict], max_tokens: int, seed: int = 42, cache_scope: str | None = None) -> str:
        request = {"model": self.config["model"], "messages": messages, "temperature": 0.0,
                   "max_tokens": max_tokens, "seed": seed,
                   "chat_template_kwargs": {"enable_thinking": False}}
        key = digest({"identity": self.identity, "role": self.role, "request": request, "cache_scope": cache_scope})
        cached = self.store.completion(key)
        if cached is not None:
            return cached
        headers = {}
        env_name = self.config.get("api_key_env")
        if env_name and os.environ.get(env_name):
            headers["Authorization"] = "Bearer " + os.environ[env_name]
        last_error = None
        for attempt in range(self.config.get("retries", 2) + 1):
            try:
                response = self.client.post(self.config["base_url"].rstrip("/") + "/chat/completions", json=request, headers=headers)
                response.raise_for_status()
                choice = response.json()["choices"][0]
                text = choice["message"]["content"]
                if choice.get("finish_reason") == "length" or not isinstance(text, str) or not text.strip():
                    raise ValueError("Truncated or empty service completion")
                self.store.put_completion(key, text)
                return text
            except Exception as exc:
                last_error = exc
                if attempt < self.config.get("retries", 2):
                    time.sleep(min(2 ** attempt, 8))
        raise InfrastructureError(f"{self.role} service failed ({type(last_error).__name__}); no task score assigned") from last_error


class DirectoryBuilder:
    def __init__(self, store, service, config: dict, tokenizer):
        self.store, self.service, self.config, self.tokenizer = store, service, config, tokenizer
        if config["lookback_sessions"] < 5:
            raise ValueError("Writer needs at least five previous original sessions")
        identity = {"writer": service.identity, "config": config, "record_unit": ADAPTER_VERSION,
                    "storage_format": store.FORMAT, "reuse_scope": "complete_history",
                    "format_protocol": 3, "prompt": writer_messages([], {}, config["content_max_tokens"])}
        self.version = digest(identity)

    def build(self, memory_id: str) -> list[dict]:
        try:
            return self.store.directory(memory_id, self.version)
        except ValueError:
            pass
        references = self.store.memory(memory_id)
        entries = []
        lookback = self.config["lookback_sessions"]
        for index, reference in enumerate(references):
            context = [self.store.record(row) for row in references[max(0, index - lookback):index]]
            target = self.store.record(reference)
            messages = writer_messages(context, target, self.config["content_max_tokens"])
            cached = self.store.directory_entry(memory_id, self.version, reference["id"])
            content = cached["content"] if cached is not None else None
            if not target["messages"]:
                content = "Empty session; no recorded messages."
            if content is None:
                last_error = None
                current = messages
                max_attempts = self.config.get("format_retries", 2) + 1
                for attempt in range(max_attempts):
                    text = self.service.complete(current, self.config.get("max_output_tokens", 384),
                                                 cache_scope=f"{memory_id}/{self.version}")
                    overlength = False
                    try:
                        value = json.loads(text)
                        if not isinstance(value, dict) or set(value) != {"content"}:
                            raise ValueError('Expected one JSON object with only the "content" key')
                        content = value["content"]
                        if not isinstance(content, str) or not content.strip():
                            raise ValueError('The "content" value must be a nonempty string')
                        token_count = len(self.tokenizer.encode(content, add_special_tokens=False))
                        if token_count > self.config["content_max_tokens"]:
                            overlength = True
                            raise ValueError(f"Summary contains {token_count} tokens; content_max_tokens={self.config['content_max_tokens']}")
                        break
                    except (ValueError, KeyError, TypeError) as exc:
                        content, last_error = None, exc
                        if attempt + 1 < max_attempts:
                            repair = ("Shorten the summary to meet that limit, retaining the most useful facts from TARGET_SESSION. "
                                      if overlength else
                                      "Repair the JSON format; preserve useful summary facts within that limit. ")
                            feedback = (f"Validation failed: {exc}. Rewrite your previous answer as valid JSON only, "
                                        f'with exactly one key, "content", containing a nonempty summary string. '
                                        "Do not use Markdown code fences or add text outside the JSON object. "
                                        f"The decoded content limit is {self.config['content_max_tokens']} tokens. "
                                        f"{repair}Retry {attempt + 1}.")
                            current = messages + [{"role": "assistant", "content": text},
                                                  {"role": "user", "content": feedback}]
                if content is None:
                    raise ValueError(f"Writer format failure at {memory_id}/{reference['id']} "
                                     f"after {max_attempts} attempts: {last_error}") from last_error
            names = sorted({message.get("name") or message["role"] for message in target["messages"]})
            entry = {"id": reference["id"],
                     "task_session": {"task_id": None, "session_id": reference["source_session_id"]},
                     "actors_source": {"actors": names, "sources": []},
                     "time_order": {"time": reference["time"], "order": index}, "content": content}
            self.store.put_directory_entry(memory_id, self.version, entry)
            entries.append(entry)
        self.store.put_directory(memory_id, self.version, entries)
        return entries


class TaskAgent:
    def __init__(self, service, max_output_tokens=512):
        self.service, self.max_output_tokens = service, max_output_tokens

    def answer(self, example, evidence):
        from .prompts import task_messages
        messages = task_messages(example, evidence)
        return self.service.complete(messages, self.max_output_tokens), digest(messages)
