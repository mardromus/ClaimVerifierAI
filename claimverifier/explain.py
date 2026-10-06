"""Retrieval-augmented explanation generation with citations.

The verdict itself comes from the NLI pipeline; the LLM (Llama 3 / Qwen 2.5 Instruct) is asked to
*explain* it using only the retrieved evidence sentences, citing them as [1], [2], ...

Backends: ``transformers`` (local Hugging Face chat model), ``ollama`` (local Ollama server),
``openai`` (any OpenAI-compatible endpoint: vLLM, llama.cpp server, LM Studio, ...),
``template`` (deterministic extractive explanation, no LLM) and ``none``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .labels import CONTRADICTED, DISPLAY_NAMES, INSUFFICIENT, SUPPORTED

logger = logging.getLogger(__name__)

STANCE_WORDS = {SUPPORTED: "supports", CONTRADICTED: "contradicts", INSUFFICIENT: "is inconclusive about"}


@dataclass
class EvidenceForPrompt:
    citation: int
    title: str
    stance: str
    relevance: float
    sentences: List[str]
    url: str = ""


@dataclass
class Explanation:
    text: str
    backend: str
    citations: List[int] = field(default_factory=list)
    error: Optional[str] = None


SYSTEM_PROMPT = (
    "You are a careful scientific fact-checking assistant. You explain verdicts produced by an automated "
    "claim-verification system, using ONLY the evidence sentences provided from research abstracts. "
    "Cite every statement with the bracketed source number, e.g. [1] or [2][3]. Never invent findings, "
    "numbers, papers or citations, and do not use outside knowledge."
)


def build_messages(claim: str, verdict: str, confidence: float, evidence: List[EvidenceForPrompt],
                   mixed: bool = False) -> List[Dict[str, str]]:
    lines = [f'Claim: "{claim}"',
             f"Verdict from the verification model: {DISPLAY_NAMES[verdict]} (confidence {confidence:.0%}).",
             ""]
    if mixed:
        lines.append("Note: the retrieved papers contain both supporting and contradicting evidence.")
    if evidence:
        lines.append("Evidence retrieved from research abstracts:")
        for ev in evidence:
            lines.append(f'[{ev.citation}] "{ev.title}" - this abstract {STANCE_WORDS[ev.stance]} the claim '
                         f"(evidence relevance {ev.relevance:.2f})")
            lines.extend(f'    - "{s}"' for s in ev.sentences)
    else:
        lines.append("No relevant evidence was retrieved.")
    lines += [
        "",
        "Task: In 3-5 sentences, explain why this evidence supports, contradicts, or is insufficient to verify "
        "the claim. Reference sources inline as [n]. Point out important caveats (study population, design, "
        "effect size, weak or conflicting evidence) only if they appear in the evidence. Do not add information "
        "that is not in the evidence. Do not change the verdict; if the evidence looks weak, say so.",
    ]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(lines)}]


_CITE_RE = re.compile(r"\[(\d+)\]")


def postprocess(text: str, evidence: List[EvidenceForPrompt]) -> Explanation:
    """Drop citations to non-existent sources and make sure the explanation cites something."""
    valid = {ev.citation for ev in evidence}
    text = text.strip()
    text = _CITE_RE.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", text)
    cited = sorted({int(n) for n in _CITE_RE.findall(text)})
    if not cited and evidence:
        refs = ", ".join(f"[{ev.citation}]" for ev in evidence[:3])
        text = f"{text}\n\nSources: {refs}."
        cited = [ev.citation for ev in evidence[:3]]
    return Explanation(text=text, backend="", citations=cited)


# ---------------------------------------------------------------------------- template (no LLM)


def _quote(sentence: str, max_chars: int = 260) -> str:
    sentence = sentence.strip()
    return sentence if len(sentence) <= max_chars else sentence[:max_chars].rsplit(" ", 1)[0] + "..."


def template_explanation(claim: str, verdict: str, confidence: float, evidence: List[EvidenceForPrompt],
                         mixed: bool = False) -> Explanation:
    by_stance = {s: [ev for ev in evidence if ev.stance == s] for s in (SUPPORTED, CONTRADICTED, INSUFFICIENT)}
    parts: List[str] = []
    if verdict in (SUPPORTED, CONTRADICTED):
        main = by_stance[verdict] or evidence
        verb = "supported" if verdict == SUPPORTED else "contradicted"
        parts.append(f"The claim is {verb} by the retrieved literature (confidence {confidence:.0%}).")
        if main:
            top = main[0]
            parts.append(f'The strongest evidence comes from "{top.title}" [{top.citation}], which reports: '
                         f'"{_quote(top.sentences[0])}"')
            if len(top.sentences) > 1:
                parts.append(f'It also states: "{_quote(top.sentences[1])}" [{top.citation}]')
            others = [f"[{ev.citation}]" for ev in main[1:]]
            if others:
                parts.append(f"Additional {'supporting' if verdict == SUPPORTED else 'contradicting'} "
                             f"evidence: {', '.join(others)}.")
        opposite = CONTRADICTED if verdict == SUPPORTED else SUPPORTED
        if by_stance[opposite]:
            refs = ", ".join(f"[{ev.citation}]" for ev in by_stance[opposite])
            parts.append(f"Note that {refs} point{'s' if len(by_stance[opposite]) == 1 else ''} the other way, "
                         f"so the evidence is {'mixed' if mixed else 'not fully consistent'}.")
    else:
        parts.append(f"There is insufficient evidence in the retrieved abstracts to verify this claim "
                     f"(confidence {confidence:.0%}).")
        if evidence:
            top = evidence[0]
            parts.append(f'The most closely related abstract, "{top.title}" [{top.citation}], states: '
                         f'"{_quote(top.sentences[0])}", which does not directly confirm or refute the claim.')
            leaning = [ev for ev in evidence if ev.stance != INSUFFICIENT]
            if leaning:
                refs = ", ".join(f"[{ev.citation}] ({STANCE_WORDS[ev.stance]})" for ev in leaning)
                parts.append(f"Some sources lean one way ({refs}), but their relevance or the model's "
                             f"confidence is too low to reach a verdict.")
        else:
            parts.append("No sufficiently relevant research abstracts were found.")
    text = " ".join(parts)
    cited = sorted({int(n) for n in _CITE_RE.findall(text)})
    return Explanation(text=text, backend="template", citations=cited)


# ---------------------------------------------------------------------------- LLM backends


class LLMBackend:
    name = "base"

    def chat(self, messages: List[Dict[str, str]], max_new_tokens: int, temperature: float) -> str:
        raise NotImplementedError


class TransformersChatLLM(LLMBackend):
    """Local Hugging Face chat model, e.g. Qwen/Qwen2.5-1.5B-Instruct or meta-llama/Meta-Llama-3-8B-Instruct."""

    name = "transformers"

    def __init__(self, model_name: str, device: str = "cpu"):
        from .hf_utils import require_transformers

        require_transformers()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        dtype = torch.bfloat16 if device == "cuda" else (torch.float16 if device == "mps" else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except TypeError:  # transformers < 4.56 uses torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(device)
        self.model.eval()

    def chat(self, messages, max_new_tokens, temperature):
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                      "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id}
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=0.9)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _post_json(url: str, payload: dict, timeout: float, headers: Optional[Dict[str, str]] = None) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OllamaLLM(LLMBackend):
    """Local Ollama server (``ollama pull llama3.1:8b`` or ``ollama pull qwen2.5:7b``)."""

    name = "ollama"

    def __init__(self, model: str, url: str = "http://localhost:11434", timeout: float = 120.0):
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout

    def chat(self, messages, max_new_tokens, temperature):
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": temperature, "num_predict": max_new_tokens}}
        return _post_json(f"{self.url}/api/chat", payload, self.timeout)["message"]["content"]


class OpenAICompatibleLLM(LLMBackend):
    """Any server implementing ``POST /v1/chat/completions`` (vLLM, llama.cpp, LM Studio, TGI, ...)."""

    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: Optional[str] = None, timeout: float = 120.0):
        self.model, self.base_url, self.api_key, self.timeout = model, base_url.rstrip("/"), api_key, timeout

    def chat(self, messages, max_new_tokens, temperature):
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"model": self.model, "messages": messages, "max_tokens": max_new_tokens,
                   "temperature": temperature}
        response = _post_json(f"{self.base_url}/chat/completions", payload, self.timeout, headers)
        return response["choices"][0]["message"]["content"]


class ExplanationGenerator:
    """Builds the RAG prompt, calls the configured LLM and falls back to the template on failure."""

    def __init__(self, cfg, device: str = "cpu", backend: Optional[LLMBackend] = None):
        self.cfg = cfg
        self.device = device
        self._backend = backend
        self._backend_error: Optional[str] = None

    @property
    def backend_name(self) -> str:
        return self.cfg.backend

    def _get_backend(self) -> LLMBackend:
        if self._backend is not None:
            return self._backend
        if self._backend_error is not None:  # don't retry a model that failed to load on every request
            raise RuntimeError(self._backend_error)
        cfg = self.cfg
        try:
            if cfg.backend == "transformers":
                self._backend = TransformersChatLLM(cfg.model_name, self.device)
            elif cfg.backend == "ollama":
                self._backend = OllamaLLM(cfg.ollama_model, cfg.ollama_url, cfg.timeout)
            elif cfg.backend == "openai":
                self._backend = OpenAICompatibleLLM(cfg.openai_model, cfg.openai_base_url,
                                                    os.environ.get(cfg.openai_api_key_env), cfg.timeout)
            else:
                raise ValueError(f"Unknown explanation backend {cfg.backend!r}")
        except Exception as e:  # noqa: BLE001
            self._backend_error = f"{type(e).__name__}: {e}"
            raise RuntimeError(self._backend_error) from e
        return self._backend

    def generate(self, claim: str, verdict: str, confidence: float, evidence: List[EvidenceForPrompt],
                 mixed: bool = False) -> Explanation:
        evidence = evidence[: self.cfg.max_evidence]
        if self.cfg.backend == "none":
            return Explanation(text="", backend="none")
        if self.cfg.backend == "template":
            return template_explanation(claim, verdict, confidence, evidence, mixed)
        try:
            backend = self._get_backend()
            messages = build_messages(claim, verdict, confidence, evidence, mixed)
            raw = backend.chat(messages, self.cfg.max_new_tokens, self.cfg.temperature)
            if not raw or not raw.strip():
                raise RuntimeError("LLM returned an empty response")
            result = postprocess(raw, evidence)
            result.backend = f"{backend.name}:{getattr(backend, 'model_name', getattr(backend, 'model', ''))}"
            return result
        except Exception as e:  # noqa: BLE001
            if not self.cfg.fallback_to_template:
                raise
            logger.warning("LLM explanation failed (%s); using template explanation", e)
            result = template_explanation(claim, verdict, confidence, evidence, mixed)
            result.backend = "template (LLM unavailable)"
            result.error = str(e)
            return result
