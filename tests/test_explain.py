import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from claimverifier.config import ExplanationConfig
from claimverifier.explain import (
    EvidenceForPrompt,
    ExplanationGenerator,
    LLMBackend,
    OllamaLLM,
    OpenAICompatibleLLM,
    build_messages,
    postprocess,
    template_explanation,
)
from claimverifier.labels import CONTRADICTED, INSUFFICIENT, SUPPORTED

EVIDENCE = [
    EvidenceForPrompt(1, "Aspirin cohort", SUPPORTED, 0.9, ["Aspirin reduced colorectal cancer risk (HR 0.77)."]),
    EvidenceForPrompt(3, "Another trial", CONTRADICTED, 0.7, ["No reduction in cancer incidence was observed."]),
]


@pytest.mark.parametrize("verdict", [SUPPORTED, CONTRADICTED, INSUFFICIENT])
def test_template_explanation_cites_sources(verdict):
    exp = template_explanation("Aspirin prevents colorectal cancer.", verdict, 0.8, EVIDENCE, mixed=False)
    assert exp.backend == "template" and exp.text
    assert exp.citations and set(exp.citations) <= {1, 3}


def test_template_explanation_without_evidence():
    exp = template_explanation("x", INSUFFICIENT, 0.9, [])
    assert "insufficient evidence" in exp.text.lower() and exp.citations == []


def test_build_messages_contains_claim_and_numbered_evidence():
    msgs = build_messages("Aspirin prevents colorectal cancer.", SUPPORTED, 0.83, EVIDENCE, mixed=True)
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    user = msgs[1]["content"]
    assert "Aspirin prevents colorectal cancer." in user and "Supported (confidence 83%)" in user
    assert '[1] "Aspirin cohort"' in user and '[3] "Another trial"' in user
    assert "both supporting and contradicting" in user


def test_postprocess_removes_invalid_citations_and_adds_sources():
    exp = postprocess("Aspirin helps [1][7]. See also [3].", EVIDENCE)
    assert "[7]" not in exp.text and exp.citations == [1, 3]
    exp = postprocess("The evidence supports the claim.", EVIDENCE)
    assert "Sources: [1], [3]" in exp.text and exp.citations == [1, 3]


class FakeLLM(LLMBackend):
    name = "fake"

    def __init__(self, reply=None, fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    def chat(self, messages, max_new_tokens, temperature):
        self.calls.append(messages)
        if self.fail:
            raise ConnectionError("server down")
        return self.reply


def test_generator_uses_llm_and_falls_back():
    cfg = ExplanationConfig(backend="ollama")
    gen = ExplanationGenerator(cfg, backend=FakeLLM("Supported by [1] despite [3]."))
    exp = gen.generate("claim", SUPPORTED, 0.9, EVIDENCE)
    assert exp.backend.startswith("fake") and exp.citations == [1, 3]
    gen = ExplanationGenerator(cfg, backend=FakeLLM(fail=True))
    exp = gen.generate("claim", SUPPORTED, 0.9, EVIDENCE)
    assert exp.backend.startswith("template") and "server down" in exp.error
    cfg.fallback_to_template = False
    with pytest.raises(ConnectionError):
        ExplanationGenerator(cfg, backend=FakeLLM(fail=True)).generate("claim", SUPPORTED, 0.9, EVIDENCE)


def test_generator_none_and_template_backends():
    assert ExplanationGenerator(ExplanationConfig(backend="none")).generate("c", SUPPORTED, 1, EVIDENCE).text == ""
    exp = ExplanationGenerator(ExplanationConfig(backend="template")).generate("c", SUPPORTED, 1, EVIDENCE)
    assert exp.backend == "template"


def test_unloadable_backend_falls_back_once():
    gen = ExplanationGenerator(ExplanationConfig(backend="no-such-backend"))
    first = gen.generate("c", SUPPORTED, 1, EVIDENCE)
    second = gen.generate("c", SUPPORTED, 1, EVIDENCE)
    assert first.backend.startswith("template") and second.error == first.error


@pytest.fixture()
def fake_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, dict(self.headers), body))
            if self.path == "/api/chat":
                reply = {"message": {"role": "assistant", "content": "Ollama says [1]."}}
            else:
                reply = {"choices": [{"message": {"role": "assistant", "content": "OpenAI-style says [1]."}}]}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", received
    server.shutdown()


def test_ollama_backend(fake_server):
    url, received = fake_server
    out = OllamaLLM("llama3.1:8b", url).chat([{"role": "user", "content": "hi"}], 50, 0.1)
    assert out == "Ollama says [1]."
    path, _, body = received[0]
    assert path == "/api/chat" and body["model"] == "llama3.1:8b" and body["stream"] is False
    assert body["options"]["num_predict"] == 50


def test_openai_compatible_backend(fake_server):
    url, received = fake_server
    out = OpenAICompatibleLLM("qwen2.5", f"{url}/v1", api_key="secret").chat([{"role": "user", "content": "hi"}], 20, 0)
    assert out == "OpenAI-style says [1]."
    path, headers, body = received[0]
    assert path == "/v1/chat/completions" and body["max_tokens"] == 20
    assert headers.get("Authorization") == "Bearer secret"
