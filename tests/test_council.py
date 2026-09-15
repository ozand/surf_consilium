from pathlib import Path

from surf_consilium.adapters import (
    ClaudeSurfAdapter,
    CommandResult,
    DirectSurfAdapter,
    FailureKind,
    extract_claude_response,
)
from surf_consilium.council import run_council


class FakeRunner:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def run(self, argv, timeout):
        self.calls.append((list(argv), timeout))
        return self.results.pop(0)




def test_missing_adapters_are_not_reported_as_complete(tmp_path: Path):
    manifest = run_council("question", tmp_path, adapters={})
    assert manifest["completion"] == "FAILED"
    assert (tmp_path / "question.md").read_text(encoding="utf-8") == "question"
    assert (tmp_path / "manifest.json").exists()


def test_successful_stage1_has_explicit_degraded_stage2(tmp_path: Path):
    adapters = {name: (lambda prompt, name=name: f"{name}: answer") for name in ("chatgpt", "gemini", "claude")}
    manifest = run_council("question", tmp_path, adapters=adapters)
    assert manifest["completion"] == "COMPLETE"
    assert len([r for r in manifest["results"] if r["stage"] == "stage2" and r["status"] == "success"]) == 3


def test_direct_adapter_uses_argv_and_classifies_failures():
    runner = FakeRunner([CommandResult(1, "", "The command line is too long")])
    result = DirectSurfAdapter("chatgpt", runner=runner)("question")
    assert result.failure == FailureKind.TRANSPORT
    assert runner.calls[0][0][-1] == "question"


def test_direct_adapter_classifies_login_required():
    runner = FakeRunner([CommandResult(1, "", "ChatGPT login required")])
    result = DirectSurfAdapter("chatgpt", runner=runner)("question")
    assert result.failure == FailureKind.AUTHENTICATION


def test_direct_adapter_reads_provider_json():
    runner = FakeRunner([CommandResult(0, '{"response":"answer"}', "")])
    result = DirectSurfAdapter("gemini", runner=runner)("question")
    assert result.ok is True
    assert result.text == "answer"


def test_direct_adapter_rejects_large_prompt_before_spawn():
    runner = FakeRunner([])
    result = DirectSurfAdapter("gemini", runner=runner, prompt_limit=4)("too long")
    assert result.failure == FailureKind.PROMPT_TOO_LARGE
    assert runner.calls == []


def test_claude_extractor_returns_latest_assistant_message_only():
    page = "--- Page Text --- You said: hi Claude responded: final answer Claude finished the response"
    assert extract_claude_response(page) == "final answer"


def test_claude_adapter_requires_tab_id():
    result = ClaudeSurfAdapter(runner=FakeRunner([]))("question")
    assert result.failure == FailureKind.NOT_CONFIGURED


def test_claude_adapter_uses_supported_ui_and_extracts_response():
    runner = FakeRunner([
        CommandResult(0, "switched", ""),
        CommandResult(0, "--- Page Text --- Claude responded: old answer", ""),
        CommandResult(0, "filled", ""),
        CommandResult(0, "sent", ""),
        CommandResult(0, "--- Page Text --- Claude responded: answer", ""),
    ])
    adapter = ClaudeSurfAdapter(runner=runner, tab_id=123, poll_interval=0, poll_attempts=1)
    result = adapter("question")
    assert result.ok is True
    assert result.text == "answer"
    assert all("extract.text" not in call[0] for call in runner.calls)
