"""Provider adapters for the supported surf-cli browser interface.

The adapters keep transport separate from council orchestration. They use argv
arrays (never a shell string), apply a conservative prompt bound, and preserve
transport/extraction failures as typed results instead of model answers.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Sequence


DEFAULT_PROMPT_LIMIT = 12_000


class FailureKind(StrEnum):
    NOT_CONFIGURED = "not_configured"
    PROMPT_TOO_LARGE = "prompt_too_large"
    TIMEOUT = "timeout"
    TRANSPORT = "transport_failed"
    AUTHENTICATION = "authentication_failed"
    RATE_LIMITED = "rate_limited"
    BROWSER_ROUTING = "browser_routing_failed"
    GENERATION_INCOMPLETE = "generation_incomplete"
    EXTRACTION = "extraction_failed"
    EMPTY_RESPONSE = "empty_response"


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    ok: bool
    text: str = ""
    failure: FailureKind | None = None
    detail: str | None = None


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str], timeout: float) -> CommandResult: ...


class SubprocessRunner:
    """Run Surf without a shell and return bounded diagnostic output."""

    def __init__(self, executable: str | None = None, output_limit: int = 50_000):
        self.executable = executable or self._resolve_executable()
        self.output_limit = output_limit

    @staticmethod
    def _resolve_executable() -> str:
        candidates = ["surf.cmd", "surf"] if os.name == "nt" else ["surf"]
        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        raise FileNotFoundError("surf-cli executable not found on PATH")

    def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
        command = [self.executable, *argv]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or "")
            stderr = (exc.stderr or "")
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            return CommandResult(124, stdout[-self.output_limit :], stderr[-self.output_limit :], True)
        except OSError as exc:
            return CommandResult(127, "", f"{type(exc).__name__}: {exc}")
        return CommandResult(
            completed.returncode,
            completed.stdout[-self.output_limit :],
            completed.stderr[-self.output_limit :],
        )


def _detail(result: CommandResult) -> str:
    return (result.stderr or result.stdout or "provider command failed").strip()[-500:]


def _known_failure(text: str) -> FailureKind | None:
    lowered = text.lower()
    if "cloudflare challenge" in lowered or "complete in browser" in lowered:
        return FailureKind.AUTHENTICATION
    if "login required" in lowered or "not logged in" in lowered or "authentication required" in lowered:
        return FailureKind.AUTHENTICATION
    if "rate limit" in lowered or "too many requests" in lowered:
        return FailureKind.RATE_LIMITED
    if "content script not loaded" in lowered or "wrong profile" in lowered:
        return FailureKind.BROWSER_ROUTING
    if "timed out" in lowered or "timeout" in lowered:
        return FailureKind.TIMEOUT
    return None


def _direct_result(provider: str, result: CommandResult) -> ProviderResult:
    diagnostic = _detail(result)
    if result.timed_out:
        return ProviderResult(provider, False, failure=FailureKind.TIMEOUT, detail=diagnostic)
    known = _known_failure(diagnostic)
    if known is not None:
        return ProviderResult(provider, False, failure=known, detail=diagnostic)
    if result.returncode == 127:
        return ProviderResult(provider, False, failure=FailureKind.NOT_CONFIGURED, detail=diagnostic)
    if result.returncode != 0:
        return ProviderResult(provider, False, failure=FailureKind.TRANSPORT, detail=diagnostic)
    text = result.stdout.strip()
    if text.startswith("[Error") or text.startswith("Error:"):
        return ProviderResult(provider, False, failure=FailureKind.TRANSPORT, detail=text[-500:])
    if not text:
        return ProviderResult(provider, False, failure=FailureKind.EMPTY_RESPONSE, detail="empty stdout")
    # Surf's direct provider commands currently return plain text. If a future
    # version emits JSON, accept it without making JSON an unverified contract.
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ProviderResult(provider, True, text=text)
    if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
        return ProviderResult(provider, False, failure=FailureKind.EXTRACTION, detail="malformed provider JSON")
    if not payload["response"].strip():
        return ProviderResult(provider, False, failure=FailureKind.EMPTY_RESPONSE, detail="empty response field")
    return ProviderResult(provider, True, text=payload["response"].strip())


def _check_prompt(provider: str, prompt: str, limit: int) -> ProviderResult | None:
    if not prompt.strip():
        return ProviderResult(provider, False, failure=FailureKind.EMPTY_RESPONSE, detail="empty prompt")
    if len(prompt) > limit:
        return ProviderResult(
            provider,
            False,
            failure=FailureKind.PROMPT_TOO_LARGE,
            detail=f"prompt exceeds configured limit of {limit} characters",
        )
    return None


class DirectSurfAdapter:
    """Adapter for Surf's direct ChatGPT/Gemini commands."""

    def __init__(
        self,
        provider: str,
        runner: CommandRunner | None = None,
        *,
        model: str | None = None,
        prompt_limit: int = DEFAULT_PROMPT_LIMIT,
        timeout: float = 300,
    ):
        if provider not in {"chatgpt", "gemini"}:
            raise ValueError("direct provider must be chatgpt or gemini")
        self.provider = provider
        self.runner = runner or SubprocessRunner()
        self.model = model
        self.prompt_limit = prompt_limit
        self.timeout = timeout

    def __call__(self, prompt: str) -> ProviderResult:
        invalid = _check_prompt(self.provider, prompt, self.prompt_limit)
        if invalid is not None:
            return invalid
        argv = [self.provider]
        if self.model:
            argv.extend(["--model", self.model])
        argv.append(prompt)
        return _direct_result(self.provider, self.runner.run(argv, self.timeout))


def _page_text(read_output: str) -> str:
    return read_output.split("--- Page Text ---", 1)[-1].strip()


def extract_claude_response(read_output: str) -> str:
    """Extract the newest Claude answer from Surf accessibility/page output."""
    page = _page_text(read_output)
    matches = list(re.finditer(r"(?:Claude responded|Claude said):\s*", page, re.IGNORECASE))
    if not matches:
        return ""
    text = page[matches[-1].end() :]
    for marker in (
        "Claude is AI and can make mistakes.",
        "Want to be notified when Claude responds?",
        "Claude finished the response",
        "Use the up and down arrow keys",
    ):
        text = text.split(marker, 1)[0]
    # Accessibility output may repeat the visible answer after a thought label.
    text = re.sub(r"\bThought for \d+s\b", "", text, flags=re.IGNORECASE)
    return text.strip()


class ClaudeSurfAdapter:
    """Browser-UI adapter for Claude using only verified Surf primitives."""

    def __init__(
        self,
        runner: CommandRunner | None = None,
        *,
        window_id: str | int | None = None,
        tab_id: str | int | None = None,
        prompt_limit: int = DEFAULT_PROMPT_LIMIT,
        timeout: float = 300,
        poll_interval: float = 3,
        poll_attempts: int = 30,
    ):
        self.runner = runner or SubprocessRunner()
        self.window_id = str(window_id) if window_id is not None else None
        self.tab_id = str(tab_id) if tab_id is not None else None
        self.prompt_limit = prompt_limit
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.poll_attempts = poll_attempts

    def _surf(self, args: list[str], timeout: float | None = None) -> CommandResult:
        prefix = []
        if self.window_id:
            prefix.extend(["--window-id", self.window_id])
        if self.tab_id:
            prefix.extend(["--tab-id", self.tab_id])
        return self.runner.run([*prefix, *args], timeout or self.timeout)

    def __call__(self, prompt: str) -> ProviderResult:
        invalid = _check_prompt("claude", prompt, self.prompt_limit)
        if invalid is not None:
            return invalid
        if not self.tab_id:
            return ProviderResult("claude", False, failure=FailureKind.NOT_CONFIGURED, detail="tab_id is required")

        baseline_read = self._surf(["read", "--depth", "5", "--compact"], timeout=min(self.timeout, 60))
        if baseline_read.returncode != 0:
            return ProviderResult("claude", False, failure=FailureKind.BROWSER_ROUTING, detail=_detail(baseline_read))
        baseline = extract_claude_response(baseline_read.stdout)

        switched = self._surf(["tab.switch", self.tab_id])
        if switched.returncode != 0:
            return ProviderResult("claude", False, failure=FailureKind.BROWSER_ROUTING, detail=_detail(switched))

        filled = self._surf(
            [
                "locate.role",
                "textbox",
                "--name",
                "Write your prompt to Claude",
                "--action",
                "fill",
                "--value",
                prompt,
            ]
        )
        if filled.returncode != 0:
            return ProviderResult("claude", False, failure=FailureKind.TRANSPORT, detail=_detail(filled))

        sent = self._surf(["locate.role", "button", "--name", "Send message", "--action", "click"])
        if sent.returncode != 0:
            return ProviderResult("claude", False, failure=FailureKind.TRANSPORT, detail=_detail(sent))

        deadline = time.monotonic() + self.timeout
        last_read = ""
        for _ in range(self.poll_attempts):
            if time.monotonic() >= deadline:
                break
            time.sleep(self.poll_interval)
            read = self._surf(["read", "--depth", "5", "--compact"], timeout=min(self.timeout, 60))
            last_read = read.stdout
            if read.returncode != 0:
                return ProviderResult("claude", False, failure=FailureKind.EXTRACTION, detail=_detail(read))
            answer = extract_claude_response(last_read)
            if (
                answer
                and answer != baseline
                and "Stop response" not in last_read
                and "Claude is responding" not in last_read
            ):
                return ProviderResult("claude", True, text=answer)

        answer = extract_claude_response(last_read)
        if answer:
            return ProviderResult("claude", False, text=answer, failure=FailureKind.GENERATION_INCOMPLETE)
        return ProviderResult("claude", False, failure=FailureKind.EXTRACTION, detail="no assistant response found")
