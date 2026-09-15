"""Three-stage advisory council orchestration over injected provider adapters."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol

from .adapters import ProviderResult

PROVIDERS = ("chatgpt", "gemini", "claude")
MAX_REVIEW_EXCERPT = 2500
MAX_SYNTHESIS_EXCERPT = 1800


class Provider(Protocol):
    def __call__(self, prompt: str) -> ProviderResult | str: ...


@dataclass
class Result:
    provider: str
    stage: str
    status: str
    artifact: str | None = None
    error: str | None = None


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[TRUNCATED: full output is in the stage artifact]"


def _call(provider: str, stage: str, prompt: str, output: Path, adapters: Mapping[str, Provider]) -> Result:
    adapter = adapters.get(provider)
    if adapter is None:
        return Result(provider, stage, "not_configured", error="provider adapter unavailable")
    try:
        answer = adapter(prompt)
    except Exception as exc:
        return Result(provider, stage, "failed", error=f"adapter_error: {type(exc).__name__}")
    if isinstance(answer, ProviderResult):
        if not answer.ok:
            return Result(provider, stage, "failed", error=answer.failure.value if answer.failure else "provider_failed")
        text = answer.text
    else:
        text = answer
    if not isinstance(text, str) or not text.strip():
        return Result(provider, stage, "failed", error="empty_response")
    _write(output, text)
    return Result(provider, stage, "success", artifact=str(output))


def _stage_results(results: list[Result], stage: str) -> list[Result]:
    return [result for result in results if result.stage == stage]


def _build_review_prompt(stage1: list[Result]) -> str:
    responses = []
    for index, result in enumerate(stage1):
        if not result.artifact:
            continue
        text = Path(result.artifact).read_text(encoding="utf-8")
        responses.append(f"Response {chr(65 + index)}:\n{_bounded(text, MAX_REVIEW_EXCERPT)}")
    return (
        "Review these anonymized independent responses. Identify agreement, disagreement, "
        "unsupported claims, risks, and a corrected recommendation.\n\n"
        + "\n\n".join(responses)
    )


def _build_synthesis_prompt(question: str, results: list[Result]) -> str:
    sections = [f"Original question:\n{question}"]
    for stage in ("stage1", "stage2"):
        entries = []
        for result in _stage_results(results, stage):
            if result.artifact:
                text = Path(result.artifact).read_text(encoding="utf-8")
                entries.append(f"{result.provider}:\n{_bounded(text, MAX_SYNTHESIS_EXCERPT)}")
            else:
                entries.append(f"{result.provider}: [status={result.status}; error={result.error}]")
        sections.append(f"{stage.upper()} EVIDENCE:\n" + "\n\n".join(entries))
    sections.append(
        "Synthesize a final advisory memo. State consensus, disagreements, evidence gaps, "
        "confidence, owner decisions, and actionable next steps. Do not present unsupported "
        "claims as established facts."
    )
    return "\n\n".join(sections)


def run_council(
    question: str,
    output_dir: Path,
    adapters: Mapping[str, Provider],
    providers: tuple[str, ...] = PROVIDERS,
    chairman: Provider | None = None,
) -> dict:
    """Run all bounded council stages and return the persisted run manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "question.md", question)
    results: list[Result] = []

    for provider in providers:
        results.append(_call(provider, "stage1", question, output_dir / f"stage1_{provider}.md", adapters))

    stage1 = _stage_results(results, "stage1")
    stage1_ok = [result for result in stage1 if result.status == "success"]
    if not stage1_ok:
        completion = "FAILED"
    else:
        review_prompt = _build_review_prompt(stage1_ok)
        for provider in providers:
            results.append(_call(provider, "stage2", review_prompt, output_dir / f"stage2_{provider}.md", adapters))

        if chairman is None:
            chairman = adapters.get("gemini") or next(iter(adapters.values()), None)
        if chairman is None:
            results.append(Result("chairman", "stage3", "not_configured", error="chairman adapter unavailable"))
        else:
            try:
                answer = chairman(_build_synthesis_prompt(question, results))
                if isinstance(answer, ProviderResult):
                    if answer.ok:
                        _write(output_dir / "stage3_final.md", answer.text)
                        results.append(Result("chairman", "stage3", "success", artifact=str(output_dir / "stage3_final.md")))
                    else:
                        results.append(Result("chairman", "stage3", "failed", error=answer.failure.value if answer.failure else "provider_failed"))
                elif isinstance(answer, str) and answer.strip():
                    _write(output_dir / "stage3_final.md", answer)
                    results.append(Result("chairman", "stage3", "success", artifact=str(output_dir / "stage3_final.md")))
                else:
                    results.append(Result("chairman", "stage3", "failed", error="empty_response"))
            except Exception as exc:
                results.append(Result("chairman", "stage3", "failed", error=f"adapter_error: {type(exc).__name__}"))

        stage2 = _stage_results(results, "stage2")
        stage3 = _stage_results(results, "stage3")
        if any(result.status != "success" for result in stage1):
            completion = "PARTIAL_STAGE1"
        elif any(result.status != "success" for result in stage2) or not stage3 or stage3[0].status != "success":
            completion = "PARTIAL_STAGE2"
        else:
            completion = "COMPLETE"

    manifest = {
        "version": 1,
        "created_at": int(time.time()),
        "providers": list(providers),
        "completion": completion,
        "results": [asdict(result) for result in results],
    }
    _write(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded surf-consilium protocol")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question")
    source.add_argument("--question-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(".council/run"))
    parser.add_argument("--window-id", default=None)
    parser.add_argument("--claude-tab-id", default=None)
    args = parser.parse_args()
    question = args.question if args.question is not None else args.question_file.read_text(encoding="utf-8")

    from .adapters import ClaudeSurfAdapter, DirectSurfAdapter

    adapters: dict[str, Provider] = {
        "chatgpt": DirectSurfAdapter("chatgpt"),
        "gemini": DirectSurfAdapter("gemini"),
    }
    if args.claude_tab_id:
        adapters["claude"] = ClaudeSurfAdapter(window_id=args.window_id, tab_id=args.claude_tab_id)
    manifest = run_council(question, args.output_dir, adapters=adapters)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["completion"] != "FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
