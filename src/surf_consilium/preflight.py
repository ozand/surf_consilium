"""Fail-closed preflight checks for Surf browser/profile routing."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


EXPECTED_PROVIDER_HOSTS = frozenset({"chatgpt.com", "gemini.google.com", "claude.ai"})


class PreflightFailure(RuntimeError):
    """Raised when Surf cannot be proven to target the intended browser session."""


class SurfProbe(Protocol):
    def __call__(self, args: Sequence[str]) -> str: ...


@dataclass(frozen=True)
class BrowserPreflight:
    browser: str
    window_id: str | None
    tabs: tuple[Mapping[str, Any], ...]
    surf_version: str


def _host(url: str) -> str:
    from urllib.parse import urlparse

    return (urlparse(url).hostname or "").lower()


def parse_tabs(raw: str) -> list[dict[str, Any]]:
    """Parse Surf JSON output and reject malformed/non-tab payloads."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PreflightFailure("tab.list returned invalid JSON") from exc
    if not isinstance(payload, list):
        raise PreflightFailure("tab.list returned a non-list payload")
    tabs = []
    for tab in payload:
        if not isinstance(tab, dict) or not isinstance(tab.get("url"), str) or not isinstance(tab.get("windowId"), (str, int)):
            raise PreflightFailure("tab.list contains an invalid tab record")
        tabs.append(tab)
    return tabs


def verify_browser_profile(
    tabs: Sequence[Mapping[str, Any]],
    *,
    expected_window_id: str | int | None = None,
    allowed_hosts: frozenset[str] = EXPECTED_PROVIDER_HOSTS,
    require_provider_tabs: bool = True,
) -> str:
    """Verify all Surf-visible tabs belong to the intended window and provider set.

    Surf itself does not expose the Chrome executable or user-data-dir in tab JSON.
    This function therefore proves only the controlled tab inventory/window, not
    the OS process identity or network route.
    """
    if not tabs:
        raise PreflightFailure("Surf returned no tabs")
    window_ids = {str(tab["windowId"]) for tab in tabs}
    if expected_window_id is not None and window_ids != {str(expected_window_id)}:
        raise PreflightFailure("Surf tabs are outside the expected window")
    unexpected = [tab["url"] for tab in tabs if _host(tab["url"]) not in allowed_hosts]
    if unexpected:
        raise PreflightFailure("unrelated tabs detected in Surf-controlled browser")
    if require_provider_tabs and not any(_host(tab["url"]) in allowed_hosts for tab in tabs):
        raise PreflightFailure("no configured provider tab detected")
    return next(iter(window_ids))


def run_preflight(probe: SurfProbe, *, expected_window_id: str | int | None = None) -> BrowserPreflight:
    """Run version, doctor, and tab inventory checks through an injected probe."""
    version = probe(["--version"]).strip()
    if not version:
        raise PreflightFailure("surf-cli version check returned no output")
    doctor = probe(["doctor"])
    if "Doctor result: OK" not in doctor:
        raise PreflightFailure("surf doctor did not confirm a healthy native host")
    tabs = parse_tabs(probe(["tab.list", "--json"]))
    window_id = verify_browser_profile(tabs, expected_window_id=expected_window_id)
    return BrowserPreflight("chrome", window_id, tuple(tabs), version)
