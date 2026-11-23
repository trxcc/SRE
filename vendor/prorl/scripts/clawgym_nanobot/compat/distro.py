"""Minimal fallback for SGLang images missing the optional ``distro`` package."""

from __future__ import annotations


def id() -> str:
    return ""


def name(pretty: bool = False) -> str:
    del pretty
    return ""


def version(pretty: bool = False, best: bool = False) -> str:
    del pretty, best
    return ""


def codename() -> str:
    return ""


def like() -> str:
    return ""


def linux_distribution(full_distribution_name: bool = True) -> tuple[str, str, str]:
    return (name(full_distribution_name), version(), codename())


def os_release_info() -> dict[str, str]:
    return {}


def distro_release_info() -> dict[str, str]:
    return {}
