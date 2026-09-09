"""YOLOv26 pruning helpers with no import-time Vitis AI dependency."""

from __future__ import annotations

import fnmatch
import importlib
from typing import Any


def _patterns_from(value: Any) -> tuple[Any, ...]:
    """Accept a typed config or a direct iterable of exclusion patterns."""

    if hasattr(value, "prune_excludes"):
        value = value.prune_excludes
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(value)
    except TypeError as error:
        raise TypeError(
            "Pruning exclusions must be a model config or an iterable of patterns"
        ) from error


def resolve_prune_excludes(model: Any, patterns_or_config: Any) -> list[Any]:
    """Resolve configured module globs to objects accepted by Vitis AI.

    Vitis AI's ``excluded_node_names`` handling does not expand shell globs.
    Dotted names are consequently matched against ``model.named_modules()``
    here and replaced with the corresponding module objects.  Exact XIR graph
    node names (which contain ``::``) pass through unchanged.

    If ``model`` is ``None``, the unresolved values are returned.  This mirrors
    the sensitivity-analysis setup where a model may not yet be available.
    Unmatched values are retained so an exact Vitis node name without ``::``
    can still be consumed by the runner, while a warning makes a likely typo
    visible.
    """

    patterns = _patterns_from(patterns_or_config)
    if model is None:
        return list(patterns)
    if not hasattr(model, "named_modules"):
        raise TypeError(f"{type(model).__name__} does not provide named_modules()")

    modules_by_name = dict(model.named_modules())
    resolved: list[Any] = []
    for pattern in patterns:
        if not isinstance(pattern, str):
            # Already-resolved module objects are valid runner exclusions.
            resolved.append(pattern)
            continue
        if "::" in pattern:
            resolved.append(pattern)
            continue

        matches = [
            module
            for name, module in modules_by_name.items()
            if fnmatch.fnmatchcase(name, pattern)
        ]
        if matches:
            resolved.extend(matches)
        else:
            print(f"[WARN] prune exclude pattern matched no module: {pattern}")
            resolved.append(pattern)
    return resolved


def create_pruning_runner(model: Any, example_input: Any, method: str = "iterative") -> Any:
    """Create a Vitis coarse-grained runner, importing Vitis only on demand."""

    normalized_method = method.lower().replace("-", "_")
    if normalized_method == "onestep":
        normalized_method = "one_step"
    if normalized_method not in {"iterative", "one_step"}:
        raise ValueError(
            f"Unknown coarse-grained pruning method {method!r}; "
            "expected 'iterative' or 'one_step'"
        )

    try:
        pytorch_nndct = importlib.import_module("pytorch_nndct")
    except ImportError as error:
        raise ImportError(
            "YOLOv26 pruning requires pytorch_nndct from the Vitis AI environment"
        ) from error
    try:
        factory = pytorch_nndct.get_pruning_runner
    except AttributeError as error:
        raise ImportError(
            "Installed pytorch_nndct does not expose get_pruning_runner"
        ) from error
    return factory(model, example_input, normalized_method)


def prune_excludes(model: Any, config: Any) -> list[Any]:
    """Short config-oriented alias used by optimizer stage code."""

    return resolve_prune_excludes(model, config)


__all__ = [
    "create_pruning_runner",
    "prune_excludes",
    "resolve_prune_excludes",
]
