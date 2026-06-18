"""Emit an identified model as a standalone Modelica file.

Closes the round-trip: the benchmark consumes Modelica, and once a method *finds*
a model (fits its parameters), the result is written back out as Modelica. The
identified ``.mo`` is the base model with the fitted parameter values baked into
the ``parameter`` defaults -- self-contained and re-compilable with Rumoca, so
the identified system can itself be re-exported to CasADi/JAX.

Used by ``greybox_oem_fit`` to write ``results/<Model>Identified.mo`` alongside
the fitted-parameter JSON.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _format_value(value: float) -> str:
    # Round-trip-safe float text; ``repr`` keeps full precision without exponent noise.
    return repr(float(value))


def identified_model_source(
    base_mo: str | Path,
    param_values: dict[str, float],
    new_model_name: str,
    *,
    provenance: str | None = None,
) -> str:
    """Return Modelica source for ``base_mo`` with ``param_values`` baked in.

    Substitutes the default of each ``parameter Real <name> = <default>`` listed
    in ``param_values`` and renames the model to ``new_model_name``. Parameter
    names not present in the source are ignored; missing ones raise.
    """
    text = Path(base_mo).read_text()
    base_name = Path(base_mo).stem

    remaining = dict(param_values)
    def _sub(match: re.Match) -> str:
        head, name, _old, tail = match.group(1), match.group(2), match.group(3), match.group(4)
        if name in remaining:
            return f"{head}{name} = {_format_value(remaining.pop(name))}{tail}"
        return match.group(0)

    # parameter Real <name> = <value>[ "comment"];  (value is a numeric literal)
    pattern = re.compile(
        r"(parameter\s+Real\s+)([A-Za-z_]\w*)(\s*=\s*[-+0-9.eE]+)(\s*(?:\"[^\"]*\")?\s*;)"
    )
    text = pattern.sub(_sub, text)
    if remaining:
        raise KeyError(f"parameters not found in {base_name}.mo: {sorted(remaining)}")

    # Rename the model class (first `model <base>` -> `model <new>` and the
    # matching `end <base>;`).
    text = re.sub(rf"\bmodel\s+{re.escape(base_name)}\b", f"model {new_model_name}", text, count=1)
    text = re.sub(rf"\bend\s+{re.escape(base_name)}\s*;", f"end {new_model_name};", text, count=1)

    header = f"// Identified model generated from {base_name}.mo.\n"
    if provenance:
        header += f"// {provenance}\n"
    return header + text


def write_identified_modelica(
    base_stem: str,
    param_values: dict[str, float],
    out_path: str | Path,
    *,
    new_model_name: str | None = None,
    provenance: str | None = None,
) -> Path:
    """Write an identified ``.mo`` for the base model ``base_stem`` to ``out_path``.

    ``base_stem`` names a ``.mo`` in this directory (e.g. ``"SportCubGreybox"``).
    Returns the written path.
    """
    base_mo = HERE / f"{base_stem}.mo"
    name = new_model_name or f"{base_stem}Identified"
    source = identified_model_source(base_mo, param_values, name, provenance=provenance)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(source)
    return out_path
