"""Regenerate cached Rumoca solve-IR artifacts from the Modelica sources.

Run this whenever a ``.mo`` model in this directory changes. It invokes the
Rumoca compiler (``rumoca`` on PATH, or ``$RUMOCA``) and writes the
``solve``-IR JSON the runtime backend consumes into ``generated/``. The runtime
(``dynamics.py`` / ``rumoca_backend.py``) reads only the committed JSON, so the
benchmark itself never needs Rumoca installed.

Usage::

    python -m models.aircraft6dof.modelica.generate
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"

# Modelica file (stem) -> model class name to compile.
MODELS = {
    "Aircraft6DOF": "Aircraft6DOF",
    "SportCubGreybox": "SportCubGreybox",
    "SportCubGreyboxLag": "SportCubGreyboxLag",
}


def _rumoca() -> str:
    exe = os.environ.get("RUMOCA") or shutil.which("rumoca")
    if not exe:
        raise SystemExit(
            "rumoca not found. Install it or set $RUMOCA to the binary path."
        )
    return exe


def generate(model_stem: str, model_class: str, rumoca: str) -> Path:
    mo = HERE / f"{model_stem}.mo"
    if not mo.exists():
        raise SystemExit(f"missing Modelica source: {mo}")
    GENERATED.mkdir(exist_ok=True)
    out = GENERATED / f"{model_stem}.solve.json"
    proc = subprocess.run(
        [rumoca, "compile", str(mo), "--model", model_class, "--emit", "solve-json"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"rumoca failed for {model_stem}:\n{proc.stderr}")
    out.write_text(proc.stdout)
    return out


def main(argv: list[str]) -> int:
    rumoca = _rumoca()
    targets = argv or list(MODELS)
    for stem in targets:
        if stem not in MODELS:
            raise SystemExit(f"unknown model '{stem}'; choices: {list(MODELS)}")
        out = generate(stem, MODELS[stem], rumoca)
        print(f"wrote {out.relative_to(HERE.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
