"""Regenerate cached Rumoca artifacts from the Modelica sources.

Run this whenever a ``.mo`` model in this directory changes, or let
``dynamics.py`` call ``ensure_generated`` on demand. It uses the ``rumoca``
Python package when available, with ``$RUMOCA`` / ``rumoca`` CLI fallback for
developer checkouts. The generated files are local cache artifacts and are not
committed; the committed source of truth is the Modelica source plus the pinned
Rumoca dependency.

Usage::

    python -m models.aircraft6dof.modelica.generate
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"

# Modelica file (stem) -> model class name to compile.
MODELS = {
    "Aircraft6DOF": "Aircraft6DOF",
    "SportCubGreybox": "SportCubGreybox",
    "SafeController": "SafeController",
    "GroundRoll": "GroundRoll",
}


def _rumoca_binary() -> str:
    exe = os.environ.get("RUMOCA") or shutil.which("rumoca")
    if not exe:
        raise SystemExit(
            "rumoca not found. Install the Python package (`pip install -e .`) "
            "or set $RUMOCA to the compiler binary path."
        )
    return exe


def _python_rumoca():
    try:
        import rumoca
    except ImportError:
        return None
    return rumoca


def _compile_with_python(mo: Path, model_class: str, target: str | None) -> str:
    rumoca = _python_rumoca()
    if rumoca is None:
        raise ImportError("rumoca Python package is not installed")
    if target is None:
        return rumoca.compile_file_to_json(str(mo), model_name=model_class)
    return rumoca.render_target_file(str(mo), target=target, model_name=model_class)


def _compile_with_binary(mo: Path, model_class: str, target: str | None, rumoca: str) -> str:
    if target is None:
        cmd = [rumoca, "compile", str(mo), "--model", model_class, "--emit", "solve-json"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"rumoca solve-json failed for {mo.stem}:\n{proc.stderr}")
        return proc.stdout
    else:
        with tempfile.TemporaryDirectory(prefix="rumoca-codegen-") as tmp:
            cmd = [rumoca, "compile", str(mo), "--model", model_class, "--target", target, "--output", tmp]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise SystemExit(f"rumoca {target} failed for {mo.stem}:\n{proc.stderr}")
            filename = f"{mo.stem}_{target.replace('-', '_')}.py"
            return (Path(tmp) / filename).read_text()


def _compile(mo: Path, model_class: str, target: str | None, rumoca: str | None) -> str:
    if rumoca is None:
        try:
            return _compile_with_python(mo, model_class, target)
        except ImportError:
            rumoca = _rumoca_binary()
    return _compile_with_binary(mo, model_class, target, rumoca)


def generated_paths(model_stem: str) -> list[Path]:
    return [
        GENERATED / f"{model_stem}_casadi_solve.py",
        GENERATED / f"{model_stem}_jax_solve.py",
    ]


def is_stale(model_stem: str) -> bool:
    mo = HERE / f"{model_stem}.mo"
    paths = generated_paths(model_stem)
    if not mo.exists() or any(not p.exists() for p in paths):
        return True
    source_mtime = mo.stat().st_mtime
    return any(p.stat().st_mtime < source_mtime for p in paths)


def generate(model_stem: str, model_class: str, rumoca: str | None = None) -> Path:
    mo = HERE / f"{model_stem}.mo"
    if not mo.exists():
        raise SystemExit(f"missing Modelica source: {mo}")
    GENERATED.mkdir(exist_ok=True)

    # Solve-IR JSON dump (canonical IR; for inspection / debugging only).
    out = GENERATED / f"{model_stem}.solve.json"
    out.write_text(_compile(mo, model_class, None, rumoca))

    # Standalone explicit-ODE kernels  xdot = rhs(x, u, p)  for CasADi and JAX,
    # rendered directly from Rumoca. These are local runtime cache files.
    for target in ("casadi-solve", "jax-solve"):
        text = _compile(mo, model_class, target, rumoca)
        filename = f"{model_stem}_{target.replace('-', '_')}.py"
        (GENERATED / filename).write_text(text)
    return out


def ensure_generated(model_stem: str) -> None:
    if model_stem not in MODELS:
        raise SystemExit(f"unknown model '{model_stem}'; choices: {list(MODELS)}")
    if is_stale(model_stem):
        generate(model_stem, MODELS[model_stem])


def main(argv: list[str]) -> int:
    rumoca = os.environ.get("RUMOCA")
    targets = argv or list(MODELS)
    for stem in targets:
        if stem not in MODELS:
            raise SystemExit(f"unknown model '{stem}'; choices: {list(MODELS)}")
        out = generate(stem, MODELS[stem], rumoca)
        print(f"wrote {out.relative_to(HERE.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
