"""Turn a Rumoca ``solve``-IR dump into an explicit ``xdot = f(x, u, p)`` kernel.

Rumoca lowers a Modelica model's continuous dynamics to a small register-machine
bytecode (``continuous.derivative_rhs``). Each state derivative is a self-contained
straight-line program whose algebraics are already inlined, so we can replay the
bytecode directly into either a CasADi ``Function`` or a JAX-traceable closure --
no DAE rootfinder, no algebraic projection. The same interpreter drives both
backends; only the scalar primitives differ.

The IR is produced by ``rumoca compile <model>.mo --emit solve-json`` (see
``generate.py``); this module consumes the committed JSON so the benchmark does
not need Rumoca installed at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelMeta:
    """Layout of the generated kernel's argument vectors."""

    state_names: tuple[str, ...]
    input_names: tuple[str, ...]
    param_names: tuple[str, ...]
    compiled_param_len: int  # length of the flat parameter vector the bytecode indexes

    @property
    def n_x(self) -> int:
        return len(self.state_names)

    @property
    def n_u(self) -> int:
        return len(self.input_names)

    @property
    def n_p(self) -> int:
        return len(self.param_names)

    def param_index(self, name: str) -> int:
        return self.param_names.index(name)


def load_ir(path: str | Path) -> dict:
    """Load a Rumoca solve-IR JSON dump."""
    return json.loads(Path(path).read_text())


def _meta_from_ir(ir: dict) -> KernelMeta:
    layout = ir["solve_layout"]
    names = layout["solver_maps"]["names"]
    n_state = layout["state_scalar_count"]
    state_names = tuple(names[:n_state])
    input_names = tuple(layout["input_scalar_names"])
    # Parameter names in compiled order come from the layout bindings (P slots).
    p_by_index: dict[int, str] = {}
    for name, bind in ir["layout"]["bindings"].items():
        if "[" in name or name == "time":
            continue
        if isinstance(bind, dict) and "P" in bind:
            p_by_index[bind["P"]["index"]] = name
    param_names = tuple(p_by_index[i] for i in range(layout["parameter_count"]))
    return KernelMeta(
        state_names=state_names,
        input_names=input_names,
        param_names=param_names,
        compiled_param_len=layout["compiled_parameter_len"],
    )


def _programs(ir: dict) -> list[list[dict]]:
    nodes = ir["continuous"]["derivative_rhs"]["nodes"]
    programs: list[list[dict]] = []
    for node in nodes:
        sp = node.get("ScalarPrograms")
        if sp is None:
            raise ValueError(f"unsupported derivative_rhs node kind: {list(node)}")
        programs.extend(sp["programs"])
    return programs


def _run(programs: list[list[dict]], load_y, load_p, ops: dict) -> list:
    """Replay every program; return the flat list of StoreOutput values in order."""
    outputs: list = []
    for prog in programs:
        reg: dict[int, object] = {}
        for ins in prog:
            (kind, v), = ins.items()
            if kind == "Const":
                reg[v["dst"]] = ops["const"](v["value"])
            elif kind == "LoadY":
                reg[v["dst"]] = load_y(v["index"])
            elif kind == "LoadP":
                reg[v["dst"]] = load_p(v["index"])
            elif kind == "Binary":
                reg[v["dst"]] = ops["binary"][v["op"]](reg[v["lhs"]], reg[v["rhs"]])
            elif kind == "Unary":
                reg[v["dst"]] = ops["unary"][v["op"]](reg[v["arg"]])
            elif kind == "Compare":
                reg[v["dst"]] = ops["compare"][v["op"]](reg[v["lhs"]], reg[v["rhs"]])
            elif kind == "Select":
                reg[v["dst"]] = ops["select"](reg[v["cond"]], reg[v["if_true"]], reg[v["if_false"]])
            elif kind == "StoreOutput":
                outputs.append(reg[v["src"]])
            else:
                raise ValueError(f"unsupported solve-IR opcode: {kind}")
    return outputs


# --------------------------------------------------------------------------- #
# CasADi backend
# --------------------------------------------------------------------------- #
def _casadi_ops(ca):
    return {
        "const": lambda v: ca.SX(float(v)),
        "binary": {
            "Add": lambda a, b: a + b, "Sub": lambda a, b: a - b,
            "Mul": lambda a, b: a * b, "Div": lambda a, b: a / b,
            "Pow": lambda a, b: a ** b, "Max": ca.fmax, "Min": ca.fmin,
            "Atan2": ca.atan2,
        },
        "unary": {
            "Neg": lambda a: -a, "Abs": ca.fabs, "Sign": ca.sign,
            "Sin": ca.sin, "Cos": ca.cos, "Tan": ca.tan,
            "Asin": ca.asin, "Acos": ca.acos, "Atan": ca.atan,
            "Sqrt": ca.sqrt, "Exp": ca.exp, "Log": ca.log, "Tanh": ca.tanh,
        },
        "compare": {
            "Ge": lambda a, b: a >= b, "Gt": lambda a, b: a > b,
            "Le": lambda a, b: a <= b, "Lt": lambda a, b: a < b,
            "Eq": lambda a, b: a == b, "Neq": lambda a, b: a != b,
        },
        "select": lambda c, t, f: ca.if_else(c, t, f),
    }


def build_casadi_rhs(ir: dict):
    """Build a CasADi ``Function`` ``rhs(x, u, p) -> xdot`` from a solve-IR dump.

    Returns ``(rhs_function, meta)``. ``p`` is the model parameter vector in
    ``meta.param_names`` order; inputs ``u`` are in ``meta.input_names`` order.
    """
    import casadi as ca

    meta = _meta_from_ir(ir)
    programs = _programs(ir)
    x = ca.SX.sym("x", meta.n_x)
    u = ca.SX.sym("u", meta.n_u)
    p = ca.SX.sym("p", meta.n_p)
    # Compiled parameter vector the bytecode indexes: [params, inputs, pad...].
    pad = meta.compiled_param_len - meta.n_p - meta.n_u
    parts = [p, u]
    if pad > 0:
        parts.append(ca.SX.zeros(pad))
    P = ca.vertcat(*parts)
    outputs = _run(programs, lambda i: x[i], lambda i: P[i], _casadi_ops(ca))
    xdot = ca.vertcat(*outputs)
    rhs = ca.Function("rhs", [x, u, p], [xdot], ["x", "u", "p"], ["xdot"])
    return rhs, meta


# --------------------------------------------------------------------------- #
# NumPy backend (dependency-free; also the verification mirror for JAX, whose
# closure is identical with jnp swapped for np)
# --------------------------------------------------------------------------- #
def _numpy_ops(np):
    return {
        "const": lambda v: np.float64(v),
        "binary": {
            "Add": lambda a, b: a + b, "Sub": lambda a, b: a - b,
            "Mul": lambda a, b: a * b, "Div": lambda a, b: a / b,
            "Pow": lambda a, b: a ** b, "Max": np.maximum, "Min": np.minimum,
            "Atan2": np.arctan2,
        },
        "unary": {
            "Neg": lambda a: -a, "Abs": np.abs, "Sign": np.sign,
            "Sin": np.sin, "Cos": np.cos, "Tan": np.tan,
            "Asin": np.arcsin, "Acos": np.arccos, "Atan": np.arctan,
            "Sqrt": np.sqrt, "Exp": np.exp, "Log": np.log, "Tanh": np.tanh,
        },
        "compare": {
            "Ge": lambda a, b: a >= b, "Gt": lambda a, b: a > b,
            "Le": lambda a, b: a <= b, "Lt": lambda a, b: a < b,
            "Eq": lambda a, b: a == b, "Neq": lambda a, b: a != b,
        },
        "select": lambda c, t, f: np.where(c, t, f),
    }


def build_numpy_rhs(ir: dict):
    """Build a pure-NumPy ``rhs(x, u, p) -> xdot`` from a solve-IR dump.

    Returns ``(rhs_callable, meta)``. No third-party dependencies; useful as a
    reference and as the structural mirror of the JAX backend.
    """
    import numpy as np

    meta = _meta_from_ir(ir)
    programs = _programs(ir)
    pad = meta.compiled_param_len - meta.n_p - meta.n_u
    ops = _numpy_ops(np)

    def rhs(x, u, p):
        x = np.asarray(x, dtype=float); u = np.asarray(u, dtype=float); p = np.asarray(p, dtype=float)
        P = np.concatenate([p, u, np.zeros(pad)]) if pad > 0 else np.concatenate([p, u])
        return np.array(_run(programs, lambda i: x[i], lambda i: P[i], ops))

    return rhs, meta


# --------------------------------------------------------------------------- #
# JAX backend
# --------------------------------------------------------------------------- #
def build_jax_rhs(ir: dict):
    """Build a JAX-traceable ``rhs(x, u, p) -> xdot`` closure from a solve-IR dump.

    Returns ``(rhs_callable, meta)``. The closure runs the interpreter over
    ``jax.numpy`` arrays, so it is ``jit``/``grad``/``vmap``-compatible.
    """
    import jax.numpy as jnp

    meta = _meta_from_ir(ir)
    programs = _programs(ir)
    pad = meta.compiled_param_len - meta.n_p - meta.n_u
    ops = {
        "const": lambda v: jnp.asarray(float(v)),
        "binary": {
            "Add": lambda a, b: a + b, "Sub": lambda a, b: a - b,
            "Mul": lambda a, b: a * b, "Div": lambda a, b: a / b,
            "Pow": lambda a, b: a ** b, "Max": jnp.maximum, "Min": jnp.minimum,
            "Atan2": jnp.arctan2,
        },
        "unary": {
            "Neg": lambda a: -a, "Abs": jnp.abs, "Sign": jnp.sign,
            "Sin": jnp.sin, "Cos": jnp.cos, "Tan": jnp.tan,
            "Asin": jnp.arcsin, "Acos": jnp.arccos, "Atan": jnp.arctan,
            "Sqrt": jnp.sqrt, "Exp": jnp.exp, "Log": jnp.log, "Tanh": jnp.tanh,
        },
        "compare": {
            "Ge": lambda a, b: a >= b, "Gt": lambda a, b: a > b,
            "Le": lambda a, b: a <= b, "Lt": lambda a, b: a < b,
            "Eq": lambda a, b: a == b, "Neq": lambda a, b: a != b,
        },
        "select": lambda c, t, f: jnp.where(c, t, f),
    }

    def rhs(x, u, p):
        x = jnp.asarray(x, dtype=jnp.float64)
        u = jnp.asarray(u, dtype=jnp.float64)
        p = jnp.asarray(p, dtype=jnp.float64)
        P = jnp.concatenate([p, u, jnp.zeros(pad)]) if pad > 0 else jnp.concatenate([p, u])
        outputs = _run(programs, lambda i: x[i], lambda i: P[i], ops)
        return jnp.stack(outputs)

    return rhs, meta
