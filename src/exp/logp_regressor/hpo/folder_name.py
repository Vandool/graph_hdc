from collections.abc import Mapping, Sequence
from typing import Any


def _fmt_float(x: float) -> str:
    if x == 0:
        return "0"
    s = f"{x:.6g}"
    if "e" in s:
        base, exp = s.split("e")
        return f"{base}e{int(exp)}"
    return s


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return _fmt_float(v)
    return str(v).replace(" ", "").replace("/", "_")


# --- helpers specific to the MLP regressor space ---
def _derive_hidden(cfg: Mapping[str, Any]) -> str | None:
    """
    Prefer explicit cfg['hidden'] if provided; otherwise build from depth/h1..h3.
    Returns e.g. '1024-256-64' or None if not derivable.
    """
    if cfg.get("hidden"):
        # Accept list/tuple/str
        h = cfg["hidden"]
        if isinstance(h, (list, tuple)):
            return "-".join(str(int(x)) for x in h)
        return str(h)
    # Build from depth + h1..h3 (our Optuna space)
    d = int(cfg.get("depth", 0) or 0)
    if d <= 0:
        return None
    hs = []
    if d >= 1 and "h1" in cfg:
        hs.append(int(cfg["h1"]))
    if d >= 2 and "h2" in cfg:
        hs.append(int(cfg["h2"]))
    if d >= 3 and "h3" in cfg:
        hs.append(int(cfg["h3"]))
    return "-".join(map(str, hs)) if hs else None


def _normalize_norm(norm_val: Any) -> str:
    # map to short tokens
    s = (str(norm_val) if norm_val is not None else "none").lower()
    if s in ("lay_norm", "layernorm", "layer_norm", "ln"):
        return "ln"
    if s in ("batch_norm", "batchnorm", "bn"):
        return "bn"
    if s in ("none", "", "null", "false", "0"):
        return "none"
    return s


def _normalize_act(act_val: Any) -> str:
    s = str(act_val).lower()
    if s == "leakyrelu":
        return "lrelu"
    return s  # relu/gelu/silu already concise


def make_run_folder_name(
    cfg: Mapping[str, Any],
    dataset: str,
    *,
    prefix: str = "lpr",
    key_alias: Mapping[str, str] | None = None,
    ordered_keys: Sequence[str] = (
        "hidden",  # derived from depth/h1..h3 if not explicitly given
        "activation",
        "norm",
        "dropout",
        "batch_size",
        "lr",
        "weight_decay",
        "seed",
    ),
) -> str:
    """
    Build a compact folder name reflecting MLP regressor HPs.

    Examples:
      lpr_qm9_h1024-256-64_actgelu_nmln_dp0_lr1e-3_wd1e-4_s42
      lpr_zinc_h1536-384_actrelu_nmbn_dp0.2_lr5e-4_wd0_s7
    """
    aliases = {
        "hidden": "h",
        "activation": "act",
        "norm": "nm",
        "dropout": "dp",
        "batch_size": "bs",
        "lr": "lr",
        "weight_decay": "wd",
        "seed": "s",
        **(key_alias or {}),
    }

    parts: list[str] = [prefix, dataset.lower()]

    # Ensure 'hidden' is present (derive from depth/h1..h3 if needed)
    derived_hidden = _derive_hidden(cfg)
    if derived_hidden is not None:
        cfg = dict(cfg)  # shallow copy
        cfg.setdefault("hidden", derived_hidden)

    def render(k: str, v: Any) -> str:
        alias = aliases.get(k, k[:3])
        # special formatting
        if k == "norm":
            v = _normalize_norm(v)
        elif k == "activation":
            v = _normalize_act(v)
        elif k == "dropout":
            # keep short like 0, 0.1, 0.25
            v = _fmt_float(float(v))
        elif k == "hidden" and isinstance(v, (list, tuple)):
            v = "-".join(str(int(x)) for x in v)
        return f"{alias}{_fmt_value(v)}"

    seen: set[str] = set()
    for k in ordered_keys:
        if k in cfg and cfg[k] is not None:
            parts.append(render(k, cfg[k]))
            seen.add(k)

    # append any remaining keys (sorted) to avoid losing info
    for k in sorted(cfg.keys()):
        if k not in seen and cfg[k] is not None:
            parts.append(render(k, cfg[k]))

    return "_".join(parts)
