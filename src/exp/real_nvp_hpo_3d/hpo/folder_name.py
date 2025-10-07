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


def make_run_folder_name(
    cfg: Mapping[str, Any],
    dataset: str,
    *,
    prefix: str = "nvp-3d-f64",
    key_alias: Mapping[str, str] | None = None,
    ordered_keys: Sequence[str] = (
        "hidden",
        "num_flows",
        "num_hidden_channels",
        "seed",
        "lr",
        "weight_decay",
    ),
) -> str:
    """
    Build a compact folder name. Always appends '_an'.

    Example
    -------
    nvp_qm9_h1600_f6_hid1024_s42_lr1e-3_wd1e-4_an
    """
    aliases = {
        "hidden": "h",
        "batch_size": "bs",
        "lr": "lr",
        "weight_decay": "wd",
        "num_flows": "f",
        "num_hidden_channels": "hid",
        "smax_initial": "smi",
        "smax_final": "smf",
        "smax_warmup_epochs": "smw",
        "seed": "s",
        **(key_alias or {}),
    }

    parts: list[str] = [prefix, dataset.lower()]

    def render(k: str, v: Any) -> str:
        alias = aliases.get(k, k[:3])
        return f"{alias}{_fmt_value(v)}"

    seen: set[str] = set()
    for k in ordered_keys:
        if k in cfg:
            parts.append(render(k, cfg[k]))
            seen.add(k)

    for k in sorted(cfg.keys()):
        if k not in seen:
            parts.append(render(k, cfg[k]))

    parts.append("an")  # always add affine-norm tag
    return "_".join(parts)
