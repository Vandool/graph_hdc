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
    *,
    prefix: str = "fm",
    key_alias: Mapping[str, str] | None = None,
    ordered_keys: Sequence[str] = (
        "num_flows",
        "num_hidden_channels",
        "num_bins",
        "num_blocks",
        "seed",
        "lr",
        "weight_decay",
        "batch_size",
        "dropout_probability",
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
        "num_flows": "nf",
        "num_hidden_channels": "nh",
        "num_bins": "nb",
        "num_blocks": "b",
        "seed": "s",
        "per_term_standardization": "np",  # norm_per: np1=term, np0=dim
        "dropout_probability": "dp",
        **(key_alias or {}),
    }

    parts: list[str] = [prefix]

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
