from __future__ import annotations

import pandas as pd


def build_binary_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()

    if "adj_close" in out.columns:
        out["fwd_ret_1d"] = (
            out.groupby("symbol")["adj_close"].shift(-horizon) / out["adj_close"] - 1.0
        )
    elif "ret_1d" in out.columns:
        out["fwd_ret_1d"] = out.groupby("symbol")["ret_1d"].shift(-horizon)
    else:
        raise KeyError("build_binary_target requires either 'adj_close' or 'ret_1d'.")

    out["y_up"] = (out["fwd_ret_1d"] > 0).astype("Int64")
    out.loc[out["fwd_ret_1d"].isna(), "y_up"] = pd.NA
    return out
