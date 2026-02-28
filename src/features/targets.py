from __future__ import annotations

import numpy as np
import pandas as pd


def _forward_compound_return(ret_1d: pd.Series, horizon: int) -> pd.Series:
    future = (1.0 + ret_1d).shift(-1)
    compounded = future.iloc[::-1].rolling(window=horizon, min_periods=horizon).apply(np.prod, raw=True).iloc[::-1]
    return compounded - 1.0


def build_binary_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    grouped = out.groupby("symbol", group_keys=False)

    if "adj_close" in out.columns:
        out["fwd_ret_1d"] = grouped["adj_close"].shift(-1) / out["adj_close"] - 1.0
        out["fwd_ret_target"] = grouped["adj_close"].shift(-horizon) / out["adj_close"] - 1.0
    elif "ret_1d" in out.columns:
        out["fwd_ret_1d"] = grouped["ret_1d"].shift(-1)
        out["fwd_ret_target"] = grouped["ret_1d"].transform(lambda s: _forward_compound_return(s, horizon))
    else:
        raise KeyError("build_binary_target requires either 'adj_close' or 'ret_1d'.")

    out["target_horizon_days"] = int(horizon)
    out["y_up"] = (out["fwd_ret_target"] > 0).astype("Int64")
    out.loc[out["fwd_ret_target"].isna(), "y_up"] = pd.NA
    return out
