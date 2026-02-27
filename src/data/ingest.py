from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import load_config


def _normalize_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(col[0]).lower().replace(" ", "_") for col in frame.columns]
    else:
        frame.columns = [str(col).lower().replace(" ", "_") for col in frame.columns]

    frame = frame.reset_index()
    frame.columns = [str(col).lower().replace(" ", "_") for col in frame.columns]
    return frame


def download_prices(symbols: list[str], start_date: str, end_date: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to run data ingest.") from exc

    frames = []
    for symbol in symbols:
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if data.empty:
            continue
        data = _normalize_yfinance_columns(data)
        frames.append(
            data
            .assign(symbol=symbol)
            .loc[:, ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]]
        )

    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"])

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["date", "symbol"]).reset_index(drop=True)
    return prices


def save_prices(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    universe = load_config("config/universe.yaml")
    prices = download_prices(
        symbols=universe["symbols"],
        start_date=universe["start_date"],
        end_date=universe.get("end_date"),
    )
    save_prices(prices, "data/processed/prices.parquet")


if __name__ == "__main__":
    main()
