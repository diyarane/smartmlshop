"""Time-series style forecasts from historical shop data."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, List, Optional

import numpy as np
import pandas as pd


def generate_forecast(
    df: pd.DataFrame,
    days: int = 30,
    festival_days: Optional[List[str]] = None,
) -> dict[str, Any]:
    """
    Build daily demand, revenue, and profit forecasts for the next ``days`` days.
    Uses evolving lag_1 (from last historical demand), non-constant calendar features
    (day / month / weekday), multiplicative jitter, and rolling-mean smoothing.
    """
    festival_days = festival_days or []
    empty = {
        'future_dates': [],
        'demand_forecast': [],
        'sales_forecast': [],
        'profit_forecast': [],
    }
    if df is None or df.empty:
        return empty

    work = df.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]
    work['date'] = pd.to_datetime(work['date'], errors='coerce')
    work = work.dropna(subset=['date'])
    if work.empty:
        return empty

    qty_col = 'quantity' if 'quantity' in work.columns else 'demand'
    rev_col = 'revenue' if 'revenue' in work.columns else 'sales'
    if qty_col not in work.columns or rev_col not in work.columns or 'profit' not in work.columns:
        return empty

    daily = (
        work.groupby(work['date'].dt.normalize())
        .agg(
            quantity=(qty_col, 'sum'),
            revenue=(rev_col, 'sum'),
            profit=('profit', 'sum'),
        )
        .reset_index()
        .sort_values('date')
    )

    if daily.empty:
        return empty

    tail = daily.tail(max(14, min(56, len(daily))))
    wd_q = tail.groupby(tail['date'].dt.weekday)['quantity'].mean()
    base_q = float(daily['quantity'].tail(14).mean() or 0)

    qty14 = daily['quantity'].tail(14).astype(float)
    mu_q = float(qty14.mean() or 1.0)

    yq = daily['quantity'].tail(30).values.astype(float)
    n = len(yq)
    if n >= 5:
        x = np.arange(n, dtype=float)
        slope_q, _intercept_q = np.polyfit(x, yq, 1)
    else:
        slope_q = 0.0

    slope_norm = float(slope_q / max(mu_q, 1e-6))
    seed = int(abs(int(daily['quantity'].sum()) % (2**31 - 1)))
    rng = np.random.default_rng(seed)

    lag_1 = float(daily['quantity'].iloc[-1])
    unit_rev = float(
        daily['revenue'].tail(14).sum() / max(daily['quantity'].tail(14).sum(), 1.0)
    )
    unit_pr = float(
        daily['profit'].tail(14).sum() / max(daily['quantity'].tail(14).sum(), 1.0)
    )

    future_dates: list[str] = []
    raw_demand: list[float] = []

    start = datetime.now().date()
    for i in range(1, days + 1):
        d = start + timedelta(days=i)
        ds = d.strftime('%Y-%m-%d')
        day_v = float(d.day)
        month_v = float(d.month)
        weekday_v = float(d.weekday())

        wd_lift = (
            float(wd_q.get(d.weekday(), base_q)) / max(base_q, 1e-6) if len(wd_q) else 1.0
        )
        wd_lift = float(np.clip(wd_lift, 0.82, 1.22))

        cal = (
            0.96
            + 0.04 * np.sin(2.0 * np.pi * month_v / 12.0)
            + 0.035 * np.sin(2.0 * np.pi * weekday_v / 7.0)
            + 0.025 * (day_v / 31.0)
        )
        horizon = 1.0 + slope_norm * (i / max(days, 1)) * 0.65
        q = lag_1 * (0.48 + 0.52 * wd_lift) * cal * horizon
        q = max(0.01, q + float(rng.normal(0, max(mu_q * 0.022, 0.5))))
        if ds in festival_days:
            q *= 1.12
        raw_demand.append(q)
        lag_1 = 0.44 * lag_1 + 0.56 * q
        future_dates.append(ds)

    arr_d = np.array(raw_demand, dtype=float) * rng.uniform(0.9, 1.1, size=len(raw_demand))
    demand_smooth = (
        pd.Series(arr_d).rolling(window=3, min_periods=1, center=True).mean().to_numpy(dtype=float)
    )

    demand_forecast = np.round(np.maximum(0.01, demand_smooth), 2).tolist()

    raw_rev = np.maximum(0.01, demand_smooth * max(unit_rev, 1.0))
    raw_pr = np.maximum(0.01, demand_smooth * max(unit_pr, 0.01))
    raw_rev = raw_rev * rng.uniform(0.97, 1.03, size=len(raw_rev))
    raw_pr = raw_pr * rng.uniform(0.97, 1.03, size=len(raw_pr))

    sales_forecast = (
        pd.Series(raw_rev)
        .rolling(window=3, min_periods=1, center=True)
        .mean()
        .round(2)
        .tolist()
    )
    profit_forecast = (
        pd.Series(raw_pr)
        .rolling(window=3, min_periods=1, center=True)
        .mean()
        .round(2)
        .tolist()
    )

    return {
        'future_dates': future_dates,
        'demand_forecast': demand_forecast,
        'sales_forecast': sales_forecast,
        'profit_forecast': profit_forecast,
    }
