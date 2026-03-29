"""Time-series style forecasts from historical shop data."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy.ndimage import uniform_filter1d
except ImportError:  # pragma: no cover
    uniform_filter1d = None


def generate_forecast(
    df: pd.DataFrame,
    days: int = 30,
    festival_days: Optional[List[str]] = None,
) -> dict[str, Any]:
    """
    Build daily demand, revenue, and profit forecasts for the next ``days`` days
    using recent weekday patterns and a light trend.
    """
    festival_days = festival_days or []
    if df is None or df.empty:
        return {
            'future_dates': [],
            'demand_forecast': [],
            'sales_forecast': [],
            'profit_forecast': [],
        }

    work = df.copy()
    work['date'] = pd.to_datetime(work['date'], errors='coerce')
    work = work.dropna(subset=['date'])
    if work.empty:
        return {
            'future_dates': [],
            'demand_forecast': [],
            'sales_forecast': [],
            'profit_forecast': [],
        }

    daily = (
        work.groupby(work['date'].dt.normalize())
        .agg(
            quantity=('quantity', 'sum'),
            revenue=('revenue', 'sum'),
            profit=('profit', 'sum'),
        )
        .reset_index()
        .sort_values('date')
    )

    if daily.empty:
        return {
            'future_dates': [],
            'demand_forecast': [],
            'sales_forecast': [],
            'profit_forecast': [],
        }

    tail = daily.tail(max(14, min(56, len(daily))))
    wd_q = tail.groupby(tail['date'].dt.weekday)['quantity'].mean()
    wd_r = tail.groupby(tail['date'].dt.weekday)['revenue'].mean()
    wd_p = tail.groupby(tail['date'].dt.weekday)['profit'].mean()

    base_q = float(daily['quantity'].tail(14).mean() or 0)
    base_r = float(daily['revenue'].tail(14).mean() or 0)
    base_p = float(daily['profit'].tail(14).mean() or 0)

    # Mild trend from last 30 points
    yq = daily['quantity'].tail(30).values.astype(float)
    n = len(yq)
    if n >= 5:
        x = np.arange(n, dtype=float)
        slope_q, intercept_q = np.polyfit(x, yq, 1)
    else:
        slope_q, intercept_q = 0.0, base_q

    future_dates: list[str] = []
    demand_forecast: list[float] = []
    sales_forecast: list[float] = []
    profit_forecast: list[float] = []

    start = datetime.now().date()
    for i in range(1, days + 1):
        d = start + timedelta(days=i)
        ds = d.strftime('%Y-%m-%d')
        wd = d.weekday()

        q = float(wd_q.get(wd, base_q)) if len(wd_q) else base_q
        r = float(wd_r.get(wd, base_r)) if len(wd_r) else base_r
        p = float(wd_p.get(wd, base_p)) if len(wd_p) else base_p

        trend_adj = 1.0 + slope_q * (i / max(days, 1)) * 0.02
        q = max(0.0, q * trend_adj)
        r = max(0.0, r * trend_adj)
        p = max(0.0, p * trend_adj)

        if ds in festival_days:
            q *= 1.15
            r *= 1.12
            p *= 1.12

        future_dates.append(ds)
        demand_forecast.append(round(q, 2))
        sales_forecast.append(round(r, 2))
        profit_forecast.append(round(p, 2))

    if uniform_filter1d is not None and len(demand_forecast) >= 5:
        arr_q = np.array(demand_forecast, dtype=float)
        arr_r = np.array(sales_forecast, dtype=float)
        arr_p = np.array(profit_forecast, dtype=float)
        k = min(5, len(arr_q) // 2 * 2 + 1)
        if k >= 3:
            demand_forecast = np.round(uniform_filter1d(arr_q, size=k, mode='nearest'), 2).tolist()
            sales_forecast = np.round(uniform_filter1d(arr_r, size=k, mode='nearest'), 2).tolist()
            profit_forecast = np.round(uniform_filter1d(arr_p, size=k, mode='nearest'), 2).tolist()

    return {
        'future_dates': future_dates,
        'demand_forecast': demand_forecast,
        'sales_forecast': sales_forecast,
        'profit_forecast': profit_forecast,
    }
