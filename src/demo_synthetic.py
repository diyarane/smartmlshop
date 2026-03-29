"""Deterministic synthetic demand, sales, profit, and inventory for demo (no ML)."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd


def anchor_store_daily_demand(df: Optional[pd.DataFrame]) -> Optional[float]:
    """Last calendar day store-wide sum of demand/quantity; anchors forecast lag_1."""
    if df is None or df.empty:
        return None
    w = df.copy()
    w.columns = [str(c).strip().lower() for c in w.columns]
    if 'date' not in w.columns:
        return None
    w['date'] = pd.to_datetime(w['date'], errors='coerce')
    w = w.dropna(subset=['date'])
    qcol = 'quantity' if 'quantity' in w.columns else ('demand' if 'demand' in w.columns else None)
    if qcol is None:
        return None
    daily = (
        w.groupby(w['date'].dt.normalize(), as_index=True)[qcol]
        .sum()
        .sort_index()
    )
    if daily.empty:
        return None
    return float(daily.iloc[-1])


def demo_seed(
    product_id: int | None,
    category: int,
    unit_price: float,
    product_name: str | None,
) -> int:
    if product_id is not None:
        return int((product_id * 1_000_003 + int(category or 0) * 17) % (2**31 - 1))
    key = f"{product_name or 'custom'}|{category}|{unit_price:.4f}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def product_demo_bundle(seed: int, unit_price: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    base = float(rng.uniform(200, 400))

    price_base = float(rng.uniform(20, 100))
    if unit_price and float(unit_price) > 0:
        price_base = float(np.clip(0.45 * float(unit_price) + 0.55 * price_base, 20, 100))

    cost_ratio = float(rng.uniform(0.6, 0.8))
    profit_margin = max(0.05, 1.0 - cost_ratio)

    today = datetime.now().date()
    hist_start = today - timedelta(days=30)
    hist_dates: list[str] = []
    hist_demand: list[float] = []

    for i in range(30):
        d = hist_start + timedelta(days=i)
        hist_dates.append(d.strftime('%Y-%m-%d'))
        wd = d.weekday()
        weekend = 1.32 if wd >= 5 else 1.0
        trend = 1.0 + (i / 29.0) * 0.08
        noise = float(rng.normal(0, 14.0))
        q = base * weekend * trend + noise
        hist_demand.append(round(max(8.0, q), 2))

    last_hist = datetime.strptime(hist_dates[-1], '%Y-%m-%d').date()
    fc_dates: list[str] = []
    fc_demand: list[float] = []

    for i in range(1, 31):
        d = last_hist + timedelta(days=i)
        fc_dates.append(d.strftime('%Y-%m-%d'))
        wd = d.weekday()
        weekend = 1.32 if wd >= 5 else 1.0
        trend = 1.0 + (i / 30.0) * 0.12
        noise = float(rng.normal(0, 16.0))
        q = base * weekend * trend + noise
        fc_demand.append(round(max(8.0, q), 2))

    def sales_from_demands(demands: list[float]) -> list[float]:
        return [
            round(max(0.01, dem * price_base * float(rng.uniform(0.9, 1.1))), 2)
            for dem in demands
        ]

    def profits_from_sales(sales: list[float]) -> list[float]:
        return [
            round(max(0.01, s * profit_margin * float(rng.uniform(0.92, 1.08))), 2)
            for s in sales
        ]

    hist_sales = sales_from_demands(hist_demand)
    fc_sales = sales_from_demands(fc_demand)
    hist_profit = profits_from_sales(hist_sales)
    fc_profit = profits_from_sales(fc_sales)

    daily_demand = float(np.mean(fc_demand))
    lead_time = 5
    safety_stock = daily_demand * 2.0
    reorder_point = daily_demand * float(lead_time) + safety_stock

    # Days-of-cover 5–12 × daily demand (seed-stable); not DB stock — avoids all-CRITICAL demos
    cover_days = float(rng.uniform(5.0, 12.0))
    stock_i = int(max(1, round(daily_demand * cover_days)))

    recommended_order = float(max(0.0, reorder_point - stock_i + safety_stock))

    if stock_i <= reorder_point:
        status = 'Critical'
    elif stock_i <= 1.2 * reorder_point:
        status = 'Low'
    else:
        status = 'OK'

    all_dates = hist_dates + fc_dates

    return {
        'dates': all_dates,
        'demand': {'historical': hist_demand, 'forecast': fc_demand},
        'sales': {'historical': hist_sales, 'forecast': fc_sales},
        'profit': {'historical': hist_profit, 'forecast': fc_profit},
        'inventory': {
            'current_stock': stock_i,
            'reorder_point': round(reorder_point, 2),
            'safety_stock': round(safety_stock, 2),
            'recommended_order': round(recommended_order, 2),
            'status': status,
            'daily_demand': round(daily_demand, 2),
        },
    }


def bundle_to_legacy_historical(b: dict[str, Any]) -> dict[str, Any]:
    """Keys used by product management without changing the frontend."""
    dh = b['demand']['historical']
    ds = b['demand']['forecast']
    sh = b['sales']['historical']
    ph = b['profit']['historical']
    hdates = b['dates'][:30]
    fdates = b['dates'][30:60]
    return {
        'dates': hdates,
        'demand': dh,
        'sales': sh,
        'profit': ph,
        'historical': dh,
        'forecast': ds,
        'forecast_dates': fdates,
        'timeline_dates': b['dates'],
    }


def store_demo_forecast(
    days: int,
    seed: int = 9_000_011,
    anchor_demand: Optional[float] = None,
) -> dict[str, Any]:
    """
    Store-level series for /api/forecast: calendar-varying future features, lag_1 from
    last historical demand, multiplicative jitter, rolling-3 smoothing (no weekly spike clone).
    """
    rng = np.random.default_rng(seed)
    days = max(1, min(int(days), 90))
    lag_1 = (
        float(anchor_demand)
        if anchor_demand is not None and float(anchor_demand) > 0
        else float(rng.uniform(2200, 4600))
    )
    price = float(rng.uniform(28, 72))
    margin = float(rng.uniform(0.22, 0.36))

    future_dates: list[str] = []
    demand_pre: list[float] = []

    start = datetime.now().date()
    for i in range(1, days + 1):
        d = start + timedelta(days=i)
        ds = d.strftime('%Y-%m-%d')
        day_v = float(d.day)
        month_v = float(d.month)
        weekday_v = float(d.weekday())
        cal = (
            0.94
            + 0.06 * np.sin(2.0 * np.pi * weekday_v / 7.0)
            + 0.05 * np.sin(2.0 * np.pi * month_v / 12.0)
            + 0.04 * (day_v / 31.0)
        )
        horizon = 1.0 + 0.05 * (i / float(max(days, 1)))
        q = lag_1 * cal * horizon + float(rng.normal(0, max(lag_1 * 0.028, 5.0)))
        q = max(40.0, q)
        demand_pre.append(q)
        lag_1 = 0.42 * lag_1 + 0.58 * q
        future_dates.append(ds)

    arr = np.maximum(40.0, np.array(demand_pre, dtype=float))
    arr = arr * rng.uniform(0.9, 1.1, size=len(arr))
    sm = (
        pd.Series(arr)
        .rolling(window=3, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=float)
    )
    demand_forecast = np.round(sm, 2).tolist()

    sales_pre: list[float] = []
    profit_pre: list[float] = []
    for q in demand_forecast:
        rev = max(1.0, float(q) * price * float(rng.uniform(0.97, 1.03)))
        prof = max(0.5, rev * margin * float(rng.uniform(0.97, 1.03)))
        sales_pre.append(rev)
        profit_pre.append(prof)
    sales_forecast = (
        pd.Series(sales_pre)
        .rolling(window=3, min_periods=1, center=True)
        .mean()
        .round(2)
        .tolist()
    )
    profit_forecast = (
        pd.Series(profit_pre)
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
