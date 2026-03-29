# src/insights.py
import pandas as pd
import numpy as np
from src.forecast_engine import generate_forecast


def _with_date(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a normalised __date__ datetime column."""
    work = df.copy()
    for col in ['date', 'Date', 'order_date']:
        if col in work.columns:
            work['__date__'] = pd.to_datetime(work[col], errors='coerce')
            return work
    raise ValueError("No date column found")


def get_demand_trend(df, models, forecast_days=30):
    """Return historical demand + forecast dict for export."""
    result = generate_forecast(df, days=forecast_days)
    w = _with_date(df)
    historical = (
        w.groupby(w['__date__'].dt.date, as_index=False)['quantity']
        .sum()
        .rename(columns={'__date__': 'date'})
        .assign(date=lambda x: x['date'].astype(str))
        .to_dict('records')
    )
    nu = max(df['date'].nunique(), 1) if 'date' in df.columns else 1
    return {
        'historical': historical,
        'forecast': [
            {'date': d, 'quantity': q}
            for d, q in zip(result['future_dates'], result['demand_forecast'])
        ],
        'transactions_per_day': max(len(df) / nu, 1),
    }


def get_inventory_intelligence(df, models):
    """Return per-product inventory recommendations."""
    result = generate_forecast(df, days=30)
    avg_daily_demand = float(np.mean(result['demand_forecast'])) if result['demand_forecast'] else 0.0
    products = df['product'].unique() if 'product' in df.columns else ['All Products']
    rows = []
    for product in products[:25]:
        pdf = df[df['product'] == product] if 'product' in df.columns else df
        stock = int(pdf['stock_available'].mean()) if 'stock_available' in pdf.columns else 100
        daily_demand = avg_daily_demand / max(len(products), 1)
        reorder_point = round(daily_demand * 7 + 30)
        rows.append({
            'product': product,
            'current_stock': stock,
            'daily_demand_forecast': round(daily_demand, 1),
            'reorder_point': reorder_point,
            'status': 'critical' if stock < reorder_point * 0.5 else 'low' if stock < reorder_point else 'ok',
            'recommended_order': max(0, round(reorder_point * 2 - stock)),
        })
    return rows
