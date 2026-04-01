import sys
import os

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from flask import Blueprint, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from app.auth import login_required
from src.predict import Predictor
from src.preprocessing import DataPreprocessor
from src.demo_synthetic import (
    anchor_store_daily_demand,
    bundle_to_legacy_historical,
    demo_seed,
    product_demo_bundle,
    store_demo_forecast,
)

from app.models import db, User, Employee, Product, Sale, Inventory

routes_bp = Blueprint('routes', __name__)
logger = logging.getLogger(__name__)

# Initialize components
predictor = Predictor()
preprocessor = DataPreprocessor()


def _inventory_units_for_ui(inv):
    """Current on-hand from Inventory; if stock_level is unset/zero, use optimal_stock (same row)."""
    if inv is None:
        return 0
    sl = inv.stock_level
    if sl is not None and sl > 0:
        return int(sl)
    return int(inv.optimal_stock or 0)


def _merge_product_demo_context(data: dict) -> dict:
    out = dict(data or {})
    pid = out.get('product_id')
    if pid is None:
        return out
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return out
    p = Product.query.get(pid_int)
    if not p:
        return out
    if out.get('unit_price') is None:
        out['unit_price'] = float(p.price)
    if out.get('category') is None:
        out['category'] = int(p.category or 0)
    if not out.get('name') and not out.get('product_name'):
        out['name'] = p.name
    return out


def _resolve_product_demo_bundle(data: dict) -> dict:
    out = _merge_product_demo_context(dict(data or {}))
    pid_val = None
    if out.get('product_id') is not None:
        try:
            pid_val = int(out['product_id'])
        except (TypeError, ValueError):
            pid_val = None

    unit_price = float(out.get('unit_price') or 100)
    category = int(out.get('category') if out.get('category') is not None else 0)
    name = out.get('name') or out.get('product_name')
    seed = demo_seed(pid_val, category, unit_price, name)
    return product_demo_bundle(seed, unit_price)


def get_dashboard_data():
    """Get common data for dashboards"""
    df = preprocessor.load_raw_data()
    
    if df is None or df.empty:
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    today_data = df[df['date'] == latest_date]

    # Raw CSV uses product_name; generated sample data uses product
    product_col = 'product_name' if 'product_name' in df.columns else 'product'
    revenue_col = 'revenue' if 'revenue' in df.columns else 'sales'
    
    metrics = {
        'total_revenue': today_data[revenue_col].sum() if not today_data.empty else 0,
        'total_profit': today_data['profit'].sum() if not today_data.empty else 0,
        'total_transactions': len(today_data),
        'avg_order_value': today_data[revenue_col].mean() if not today_data.empty else 0
    }
    
    product_sales = df.groupby(product_col)[revenue_col].sum().sort_values(ascending=False).head(10)
    category_sales = df.groupby('category')[revenue_col].sum()
    
    last_30_days = df[df['date'] >= (latest_date - timedelta(days=30))]
    daily_sales = last_30_days.groupby(last_30_days['date'].dt.date)[revenue_col].sum()
    daily_sales.index = daily_sales.index.astype(str)
    
    return {
        'metrics': metrics,
        'top_products': product_sales.to_dict(),
        'category_sales': category_sales.to_dict(),
        'daily_sales': daily_sales.to_dict(),
        'latest_date': latest_date
    }

@routes_bp.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@routes_bp.route('/manager-dashboard')
@login_required(role='manager')
def manager_dashboard():
    """Manager dashboard"""
    data = get_dashboard_data()
    if data is None:
        return render_template('manager_dashboard.html', error="No data available")
    return render_template('manager_dashboard.html', **data)

@routes_bp.route('/employee-dashboard')
@login_required(role='employee')
def employee_dashboard():
    """Employee dashboard — KPIs load from /api/employee/<id>/metrics in the browser."""
    return render_template(
        'employee_dashboard.html',
        employee_name=session['user']['name'],
        employee_id=session['user'].get('employee_id'),
        user_role=session['user'].get('role', 'employee'),
        now=datetime.now(),
    )

# ── API: Predictions ────────────────────────────────────────────────────────

@routes_bp.route('/api/predict', methods=['POST'])
@login_required()
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        predictions = predictor.predict_all(data)
        if predictions['sales'] is None:
            return jsonify({'error': 'Prediction failed'}), 500
        return jsonify(predictions)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@routes_bp.route('/api/sales-forecast', methods=['GET'])
@login_required()
def sales_forecast():
    """Get sales forecast for next days"""
    try:
        days = request.args.get('days', default=7, type=int)
        df = preprocessor.load_raw_data()
        if df is None or df.empty:
            return jsonify({'error': 'No data available'}), 404

        df['date'] = pd.to_datetime(df['date'])
        last_30_days = df[df['date'] >= (df['date'].max() - timedelta(days=30))]
        avg_daily_sales = last_30_days.groupby(last_30_days['date'].dt.weekday)['revenue'].mean()

        forecast = []
        current_date = datetime.now()
        for i in range(days):
            forecast_date = current_date + timedelta(days=i)
            weekday = forecast_date.weekday()
            predicted_sales = avg_daily_sales.get(weekday, avg_daily_sales.mean())
            forecast.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'predicted_sales': float(predicted_sales)
            })
        return jsonify(forecast)
    except Exception as e:
        logger.error(f"Sales forecast error: {e}")
        return jsonify({'error': str(e)}), 500

@routes_bp.route('/api/get-historical-data', methods=['POST'])
@login_required()
def get_historical_data():
    """30-day synthetic history + 30-day forecast (deterministic per product; same as other demo APIs)."""
    try:
        data = request.get_json() or {}
        b = _resolve_product_demo_bundle(data)
        return jsonify(bundle_to_legacy_historical(b))
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return jsonify(
            bundle_to_legacy_historical(
                product_demo_bundle(demo_seed(None, 0, 75.0, 'fallback'), 75.0)
            )
        )


def generate_sample_historical_data():
    b = product_demo_bundle(demo_seed(None, 0, 75.0, 'sample'), 75.0)
    return bundle_to_legacy_historical(b)

# ── API: Employee metrics ───────────────────────────────────────────────────

@routes_bp.route('/api/employee-metrics', methods=['GET'])
@login_required()
def get_all_employee_metrics():
    """Get metrics for all employees"""
    try:
        employees = Employee.query.all()
        metrics = []
        thirty_days_ago = datetime.now() - timedelta(days=30)

        for employee in employees:
            sales = Sale.query.filter(
                Sale.employee_id == employee.id,
                Sale.date >= thirty_days_ago
            ).all()

            total_sales = sum(s.total_amount for s in sales)
            total_transactions = len(sales)
            total_profit = sum(s.profit for s in sales)
            avg_transaction = total_sales / total_transactions if total_transactions > 0 else 0
            performance_rating = min(10, total_sales / 5000) if total_sales > 0 else 0

            metrics.append({
                'id': employee.id,
                'name': employee.name,
                'position': employee.position,
                'total_sales': total_sales,
                'total_transactions': total_transactions,
                'total_profit': total_profit,
                'avg_transaction': avg_transaction,
                'performance_rating': performance_rating
            })

        metrics.sort(key=lambda x: x['total_sales'], reverse=True)
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting employee metrics: {e}")
        return jsonify([])

@routes_bp.route('/api/employee/<int:employee_id>/metrics', methods=['GET'])
@login_required()
def get_single_employee_metrics(employee_id):
    """Get metrics for a single employee"""
    try:
        employee = Employee.query.get(employee_id)
        if not employee:
            return jsonify({'error': 'Employee not found'}), 404

        thirty_days_ago = datetime.now() - timedelta(days=30)
        sales = Sale.query.filter(
            Sale.employee_id == employee_id,
            Sale.date >= thirty_days_ago
        ).all()

        total_sales = sum(s.total_amount for s in sales)
        total_transactions = len(sales)
        total_profit = sum(s.profit for s in sales)
        avg_transaction = total_sales / total_transactions if total_transactions > 0 else 0

        daily_sales = {}
        product_sales = {}
        for sale in sales:
            date_str = sale.date.strftime('%Y-%m-%d')
            daily_sales[date_str] = daily_sales.get(date_str, 0) + sale.total_amount
            product_name = sale.product.name
            product_sales[product_name] = product_sales.get(product_name, 0) + sale.total_amount

        top_products = dict(sorted(product_sales.items(), key=lambda x: x[1], reverse=True)[:5])
        performance_rating = min(10, total_sales / 5000) if total_sales > 0 else 0

        return jsonify({
            'id': employee.id,
            'name': employee.name,
            'position': employee.position,
            'hire_date': employee.hire_date.strftime('%Y-%m-%d'),
            'total_sales': total_sales,
            'total_transactions': total_transactions,
            'total_profit': total_profit,
            'avg_transaction': avg_transaction,
            'performance_rating': performance_rating,
            'daily_sales': daily_sales,
            'top_products': top_products
        })
    except Exception as e:
        logger.error(f"Error getting employee metrics: {e}")
        return jsonify({'error': str(e)}), 500

# ── API: Demand / Sales / Profit / Inventory (single set, no duplicates) ───

@routes_bp.route('/api/predict-demand', methods=['POST'])
@login_required()
def predict_demand_api():
    try:
        data = dict(request.get_json() or {})
        data.pop('historical', None)
        b = _resolve_product_demo_bundle(data)
        fc = b['demand']['forecast']
        total = float(sum(fc))
        return jsonify({'demand': max(1.0, total), 'confidence': 0.9})
    except Exception as e:
        logger.error(f"Demand prediction error: {e}")
        return jsonify({'demand': 500.0, 'confidence': 0.0})

@routes_bp.route('/api/predict-sales', methods=['POST'])
@login_required()
def predict_sales_api():
    try:
        data = dict(request.get_json() or {})
        data.pop('historical', None)
        b = _resolve_product_demo_bundle(data)
        fc_d = b['demand']['forecast']
        fc_s = b['sales']['forecast']
        predicted_demand = float(sum(fc_d))
        predicted_sales = float(sum(fc_s))
        return jsonify({'sales': predicted_sales, 'demand': predicted_demand})
    except Exception as e:
        logger.error(f"Sales prediction error: {e}")
        return jsonify({'sales': 50000.0, 'demand': 500.0})

@routes_bp.route('/api/predict-profit', methods=['POST'])
@login_required()
def predict_profit_api():
    try:
        data = dict(request.get_json() or {})
        data.pop('historical', None)
        b = _resolve_product_demo_bundle(data)
        fc_s = b['sales']['forecast']
        fc_p = b['profit']['forecast']
        predicted_sales = float(sum(fc_s))
        predicted_profit = float(sum(fc_p))
        margin = float(predicted_profit / max(predicted_sales, 1e-6))
        return jsonify({'profit': predicted_profit, 'margin': margin})
    except Exception as e:
        logger.error(f"Profit prediction error: {e}")
        return jsonify({'profit': 17500.0, 'margin': 0.35})

@routes_bp.route('/api/optimize-inventory', methods=['POST'])
@login_required()
def optimize_inventory_api():
    try:
        data = dict(request.get_json() or {})
        data.pop('historical', None)
        b = _resolve_product_demo_bundle(data)
        invb = b['inventory']
        fc_d = b['demand']['forecast']
        predicted_demand = float(sum(fc_d))
        reorder_point = float(invb['reorder_point'])
        current_stock = int(invb['current_stock'])
        recommended = float(invb['recommended_order'])
        daily_demand = float(invb['daily_demand'])
        should_order = bool(current_stock <= reorder_point)
        if should_order:
            recommended = max(recommended - current_stock, 50.0)
        return jsonify({
            'reorder_point': max(10.0, round(reorder_point, 2)),
            'optimal_order_quantity': max(0.0, round(recommended, 2)),
            'predicted_demand': predicted_demand,
            'should_order': should_order,
            'daily_demand': daily_demand,
        })
    except Exception as e:
        logger.error(f"Inventory optimization error: {e}")
        return jsonify({'reorder_point': 100.0, 'optimal_order_quantity': 200.0,
                        'predicted_demand': 1000.0, 'should_order': True, 'daily_demand': 33.0})

@routes_bp.route('/api/optimize-profit', methods=['POST'])
@login_required()
def optimize_profit_api():
    try:
        data = request.get_json()
        profit_response = predict_profit_api()
        profit_data = profit_response.get_json()
        base_profit = profit_data['profit']

        scenarios = []
        for discount in [0, 5, 10, 15, 20, 25, 30]:
            demand_multiplier = 1 + (discount / 100) * 1.2
            profit_multiplier = demand_multiplier * (1 - discount / 100)
            scenarios.append({
                'discount_pct': discount,
                'predicted_profit': max(0, base_profit * profit_multiplier),
                'demand_multiplier': demand_multiplier
            })

        optimal = max(scenarios, key=lambda x: x['predicted_profit'])
        return jsonify({
            'base_profit': base_profit,
            'optimal_profit': optimal['predicted_profit'],
            'optimal_discount': optimal['discount_pct'],
            'scenarios': scenarios
        })
    except Exception as e:
        logger.error(f"Profit optimization error: {e}")
        return jsonify({
            'base_profit': 50000, 'optimal_profit': 57500, 'optimal_discount': 10,
            'scenarios': [
                {'discount_pct': 0, 'predicted_profit': 50000},
                {'discount_pct': 5, 'predicted_profit': 55000},
                {'discount_pct': 10, 'predicted_profit': 57500},
                {'discount_pct': 15, 'predicted_profit': 56000},
                {'discount_pct': 20, 'predicted_profit': 54000}
            ]
        })
    
@routes_bp.route('/api/full-forecast', methods=['GET'])
@login_required(role='manager')
def full_forecast():
    try:
        df_fc = preprocessor.load_raw_data()
        anchor = anchor_store_daily_demand(df_fc)
        sf = store_demo_forecast(30, seed=9_100_777, anchor_demand=anchor)
        forecasts = []
        for ds, q, r, p in zip(
            sf['future_dates'],
            sf['demand_forecast'],
            sf['sales_forecast'],
            sf['profit_forecast'],
        ):
            forecasts.append({
                'date': ds,
                'sales': float(r),
                'demand': float(q),
                'profit': float(p),
            })
        return jsonify(forecasts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/api/ml/optimize-inventory', methods=['POST'])
@login_required()
def optimize_inventory_ml():
    """ML-backed inventory optimization using trained demand model."""
    try:
        data = request.get_json() or {}
        rec = predictor.optimize_inventory(data)
        if rec is None:
            return jsonify({'error': 'ML optimization unavailable; train models first.'}), 503
        return jsonify(rec)
    except Exception as e:
        logger.error(f"ML inventory optimization error: {e}")
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/api/ml/optimize-profit', methods=['POST'])
@login_required()
def optimize_profit_ml():
    """ML-backed profit scenario search using trained profit model."""
    try:
        data = request.get_json() or {}
        rec = predictor.optimize_profit(data)
        if rec is None:
            return jsonify({'error': 'ML optimization unavailable; train models first.'}), 503
        return jsonify(rec)
    except Exception as e:
        logger.error(f"ML profit optimization error: {e}")
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/api/forecast', methods=['POST'])
@login_required()
def forecast_api():
    """Demo store-level synthetic forecast (same response shape as forecast_engine)."""
    try:
        data = request.get_json() or {}
        days = int(data.get('days', 30))
        df_fc = preprocessor.load_raw_data()
        anchor = anchor_store_daily_demand(df_fc)
        result = store_demo_forecast(days, seed=9_000_011, anchor_demand=anchor)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/products')
@login_required()
def product_management():
    rows = (
        db.session.query(Product, Inventory)
        .outerjoin(Inventory, Inventory.product_id == Product.id)
        .order_by(Product.id)
        .all()
    )
    products = [p for p, _ in rows]
    stock_by_product_id = {}
    for p, _inv in rows:
        _b = _resolve_product_demo_bundle({
            'product_id': p.id,
            'unit_price': float(p.price),
            'category': int(p.category or 0),
            'name': p.name,
        })
        stock_by_product_id[p.id] = int(_b['inventory']['current_stock'])
    return render_template(
        'product_management.html',
        products=products,
        stock_by_product_id=stock_by_product_id,
    )


@routes_bp.route('/api/analyze_product', methods=['POST'])
@login_required()
def analyze_product_api():
    """Validate product and return demo synthetic series + inventory (no ML)."""
    data = request.get_json() or {}
    pid = data.get('product_id')
    if pid is None:
        return jsonify({'ok': True, 'message': 'no product_id'})
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'invalid product_id'}), 400
    product = Product.query.get(pid)
    if product is None:
        return jsonify({'ok': False, 'error': 'unknown product'}), 404
    payload = {
        'product_id': pid,
        'unit_price': float(product.price),
        'category': int(product.category or 0),
        'name': product.name,
    }
    b = _resolve_product_demo_bundle(payload)
    invb = b['inventory']
    return jsonify({
        'ok': True,
        'product_id': pid,
        'stock': invb['current_stock'],
        'dates': b['dates'],
        'demand': b['demand'],
        'sales': b['sales'],
        'profit': b['profit'],
        'inventory': {
            'current_stock': invb['current_stock'],
            'reorder_point': invb['reorder_point'],
            'recommended_order': invb['recommended_order'],
            'status': invb['status'],
        },
    })


@routes_bp.route('/api/inventory-status', methods=['GET'])
@login_required()
def inventory_status():
    """Per-product inventory using the same synthetic forecast + reorder math as product APIs."""
    try:
        rows = (
            db.session.query(Product, Inventory)
            .outerjoin(Inventory, Inventory.product_id == Product.id)
            .order_by(Product.id)
            .all()
        )

        result = []
        for product, inv in rows:
            b = _resolve_product_demo_bundle({
                'product_id': product.id,
                'unit_price': float(product.price),
                'category': int(product.category or 0),
                'name': product.name,
            })
            invb = b['inventory']
            fc = b['demand']['forecast']
            forecast_30d = int(round(sum(fc)))
            daily_demand = float(np.mean(fc))
            reorder = int(round(invb['reorder_point']))
            status_map = {'Critical': 'critical', 'Low': 'low', 'OK': 'ok'}
            status = status_map.get(invb['status'], 'ok')
            suggested = int(round(max(0.0, invb['recommended_order'])))
            stock = int(invb['current_stock'])

            result.append({
                'id': product.id,
                'name': product.name,
                'category': product.category,
                'price': product.price,
                'stock': stock,
                'reorder_point': reorder,
                'forecast_30d': forecast_30d,
                'daily_demand': round(daily_demand, 1),
                'status': status,
                'suggested_order': suggested,
            })

        order = {'critical': 0, 'low': 1, 'ok': 2}
        result.sort(key=lambda x: order[x['status']])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Inventory status error: {e}")
        return jsonify([]), 500


@routes_bp.route('/api/inventory-insights', methods=['GET'])
@login_required(role='manager')
def inventory_insights():
    try:
        products = Product.query.all()

        insights = []

        for p in products:
            b = _resolve_product_demo_bundle({
                'product_id': p.id,
                'unit_price': float(p.price),
                'category': int(p.category or 0),
                'name': p.name,
            })
            invb = b['inventory']
            stock_level = int(invb['current_stock'])
            rp = float(invb['reorder_point'])
            predicted_30d = float(sum(b['demand']['forecast']))

            status = "OK"
            suggestion = "Maintain stock"

            if stock_level > max(rp * 5, 200):
                status = "OVERSTOCK"
                suggestion = "Reduce stock / run discounts"
            elif invb['status'] == 'Critical':
                status = "CRITICAL"
                suggestion = "Reorder immediately"
            elif invb['status'] == 'Low':
                status = "LOW"
                suggestion = "Reorder soon"

            insights.append({
                'product': p.name,
                'stock': stock_level,
                'predicted_demand': predicted_30d,
                'status': status,
                'suggestion': suggestion
            })

        return jsonify(insights)

    except Exception as e:
        return jsonify({'error': str(e)}), 500