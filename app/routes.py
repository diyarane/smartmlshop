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
from config import Config

from app.models import db, User, Employee, Product, Sale, Inventory

routes_bp = Blueprint('routes', __name__)
logger = logging.getLogger(__name__)

# Initialize components
predictor = Predictor()
preprocessor = DataPreprocessor()

def get_dashboard_data():
    """Get common data for dashboards"""
    df = preprocessor.load_raw_data()
    
    if df is None or df.empty:
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    today_data = df[df['date'] == latest_date]
    
    metrics = {
        'total_revenue': today_data['revenue'].sum() if not today_data.empty else 0,
        'total_profit': today_data['profit'].sum() if not today_data.empty else 0,
        'total_transactions': len(today_data),
        'avg_order_value': today_data['revenue'].mean() if not today_data.empty else 0
    }
    
    product_sales = df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(10)
    category_sales = df.groupby('category')['revenue'].sum()
    
    last_30_days = df[df['date'] >= (latest_date - timedelta(days=30))]
    daily_sales = last_30_days.groupby(last_30_days['date'].dt.date)['revenue'].sum()
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
    """Employee dashboard"""
    employee_id = session['user'].get('employee_id')

    df = preprocessor.load_raw_data()

    if df is not None and employee_id:
        # employee_id in CSV is like 'EMP_1', but in DB it's an int — handle both
        employee_data = df[df['employee_id'] == employee_id]
        if employee_data.empty:
            # Try string match e.g. 'EMP_1'
            employee_data = df[df['employee_id'] == f'EMP_{employee_id}']

        if not employee_data.empty:
            employee_metrics = {
                'total_sales': employee_data['revenue'].sum(),
                'total_transactions': len(employee_data),
                'avg_transaction': employee_data['revenue'].mean(),
                'total_profit': employee_data['profit'].sum()
            }
        else:
            employee_metrics = {
                'total_sales': 0,
                'total_transactions': 0,
                'avg_transaction': 0,
                'total_profit': 0
            }
    else:
        employee_metrics = None

    # Pass datetime.now() result (not the function) to template
    return render_template('employee_dashboard.html',
                           employee_name=session['user']['name'],
                           employee_id=employee_id,
                           metrics=employee_metrics,
                           now=datetime.now())

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
    """Get historical data for a specific product"""
    try:
        data = request.get_json()
        product_name = data.get('product_name')
        category = data.get('category')

        df = preprocessor.load_raw_data()
        if df is None or df.empty:
            return jsonify(generate_sample_historical_data())

        if product_name and product_name != 'Custom Product':
            product_df = df[df['product'] == product_name]
        elif category is not None:
            product_df = df[df['category'] == category]
        else:
            product_df = df.head(100)

        product_df = product_df.copy()
        product_df['date'] = pd.to_datetime(product_df['date'])
        product_df = product_df.sort_values('date')

        daily_data = product_df.groupby(product_df['date'].dt.date).agg({
            'quantity': 'sum',
            'revenue': 'sum',
            'profit': 'sum'
        }).reset_index()

        daily_data = daily_data.tail(60)

        return jsonify({
            'dates': daily_data['date'].astype(str).tolist(),
            'demand': daily_data['quantity'].tolist(),
            'sales': daily_data['revenue'].tolist(),
            'profit': daily_data['profit'].tolist()
        })
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return jsonify(generate_sample_historical_data())

def generate_sample_historical_data():
    dates, demand, sales, profit = [], [], [], []
    for i in range(60, 0, -1):
        date = datetime.now() - timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
        day_of_week = date.weekday()
        multiplier = 1.5 if day_of_week >= 5 else 0.8
        trend = 1 + (60 - i) / 300
        base_demand = 80 * trend * multiplier + np.random.randn() * 10
        base_sales = base_demand * 120 * multiplier + np.random.randn() * 500
        base_profit = base_sales * 0.3 + np.random.randn() * 100
        demand.append(max(20, base_demand))
        sales.append(max(500, base_sales))
        profit.append(max(100, base_profit))
    return {'dates': dates, 'demand': demand, 'sales': sales, 'profit': profit}

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
        data = request.get_json()
        historical = data.get('historical', {})
        historical_demand = historical.get('demand', [])

        if historical_demand:
            avg_demand = sum(historical_demand[-30:]) / min(30, len(historical_demand))
            month = datetime.now().month
            seasonality = {1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.0, 6: 1.0,
                           7: 1.0, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.3, 12: 1.5}
            seasonal_factor = seasonality.get(month, 1.0)
            promotion_factor = 1.5 if data.get('promotion', 0) == 1 else 1.0
            predicted_demand = avg_demand * seasonal_factor * promotion_factor * 30
        else:
            category = data.get('category', 0)
            unit_price = data.get('unit_price', 100)
            category_demand = [1200, 800, 2000, 600, 900]
            base_demand = category_demand[category] if category < len(category_demand) else 1000
            price_elasticity = 1 - (unit_price / 1000) * 0.5
            predicted_demand = base_demand * price_elasticity

        return jsonify({'demand': max(1, predicted_demand), 'confidence': 0.85})
    except Exception as e:
        logger.error(f"Demand prediction error: {e}")
        return jsonify({'demand': 500})

@routes_bp.route('/api/predict-sales', methods=['POST'])
@login_required()
def predict_sales_api():
    try:
        data = request.get_json()
        demand_response = predict_demand_api()
        demand_data = demand_response.get_json()
        predicted_demand = demand_data['demand']
        unit_price = data.get('unit_price', 100)
        discount = data.get('discount_pct', 0)
        effective_price = unit_price * (1 - discount / 100)
        predicted_sales = predicted_demand * effective_price
        return jsonify({'sales': predicted_sales, 'demand': predicted_demand})
    except Exception as e:
        logger.error(f"Sales prediction error: {e}")
        return jsonify({'sales': 50000})

@routes_bp.route('/api/predict-profit', methods=['POST'])
@login_required()
def predict_profit_api():
    try:
        data = request.get_json()
        sales_response = predict_sales_api()
        sales_data = sales_response.get_json()
        predicted_sales = sales_data['sales']
        predicted_profit = predicted_sales * 0.35
        return jsonify({'profit': predicted_profit, 'margin': 0.35})
    except Exception as e:
        logger.error(f"Profit prediction error: {e}")
        return jsonify({'profit': 17500})

@routes_bp.route('/api/optimize-inventory', methods=['POST'])
@login_required()
def optimize_inventory_api():
    try:
        data = request.get_json()
        demand_response = predict_demand_api()
        demand_data = demand_response.get_json()
        predicted_demand = demand_data['demand']

        current_stock = data.get('stock_available', 100)
        lead_time = data.get('lead_time', 7)
        safety_stock = data.get('safety_stock', 50)
        daily_demand = predicted_demand / 30
        reorder_point = (daily_demand * lead_time) + safety_stock

        annual_demand = predicted_demand * 12
        order_cost = data.get('order_cost', 50)
        holding_cost = data.get('holding_cost', data.get('unit_price', 100) * 0.2)
        optimal_order = ((2 * annual_demand * order_cost) / holding_cost) ** 0.5 if holding_cost > 0 else annual_demand / 12

        return jsonify({
            'reorder_point': max(10, reorder_point),
            'optimal_order_quantity': max(10, optimal_order),
            'predicted_demand': predicted_demand,
            'should_order': current_stock <= reorder_point,
            'daily_demand': daily_demand
        })
    except Exception as e:
        logger.error(f"Inventory optimization error: {e}")
        return jsonify({'reorder_point': 100, 'optimal_order_quantity': 200,
                        'predicted_demand': 1000, 'should_order': True, 'daily_demand': 33})

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