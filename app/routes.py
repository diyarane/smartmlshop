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
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Get latest date
    latest_date = df['date'].max()
    
    # Get today's data
    today_data = df[df['date'] == latest_date]
    
    # Calculate metrics
    metrics = {
        'total_revenue': today_data['revenue'].sum() if not today_data.empty else 0,
        'total_profit': today_data['profit'].sum() if not today_data.empty else 0,
        'total_transactions': len(today_data),
        'avg_order_value': today_data['revenue'].mean() if not today_data.empty else 0
    }
    
    # Get top products
    product_sales = df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(10)
    
    # Get category distribution
    category_sales = df.groupby('category')['revenue'].sum()
    
    # Get daily sales trend (last 30 days)
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
    
    # Get employee performance data
    df = preprocessor.load_raw_data()
    
    if df is not None and employee_id:
        employee_data = df[df['employee_id'] == employee_id]
        
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
    
    # Pass current date to template
    now = datetime.now
    
    return render_template('employee_dashboard.html', 
                         employee_name=session['user']['name'],
                         employee_id=employee_id,
                         metrics=employee_metrics,
                         now=now)

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

@routes_bp.route('/api/optimize-inventory', methods=['POST'])
@login_required()
def optimize_inventory():
    """API endpoint for inventory optimization"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        optimization = predictor.optimize_inventory(data)
        
        if optimization is None:
            return jsonify({'error': 'Optimization failed'}), 500
        
        return jsonify(optimization)
    
    except Exception as e:
        logger.error(f"Inventory optimization error: {e}")
        return jsonify({'error': str(e)}), 500

@routes_bp.route('/api/optimize-profit', methods=['POST'])
@login_required()
def optimize_profit():
    """API endpoint for profit optimization"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        optimization = predictor.optimize_profit(data)
        
        if optimization is None:
            return jsonify({'error': 'Optimization failed'}), 500
        
        return jsonify(optimization)
    
    except Exception as e:
        logger.error(f"Profit optimization error: {e}")
        return jsonify({'error': str(e)}), 500

@routes_bp.route('/api/sales-forecast', methods=['GET'])
@login_required()
def sales_forecast():
    """Get sales forecast for next days"""
    try:
        days = request.args.get('days', default=7, type=int)
        
        # Get historical data
        df = preprocessor.load_raw_data()
        
        if df is None or df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        # Simple forecast using last 30 days average
        df['date'] = pd.to_datetime(df['date'])
        last_30_days = df[df['date'] >= (df['date'].max() - timedelta(days=30))]
        avg_daily_sales = last_30_days.groupby(last_30_days['date'].dt.weekday)['revenue'].mean()
        
        # Generate forecast
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
        
        # Load historical data
        df = preprocessor.load_raw_data()
        
        if df is None or df.empty:
            # Return sample historical data
            return jsonify(generate_sample_historical_data())
        
        # Filter for the specific product
        if product_name and product_name != 'Custom Product':
            product_df = df[df['product'] == product_name]
        elif category is not None:
            # Filter by category if product not found
            product_df = df[df['category'] == category]
        else:
            product_df = df.head(100)  # Take sample
        
        # Ensure date is datetime
        product_df['date'] = pd.to_datetime(product_df['date'])
        
        # Sort by date
        product_df = product_df.sort_values('date')
        
        # Group by date
        daily_data = product_df.groupby(product_df['date'].dt.date).agg({
            'quantity': 'sum',
            'revenue': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        # Get last 60 days
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
    """Generate sample historical data for demo"""
    dates = []
    demand = []
    sales = []
    profit = []
    
    # Generate last 60 days of data with realistic patterns
    for i in range(60, 0, -1):
        date = datetime.now() - timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
        
        # Add weekly patterns
        day_of_week = date.weekday()
        is_weekend = day_of_week >= 5
        multiplier = 1.5 if is_weekend else 0.8
        
        # Add some random variation and trend
        trend = 1 + (60 - i) / 300  # Slight upward trend
        base_demand = 80 * trend * multiplier + np.random.randn() * 10
        base_sales = base_demand * 120 * multiplier + np.random.randn() * 500
        base_profit = base_sales * 0.3 + np.random.randn() * 100
        
        demand.append(max(20, base_demand))
        sales.append(max(500, base_sales))
        profit.append(max(100, base_profit))
    
    return {
        'dates': dates,
        'demand': demand,
        'sales': sales,
        'profit': profit
    }