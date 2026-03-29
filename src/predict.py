import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import Config
from src.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Check if models exist before loading
            if os.path.exists(Config.SALES_MODEL_PATH):
                self.models['sales'] = joblib.load(Config.SALES_MODEL_PATH)
            if os.path.exists(Config.DEMAND_MODEL_PATH):
                self.models['demand'] = joblib.load(Config.DEMAND_MODEL_PATH)
            if os.path.exists(Config.PROFIT_MODEL_PATH):
                self.models['profit'] = joblib.load(Config.PROFIT_MODEL_PATH)

            feature_names_path = os.path.join(str(Config.MODELS_DIR), 'feature_names.pkl')
            if os.path.exists(feature_names_path):
                self.feature_names = joblib.load(feature_names_path)
            
            feature_names_path = Config.MODELS_DIR / 'feature_names.pkl'
            if feature_names_path.exists():
                self.feature_names = joblib.load(feature_names_path)
                logger.info("Feature names loaded")
            
            if self.models:
                logger.info("Models loaded successfully")
                return True
            else:
                logger.warning("No models found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.info("Models will be retrained on next run")
            return False
    
    def predict_sales(self, input_data):
        """Predict sales (revenue)"""
        if not self.models.get('sales'):
            logger.error("Sales model not loaded")
            return None
        
        # Prepare features
        features = self._prepare_features(input_data)
        
        if features is None:
            return None
        
        # Make prediction
        try:
            prediction = self.models['sales'].predict(features)
            return prediction[0] if len(prediction) == 1 else prediction
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def predict_demand(self, input_data):
        """Predict demand (quantity)"""
        if not self.models.get('demand'):
            logger.error("Demand model not loaded")
            return None
        
        features = self._prepare_features(input_data)
        
        if features is None:
            return None
        
        try:
            prediction = self.models['demand'].predict(features)
            return prediction[0] if len(prediction) == 1 else prediction
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def predict_profit(self, input_data):
        """Predict profit"""
        if not self.models.get('profit'):
            logger.error("Profit model not loaded")
            return None
        
        features = self._prepare_features(input_data)
        
        if features is None:
            return None
        
        try:
            prediction = self.models['profit'].predict(features)
            return prediction[0] if len(prediction) == 1 else prediction
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def predict_all(self, input_data):
        """Get all predictions"""
        return {
            'sales': self.predict_sales(input_data),
            'demand': self.predict_demand(input_data),
            'profit': self.predict_profit(input_data)
        }
    
    def _prepare_features(self, input_data):
        """Prepare input data for prediction"""
        try:
            # Convert to DataFrame if dict
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            
            # Ensure all required features exist
            if self.feature_names:
                for feature in self.feature_names:
                    if feature not in input_data.columns:
                        input_data[feature] = 0
                
                # Select only required features
                features = input_data[self.feature_names]
            else:
                # Use default features that exist
                available_features = [f for f in Config.FEATURES if f in input_data.columns]
                if not available_features:
                    logger.error("No valid features found in input data")
                    return None
                features = input_data[available_features]
            
            # Handle missing values
            features = features.fillna(0)
            
            # Ensure numeric types
            features = features.astype(float)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def optimize_inventory(self, product_data):
        """
        Optimize inventory based on demand prediction
        
        Args:
            product_data: Dict with product details including current stock
        
        Returns:
            Dict with optimization recommendations
        """
        demand = self.predict_demand(product_data)
        
        if demand is None or demand <= 0:
            return None
        
        current_stock = product_data.get('stock_available', 0)
        lead_time_days = product_data.get('lead_time', 7)
        safety_stock = product_data.get('safety_stock', 50)
        
        # Calculate reorder point
        daily_demand = demand / 30  # Assuming monthly demand
        reorder_point = (daily_demand * lead_time_days) + safety_stock
        
        # Calculate optimal order quantity (EOQ style)
        order_cost = product_data.get('order_cost', 50)
        holding_cost = product_data.get('holding_cost', product_data.get('unit_price', 100) * 0.2)
        
        eoq = np.sqrt((2 * demand * order_cost) / holding_cost) if holding_cost > 0 else 0
        
        recommendation = {
            'predicted_demand': float(demand),
            'current_stock': float(current_stock),
            'reorder_point': float(reorder_point),
            'optimal_order_quantity': float(eoq),
            'should_order': bool(current_stock <= reorder_point),
            'order_quantity': float(max(0, eoq - current_stock)) if current_stock <= reorder_point else 0
        }
        
        return recommendation
    
    def optimize_profit(self, product_data):
        """
        Optimize profit by testing different scenarios
        
        Args:
            product_data: Dict with base product details
        
        Returns:
            Dict with optimization recommendations
        """
        base_profit = self.predict_profit(product_data)
        
        if base_profit is None:
            return None
        
        scenarios = []
        
        # Test different discount scenarios
        for discount in [0, 5, 10, 15, 20]:
            scenario = product_data.copy()
            scenario['discount_pct'] = discount
            
            # Adjust effective price
            if 'unit_price' in scenario:
                scenario['effective_price'] = scenario['unit_price'] * (1 - discount/100)
            
            profit = self.predict_profit(scenario)
            
            if profit is not None:
                scenarios.append({
                    'discount_pct': discount,
                    'predicted_profit': float(profit),
                    'profit_change': float(profit - base_profit)
                })
        
        if not scenarios:
            return None
        
        # Find best scenario
        best_scenario = max(scenarios, key=lambda x: x['predicted_profit'])
        
        return {
            'base_profit': float(base_profit),
            'optimal_discount': best_scenario['discount_pct'],
            'optimal_profit': best_scenario['predicted_profit'],
            'profit_improvement': best_scenario['profit_change'],
            'scenarios': scenarios
        }


def get_employee_performance(df):
    """Standalone wrapper used by export and reporting."""
    if 'employee_id' not in df.columns:
        return []
    grouped = df.groupby('employee_id').agg(
        total_sales=('revenue', 'sum'),
        total_transactions=('revenue', 'count'),
        total_profit=('profit', 'sum')
    ).reset_index()
    grouped['avg_transaction'] = grouped['total_sales'] / grouped['total_transactions'].clip(lower=1)
    mx = grouped['total_sales'].max()
    grouped['performance_rating'] = (grouped['total_sales'] / mx * 10).round(1) if mx and mx > 0 else 0.0
    return grouped.to_dict('records')


def predict_demand(input_data):
    """Standalone wrapper used by export and reporting."""
    p = Predictor()
    return p.predict_demand(input_data)


if __name__ == "__main__":
    # Test prediction
    predictor = Predictor()
    
    # Sample input
    sample_input = {
        'category': 0,  # Encoded category
        'unit_price': 500,
        'discount_pct': 10,
        'promotion': 1,
        'month': 12,
        'day': 25,
        'weekday': 0,
        'is_weekend': 0,
        'festival': 1,
        'lag_1_day_sales': 100,
        'lag_7_day_sales': 700,
        'rolling_7d_avg_sales': 100
    }
    
    predictions = predictor.predict_all(sample_input)
    print("Predictions:")
    for key, value in predictions.items():
        if value is not None:
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: Not available")