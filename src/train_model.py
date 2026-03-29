import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import Config
from src.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {
            'sales': None,
            'demand': None,
            'profit': None
        }
        
    def train_sales_model(self, X_train, y_train, X_test, y_test):
        """Train sales prediction model"""
        logger.info("Training sales prediction model...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Sales Model - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R2: {r2:.4f}")
        
        # Save model with compression to reduce size
        Config.SALES_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, Config.SALES_MODEL_PATH, compress=3)
        logger.info(f"Sales model saved to {Config.SALES_MODEL_PATH}")
        
        return model, {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def train_demand_model(self, X_train, y_train, X_test, y_test):
        """Train demand prediction model"""
        logger.info("Training demand prediction model...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Demand Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        # Save model with compression
        Config.DEMAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, Config.DEMAND_MODEL_PATH, compress=3)
        logger.info(f"Demand model saved to {Config.DEMAND_MODEL_PATH}")
        
        return model, {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def train_profit_model(self, X_train, y_train, X_test, y_test):
        """Train profit optimization model"""
        logger.info("Training profit optimization model...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Profit Model - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R2: {r2:.4f}")
        
        # Save model with compression
        Config.PROFIT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, Config.PROFIT_MODEL_PATH, compress=3)
        logger.info(f"Profit model saved to {Config.PROFIT_MODEL_PATH}")
        
        return model, {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def train_all_models(self):
        """Train all three models"""
        # Load and prepare data
        df = self.preprocessor.load_raw_data()
        
        if df is None or df.empty:
            logger.error("No data available for training")
            return None
        
        X, y_sales, y_demand, y_profit = self.preprocessor.prepare_features(df, include_target=True)
        
        # Check if targets exist
        if y_sales is None or y_demand is None or y_profit is None:
            logger.error("Target variables not found in data")
            return None
        
        # Split data
        X_train, X_test, y_sales_train, y_sales_test = train_test_split(
            X, y_sales, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        _, _, y_demand_train, y_demand_test = train_test_split(
            X, y_demand, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        _, _, y_profit_train, y_profit_test = train_test_split(
            X, y_profit, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        
        # Train models
        sales_model, sales_metrics = self.train_sales_model(
            X_train, y_sales_train, X_test, y_sales_test
        )
        
        demand_model, demand_metrics = self.train_demand_model(
            X_train, y_demand_train, X_test, y_demand_test
        )
        
        profit_model, profit_metrics = self.train_profit_model(
            X_train, y_profit_train, X_test, y_profit_test
        )
        
        self.models = {
            'sales': sales_model,
            'demand': demand_model,
            'profit': profit_model
        }
        
        # Save feature names for prediction
        feature_names = X.columns.tolist()
        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(feature_names, Config.MODELS_DIR / 'feature_names.pkl')
        
        logger.info("All models trained successfully!")
        
        return {
            'sales_metrics': sales_metrics,
            'demand_metrics': demand_metrics,
            'profit_metrics': profit_metrics
        }
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            self.models['sales'] = joblib.load(Config.SALES_MODEL_PATH)
            self.models['demand'] = joblib.load(Config.DEMAND_MODEL_PATH)
            self.models['profit'] = joblib.load(Config.PROFIT_MODEL_PATH)
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

if __name__ == "__main__":
    trainer = ModelTrainer()
    metrics = trainer.train_all_models()
    print("\nTraining Results:")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        for metric, value in model_metrics.items():
            print(f"  {metric}: {value:.4f}")