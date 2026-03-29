import sys
import pandas as pd
import numpy as np
# Add parent directory to path
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

# Sales / profit models use dedicated feature sets (not raw demand×price chaining)
SALES_FEATURE_COLS = ['unit_price', 'category', 'demand_pred', 'trend']
PROFIT_FEATURE_COLS = [
    'unit_price', 'category', 'demand_pred', 'trend',
    'sales_pred', 'cost', 'margin_hint',
]


class ModelTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {
            'sales': None,
            'demand': None,
            'profit': None
        }

    def train_sales_model(self, X_train, y_train, X_test, y_test):
        """Train sales prediction model (revenue) on [price, category, demand_pred, trend]."""
        logger.info("Training sales prediction model...")
        
        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            max_samples=0.85,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Sales Model - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R2: {r2:.4f}")
        
        Config.SALES_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, Config.SALES_MODEL_PATH, compress=3)
        logger.info(f"Sales model saved to {Config.SALES_MODEL_PATH}")
        
        return model, {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def train_demand_model(self, X_train, y_train, X_test, y_test):
        """Train demand model on daily store-level features (time-based split)."""
        logger.info("Training demand prediction model (daily series)...")

        model = RandomForestRegressor(
            n_estimators=140,
            max_depth=14,
            min_samples_leaf=3,
            min_samples_split=4,
            max_features='sqrt',
            max_samples=0.75,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Demand Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

        Config.DEMAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, Config.DEMAND_MODEL_PATH, compress=3)
        joblib.dump(list(Config.DEMAND_FEATURES), Config.DEMAND_FEATURE_NAMES_PATH)
        logger.info(f"Demand model saved to {Config.DEMAND_MODEL_PATH}")

        return model, {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def train_profit_model(self, X_train, y_train, X_test, y_test):
        """Train profit model on sales_pred, cost, margin context (not demand×constant)."""
        logger.info("Training profit prediction model...")
        
        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            max_samples=0.85,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Profit Model - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R2: {r2:.4f}")
        
        Config.PROFIT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, Config.PROFIT_MODEL_PATH, compress=3)
        logger.info(f"Profit model saved to {Config.PROFIT_MODEL_PATH}")
        
        return model, {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def train_all_models(self):
        """Train demand, then sales (with demand_pred + trend), then profit (sales_pred − cost structure)."""
        df = self.preprocessor.load_raw_data()
        
        if df is None or df.empty:
            logger.error("No data available for training")
            return None

        packed = self.preprocessor.prepare_features(df, include_target=True, return_processed_frame=True)
        if packed is None:
            logger.error("prepare_features failed")
            return None
        _, y_sales, y_demand, y_profit, dfp = packed

        if y_sales is None or y_profit is None:
            logger.error("Target variables not found in data")
            return None

        y_sales = y_sales.astype(float)
        y_profit = y_profit.astype(float).fillna(0.0)

        dem = self.preprocessor.prepare_demand_training_data(df, return_dates=True)
        if dem[0] is None:
            logger.error("Insufficient rows for demand time-series training")
            return None
        X_demand, y_demand_daily, demand_dates = dem

        split_idx = int(len(X_demand) * (1 - Config.TEST_SIZE))
        split_idx = max(1, min(split_idx, len(X_demand) - 1))
        Xd_train = X_demand.iloc[:split_idx]
        Xd_test = X_demand.iloc[split_idx:]
        yd_train = y_demand_daily.iloc[:split_idx]
        yd_test = y_demand_daily.iloc[split_idx:]

        demand_model, demand_metrics = self.train_demand_model(
            Xd_train, yd_train, Xd_test, yd_test
        )

        d_pred_all = demand_model.predict(X_demand)
        dnorm = pd.to_datetime(demand_dates).dt.normalize()
        daily_map = dict(zip(dnorm, d_pred_all))

        tx_dates = pd.to_datetime(dfp['date']).dt.normalize()
        demand_pred_col = tx_dates.map(daily_map)
        demand_pred_col = demand_pred_col.fillna(dfp[Config.TARGET_DEMAND].astype(float))

        ordinals = tx_dates.map(pd.Timestamp.toordinal)
        tmin = int(ordinals.min())
        tmax = int(ordinals.max())
        trend_col = (ordinals - tmin) / max(1, tmax - tmin)

        rng = np.random.default_rng(Config.RANDOM_STATE)
        noise = 1.0 + rng.normal(0, 0.028, size=len(demand_pred_col))
        demand_sales_feat = demand_pred_col.values.astype(float) * noise

        X_sales = pd.DataFrame({
            'unit_price': dfp['unit_price'].astype(float).values,
            'category': dfp['category'].astype(float).values,
            'demand_pred': demand_sales_feat,
            'trend': trend_col.values.astype(float),
        })

        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_sales, y_sales, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        sales_model, sales_metrics = self.train_sales_model(
            X_train_s, y_train_s, X_test_s, y_test_s
        )

        sales_hat = sales_model.predict(X_sales)

        if 'cost' in dfp.columns:
            cost_col = dfp['cost'].astype(float).values
        else:
            cost_col = (
                dfp['unit_price'].astype(float) * dfp[Config.TARGET_DEMAND].astype(float) * 0.6
            ).values

        revenue = dfp[Config.TARGET_SALES].astype(float)
        margin_col = (y_profit / revenue.replace(0, np.nan)).fillna(0.35).clip(0.08, 0.72)

        X_profit = pd.DataFrame({
            'unit_price': dfp['unit_price'].astype(float).values,
            'category': dfp['category'].astype(float).values,
            'demand_pred': demand_pred_col.values.astype(float),
            'trend': trend_col.values.astype(float),
            'sales_pred': sales_hat,
            'cost': cost_col,
            'margin_hint': margin_col.values.astype(float),
        })

        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
            X_profit, y_profit, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        profit_model, profit_metrics = self.train_profit_model(
            X_train_p, y_train_p, X_test_p, y_test_p
        )

        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        s_hat_all = sales_model.predict(X_sales)
        sales_scale = float(y_sales.mean() / max(1e-6, np.mean(s_hat_all)))

        joblib.dump(SALES_FEATURE_COLS, Config.SALES_FEATURE_NAMES_PATH)
        joblib.dump(PROFIT_FEATURE_COLS, Config.PROFIT_FEATURE_NAMES_PATH)
        joblib.dump({
            'trend_ord_min': tmin,
            'trend_ord_max': tmax,
            'sales_scale': sales_scale,
        }, Config.FORECAST_TRAIN_META_PATH)

        X_legacy, _, _, _ = self.preprocessor.prepare_features(df, include_target=True)
        joblib.dump(X_legacy.columns.tolist(), Config.MODELS_DIR / 'feature_names.pkl')

        self.models = {
            'sales': sales_model,
            'demand': demand_model,
            'profit': profit_model
        }
        
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
    if metrics:
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name}:")
            for metric, value in model_metrics.items():
                print(f"  {metric}: {value:.4f}")
    else:
        print("Training failed.")
