import os
from pathlib import Path

class Config:
    # Base directories
    BASE_DIR = Path(os.path.abspath(__file__)).parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = BASE_DIR / 'models'
    
    # File paths
    RAW_DATA_PATH = RAW_DATA_DIR / 'shop_data.csv'
    PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / 'cleaned_data.csv'
    
    # Model paths
    SALES_MODEL_PATH = MODELS_DIR / 'sales_model.pkl'
    DEMAND_MODEL_PATH = MODELS_DIR / 'demand_model.pkl'
    PROFIT_MODEL_PATH = MODELS_DIR / 'profit_model.pkl'
    DEMAND_FEATURE_NAMES_PATH = MODELS_DIR / 'demand_feature_names.pkl'
    SALES_FEATURE_NAMES_PATH = MODELS_DIR / 'sales_feature_names.pkl'
    PROFIT_FEATURE_NAMES_PATH = MODELS_DIR / 'profit_feature_names.pkl'
    FORECAST_TRAIN_META_PATH = MODELS_DIR / 'forecast_train_meta.pkl'
    
    # ML parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = True
    PORT = 5001

    # Model features
    FEATURES = [
        'category', 'unit_price', 'discount_pct', 'promotion',
        'month', 'day', 'weekday', 'is_weekend', 'festival',
        'lag_1_day_sales', 'lag_7_day_sales', 'rolling_7d_avg_sales'
    ]
    
    TARGET_SALES = 'sales'
    TARGET_DEMAND = 'demand'
    TARGET_PROFIT = 'profit'

    # Features used only by the demand model (daily store-level series + calendar)
    DEMAND_FEATURES = [
        'lag_1', 'lag_7', 'rolling_mean_7',
        'day_of_week', 'month', 'is_festival',
    ]
    
    @staticmethod
    def ensure_directories():
        Config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)