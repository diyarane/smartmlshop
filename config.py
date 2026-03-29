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
    
    TARGET_SALES = 'revenue'
    TARGET_DEMAND = 'quantity'
    TARGET_PROFIT = 'profit'
    
    @staticmethod
    def ensure_directories():
        Config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)