import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        
    def load_raw_data(self, file_path=None):
        """Load raw data from CSV"""
        if file_path is None:
            file_path = Config.RAW_DATA_PATH
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            # Create sample data if file doesn't exist
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for testing"""
        logger.info("Creating sample data...")
        
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        
        categories = ['Electronics', 'Clothing', 'Groceries', 'Home', 'Sports']
        customers = ['Regular', 'Premium', 'VIP']
        payment_methods = ['Cash', 'Card', 'UPI']
        employees = [f'EMP_{i}' for i in range(1, 11)]
        products = [f'PROD_{i}' for i in range(1, 101)]
        
        data = []
        
        for date in dates:
            num_transactions = np.random.randint(50, 200)
            for _ in range(num_transactions):
                category = np.random.choice(categories)
                product = np.random.choice(products)
                quantity = np.random.randint(1, 10)
                unit_price = np.random.choice([100, 250, 500, 1000, 2000])
                
                row = {
                    'date': date,
                    'product': product,
                    'category': category,
                    'quantity': quantity,
                    'unit_price': unit_price,
                    'employee_id': np.random.choice(employees),
                    'customer_type': np.random.choice(customers),
                    'payment_method': np.random.choice(payment_methods),
                    'discount_pct': np.random.choice([0, 5, 10, 15, 20]),
                    'promotion': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'stock_available': np.random.randint(50, 500),
                    'festival': 1 if date.month in [10, 11, 12] and date.day in [25, 26, 31] else 0
                }
                
                # Calculate derived fields
                effective_price = row['unit_price'] * (1 - row['discount_pct']/100)
                row['effective_price'] = effective_price
                row['revenue'] = quantity * effective_price
                row['cost'] = quantity * row['unit_price'] * 0.6
                row['profit'] = row['revenue'] - row['cost']
                
                data.append(row)
        
        df = pd.DataFrame(data)
        logger.info(f"Created sample data with shape: {df.shape}")
        
        # Save sample data
        os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
        df.to_csv(Config.RAW_DATA_PATH, index=False)
        
        return df
    
    # ... rest of the methods remain the same ...
    
    
    def create_time_features(self, df):
        """Create time-based features"""
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        return df
    
    def create_lag_features(self, df, group_col='product', date_col='date', target_col='quantity'):
        """Create lag features for time series"""
        df = df.sort_values([group_col, date_col])
        
        # Create daily aggregates
        daily_sales = df.groupby([group_col, date_col])[target_col].sum().reset_index()
        
        # Create lag features
        for lag in [1, 7]:
            daily_sales[f'lag_{lag}_day_sales'] = daily_sales.groupby(group_col)[target_col].shift(lag)
        
        # Rolling averages
        daily_sales['rolling_7d_avg_sales'] = daily_sales.groupby(group_col)[target_col].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        # Merge back
        df = df.merge(daily_sales, on=[group_col, date_col], suffixes=('', '_daily'))
        
        return df
    
    def encode_categorical(self, df, categorical_columns):
        """Encode categorical variables"""
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        return df
    
    def prepare_features(self, df, include_target=True):
        """Prepare features for ML models"""
        # Remove identifiers that could cause leakage
        df_processed = df.copy()
        
        # Remove identifier columns
        cols_to_drop = ['customer_id', 'employee_id', 'product'] if 'customer_id' in df_processed.columns else []
        if cols_to_drop:
            df_processed = df_processed.drop(columns=cols_to_drop, errors='ignore')
        
        # Create time features if not present
        if 'date' in df_processed.columns:
            df_processed = self.create_time_features(df_processed)
        
        # Create lag features if not present and enough data
        if 'quantity' in df_processed.columns and df_processed.shape[0] > 100:
            df_processed = self.create_lag_features(df_processed)
        
        # Encode categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            df_processed = self.encode_categorical(df_processed, categorical_cols)
        
        # Fill NaN values
        df_processed = df_processed.fillna(0)
        
        # Handle infinite values
        df_processed = df_processed.replace([np.inf, -np.inf], 0)
        
        # Select features
        available_features = [f for f in Config.FEATURES if f in df_processed.columns]
        
        if include_target:
            # For training: separate features and targets
            X = df_processed[available_features]
            y_sales = df_processed[Config.TARGET_SALES] if Config.TARGET_SALES in df_processed.columns else None
            y_demand = df_processed[Config.TARGET_DEMAND] if Config.TARGET_DEMAND in df_processed.columns else None
            y_profit = df_processed[Config.TARGET_PROFIT] if Config.TARGET_PROFIT in df_processed.columns else None
            
            return X, y_sales, y_demand, y_profit
        else:
            # For prediction: only features
            return df_processed[available_features]
    
    def save_processed_data(self, df):
        """Save processed data to CSV"""
        df.to_csv(Config.PROCESSED_DATA_PATH, index=False)
        logger.info(f"Saved processed data to {Config.PROCESSED_DATA_PATH}")
    
    def load_processed_data(self):
        """Load processed data"""
        try:
            df = pd.read_csv(Config.PROCESSED_DATA_PATH)
            logger.info(f"Loaded processed data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.warning("Processed data not found")
            return None

if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor()
    df = preprocessor.load_raw_data()
    X, y_sales, y_demand, y_profit = preprocessor.prepare_features(df)
    print(f"Features shape: {X.shape}")
    print(f"Features: {X.columns.tolist()}")