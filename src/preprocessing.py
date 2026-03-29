import sys
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
            df.columns = df.columns.str.strip().str.lower()
            assert 'demand' in df.columns
            assert 'sales' in df.columns
            assert 'profit' in df.columns
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['day'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['weekday'] = df['date'].dt.weekday
            if 'price' in df.columns and 'unit_price' not in df.columns:
                df['unit_price'] = pd.to_numeric(df['price'], errors='coerce')
            for col, default in [('discount_pct', 0), ('promotion', 0), ('festival', 0)]:
                if col not in df.columns:
                    df[col] = default
            if 'revenue' not in df.columns:
                df['revenue'] = pd.to_numeric(df['sales'], errors='coerce')
            if 'quantity' not in df.columns:
                df['quantity'] = pd.to_numeric(df['demand'], errors='coerce')
            logger.info(f"Loaded data with shape: {df.shape}")
            return self.ensure_demand_pipeline_columns(df)
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
                row['sales'] = row['revenue']
                row['cost'] = quantity * row['unit_price'] * 0.6
                row['profit'] = row['revenue'] - row['cost']
                
                data.append(row)
        
        df = pd.DataFrame(data)
        logger.info(f"Created sample data with shape: {df.shape}")
        
        # Save sample data
        os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
        df.to_csv(Config.RAW_DATA_PATH, index=False)
        
        return self.ensure_demand_pipeline_columns(df)

    @staticmethod
    def _parse_product_id_token(product_val):
        m = re.match(r'^PROD_(\d+)$', str(product_val).strip(), re.I)
        return int(m.group(1)) if m else np.nan

    def ensure_demand_pipeline_columns(self, df):
        """Ensure date, demand (from quantity if needed), and product_id (from PROD_n) exist."""
        if df is None or df.empty:
            return df
        out = df.copy()
        if 'date' not in out.columns:
            return out
        out['date'] = pd.to_datetime(out['date'], errors='coerce')
        if 'demand' not in out.columns:
            if 'quantity' in out.columns:
                out['demand'] = pd.to_numeric(out['quantity'], errors='coerce').fillna(0)
            else:
                out['demand'] = 0.0
        parsed = None
        if 'product' in out.columns:
            parsed = out['product'].map(self._parse_product_id_token)
        if 'product_id' not in out.columns:
            out['product_id'] = parsed if parsed is not None else np.nan
        else:
            out['product_id'] = pd.to_numeric(out['product_id'], errors='coerce')
            if parsed is not None:
                out['product_id'] = out['product_id'].fillna(parsed)
        return out

    def slice_product_series(self, df, product_id=None, product_name=None):
        """Rows for one product: match numeric product_id and/or catalog product name."""
        df = self.ensure_demand_pipeline_columns(df)
        sub = pd.DataFrame()
        if product_id is not None:
            try:
                pid = int(product_id)
            except (TypeError, ValueError):
                pid = None
            if pid is not None:
                ids = pd.to_numeric(df['product_id'], errors='coerce')
                sub = df[ids == float(pid)].copy()
        if sub.empty and product_name:
            sub = df[df['product'] == product_name].copy()
        if sub.empty:
            return sub
        return sub.sort_values('date')

    def apply_synthetic_transaction_demand(self, product_data):
        """If demand is missing, all zero, or constant on transaction rows, add realistic variation."""
        if product_data is None or product_data.empty:
            return product_data
        out = product_data.copy()
        d = pd.to_numeric(out['demand'], errors='coerce').fillna(0)
        if len(d) == 0:
            return out
        if d.nunique() <= 1 or float(d.sum()) <= 0:
            n = len(out)
            rng = np.random.default_rng(42)
            base = int(rng.integers(20, 100))
            noise = rng.normal(0, 10, size=n)
            trend = np.linspace(0, 10, n)
            out['demand'] = np.clip(base + noise + trend, 1, None)
        return out

    def aggregate_product_daily(self, sub):
        """Daily sums of demand (and revenue/profit when present)."""
        if sub is None or sub.empty:
            return pd.DataFrame(columns=['date', 'demand', 'revenue', 'profit'])
        s = sub.copy()
        s['_d'] = pd.to_datetime(s['date'], errors='coerce').dt.normalize()
        s = s.dropna(subset=['_d'])
        if s.empty:
            return pd.DataFrame(columns=['date', 'demand', 'revenue', 'profit'])
        agg_dict = {'demand': 'sum'}
        if 'revenue' in s.columns:
            agg_dict['revenue'] = 'sum'
        if 'profit' in s.columns:
            agg_dict['profit'] = 'sum'
        g = s.groupby('_d').agg(agg_dict).reset_index()
        g.rename(columns={'_d': 'date'}, inplace=True)
        g['date'] = pd.to_datetime(g['date']).dt.date
        return g.sort_values('date').reset_index(drop=True)

    def apply_synthetic_daily_demand(self, daily_df, min_days=30):
        """If daily series empty or flat, build or replace with synthetic non-flat demand."""
        if daily_df is None or daily_df.empty:
            rng = np.random.default_rng(99)
            end = datetime.now().date()
            dates = [end - timedelta(days=i) for i in range(min_days - 1, -1, -1)]
            n = len(dates)
            base = int(rng.integers(25, 95))
            noise = rng.normal(0, 9, size=n)
            trend = np.linspace(0, 11, n)
            dem = np.clip(base + noise + trend, 1, None)
            return pd.DataFrame({
                'date': dates,
                'demand': dem,
                'revenue': dem * 120.0,
                'profit': dem * 35.0,
            })
        out = daily_df.copy()
        d = pd.to_numeric(out['demand'], errors='coerce').fillna(0)
        if d.nunique() <= 1 or float(d.max()) <= 0:
            n = len(out)
            rng = np.random.default_rng(42)
            base = int(rng.integers(20, 100))
            noise = rng.normal(0, 10, size=n)
            trend = np.linspace(0, 10, n)
            out['demand'] = np.clip(base + noise + trend, 1, None)
            if 'revenue' in out.columns:
                out['revenue'] = out['demand'] * 100.0
            if 'profit' in out.columns:
                out['profit'] = out['demand'] * 32.0
        return out
    
    def create_time_features(self, df):
        """Create time-based features"""
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        return df
    
    def create_lag_features(self, df, group_col='product', date_col='date', target_col=None):
        """Create lag features for time series"""
        if target_col is None:
            if 'quantity' in df.columns:
                target_col = 'quantity'
            elif 'demand' in df.columns:
                target_col = 'demand'
            else:
                return df
        if group_col not in df.columns:
            if 'product_id' in df.columns:
                group_col = 'product_id'
            elif 'product_name' in df.columns:
                group_col = 'product_name'
            else:
                return df
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
    
    def prepare_features(self, df, include_target=True, return_processed_frame=False):
        """Prepare features for ML models"""
        df_processed = df.copy()
        
        if 'date' in df_processed.columns:
            df_processed = self.create_time_features(df_processed)
        
        has_qty_or_demand = (
            'quantity' in df_processed.columns or 'demand' in df_processed.columns
        )
        if has_qty_or_demand and df_processed.shape[0] > 100:
            df_processed = self.create_lag_features(df_processed)

        id_cols = [
            'customer_id', 'employee_id', 'product', 'product_id', 'product_name',
        ]
        df_processed = df_processed.drop(
            columns=[c for c in id_cols if c in df_processed.columns],
            errors='ignore',
        )
        
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
            assert Config.TARGET_DEMAND in df_processed.columns
            assert Config.TARGET_SALES in df_processed.columns
            assert Config.TARGET_PROFIT in df_processed.columns
            X = df_processed[available_features]
            y_sales = df_processed[Config.TARGET_SALES]
            y_demand = df_processed[Config.TARGET_DEMAND]
            y_profit = df_processed[Config.TARGET_PROFIT]

            if return_processed_frame:
                return X, y_sales, y_demand, y_profit, df_processed
            return X, y_sales, y_demand, y_profit
        else:
            # For prediction: only features
            return df_processed[available_features]

    def prepare_demand_training_data(self, df, return_dates=False):
        """
        Daily store-level demand (sum of demand or quantity) with lag/rolling/calendar features.
        Rows with incomplete lags are dropped (shift + rolling), not back-filled.
        """
        if df is None or df.empty:
            return (None, None, None) if return_dates else (None, None)
        dem_col = 'quantity' if 'quantity' in df.columns else 'demand'
        if dem_col not in df.columns:
            return (None, None, None) if return_dates else (None, None)

        work = df.copy()
        work['date'] = pd.to_datetime(work['date'], errors='coerce')
        work = work.dropna(subset=['date'])
        if work.empty:
            return (None, None, None) if return_dates else (None, None)

        if 'festival' not in work.columns:
            work['festival'] = 0

        work['_day'] = work['date'].dt.normalize()
        daily = (
            work.groupby('_day', as_index=False)
            .agg(demand=(dem_col, 'sum'), is_festival=('festival', 'max'))
            .rename(columns={'_day': 'date'})
            .sort_values('date')
            .reset_index(drop=True)
        )

        daily['lag_1'] = daily['demand'].shift(1)
        daily['lag_7'] = daily['demand'].shift(7)
        daily['rolling_mean_7'] = daily['demand'].rolling(window=7).mean()
        daily['day_of_week'] = daily['date'].dt.weekday
        daily['month'] = daily['date'].dt.month
        daily['is_festival'] = daily['is_festival'].fillna(0).astype(float)

        daily = daily.dropna(subset=['lag_1', 'lag_7', 'rolling_mean_7']).reset_index(drop=True)
        if len(daily) < 30:
            return (None, None, None) if return_dates else (None, None)

        feature_cols = list(Config.DEMAND_FEATURES)
        X = daily[feature_cols].replace([np.inf, -np.inf], 0).astype(float)
        y = daily['demand'].astype(float)
        if return_dates:
            dts = daily['date'].copy().reset_index(drop=True)
            return X, y, dts
        return X, y

    def store_daily_history(self, df, n=90):
        """Last ``n`` days of store-wide total quantity for demand inference."""
        if df is None or df.empty:
            return {'dates': [], 'demand': []}
        w = df.copy()
        w['date'] = pd.to_datetime(w['date'], errors='coerce')
        w = w.dropna(subset=['date'])
        if w.empty:
            return {'dates': [], 'demand': []}
        w['_d'] = w['date'].dt.normalize()
        qcol = 'quantity' if 'quantity' in w.columns else 'demand'
        totals = w.groupby('_d')[qcol].sum().sort_index()
        tail_dates = totals.index.strftime('%Y-%m-%d').tolist()[-n:]
        tail_dem = [float(x) for x in totals.values.tolist()[-n:]]
        return {'dates': tail_dates, 'demand': tail_dem}

    def build_demand_prediction_row(self, input_data):
        """
        Build one feature row for the demand model from API payload / dict.
        Prefers historical_store (store-wide daily totals) to match training distribution.
        """
        if isinstance(input_data, dict):
            h_store = input_data.get('historical_store') or {}
            h_prod = input_data.get('historical') or {}
            d_prod = h_prod.get('demand') if isinstance(h_prod, dict) else None
            if isinstance(d_prod, list) and len(d_prod) >= 7:
                hist = h_prod
            else:
                hist = h_store or h_prod or {}
        else:
            hist = {}

        dates = hist.get('dates') or []
        demands = hist.get('demand') or []
        # Align and aggregate by date if duplicates
        if dates and demands and len(dates) == len(demands):
            s = (
                pd.DataFrame({'d': pd.to_datetime(dates, errors='coerce'), 'q': demands})
                .dropna(subset=['d'])
                .groupby('d', as_index=False)['q']
                .sum()
            )
            s = s.sort_values('d')
            qseries = s['q'].tolist()
        else:
            qseries = [float(x) for x in demands if x is not None and str(x) != '']

        if not qseries:
            base = float(input_data.get('stock_available') or 50) / 10.0
            qseries = [max(1.0, base)] * 8

        lag_1 = float(qseries[-1])
        lag_7 = float(qseries[-7]) if len(qseries) >= 7 else lag_1
        rolling_mean_7 = float(sum(qseries[-7:]) / min(7, len(qseries)))

        now = datetime.now()
        fest = float(input_data.get('festival', input_data.get('is_festival', 0)) or 0)

        row = {
            'lag_1': lag_1,
            'lag_7': lag_7,
            'rolling_mean_7': rolling_mean_7,
            'day_of_week': float(now.weekday()),
            'month': float(now.month),
            'is_festival': fest,
        }
        return pd.DataFrame([row])
    
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