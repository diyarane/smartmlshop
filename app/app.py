import sys
import os

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from flask import Flask
from flask_cors import CORS
import logging
import warnings

# Suppress sklearn warnings temporarily
warnings.filterwarnings("ignore", category=UserWarning)

# Now import from parent directory
from config import Config

# Import blueprints - using direct imports since we're in the app directory
from app.auth import auth_bp
from app.routes import routes_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = Config.SECRET_KEY
    app.config['DEBUG'] = Config.DEBUG
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(routes_bp, url_prefix='/')
    
    # Ensure directories exist
    Config.ensure_directories()
    
    # Create sample data if needed
    from src.preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    raw_data = preprocessor.load_raw_data()
    
    if raw_data is None or raw_data.empty:
        logger.info("Creating sample data...")
        raw_data = preprocessor.create_sample_data()
    
    # Train models if not exist OR if they're incompatible
    from src.train_model import ModelTrainer
    trainer = ModelTrainer()
    
    # Check if models exist and are compatible
    models_exist = (os.path.exists(Config.SALES_MODEL_PATH) and 
                    os.path.exists(Config.DEMAND_MODEL_PATH) and 
                    os.path.exists(Config.PROFIT_MODEL_PATH))
    
    # If models exist, try to load them to check compatibility
    if models_exist:
        try:
            # Try to load models to check compatibility
            import joblib
            test_model = joblib.load(Config.SALES_MODEL_PATH)
            logger.info("Existing models are compatible")
        except Exception as e:
            logger.warning(f"Models incompatible or corrupted: {e}")
            logger.info("Retraining models...")
            models_exist = False
    
    if not models_exist:
        logger.info("Training new models...")
        trainer.train_all_models()
    
    return app

if __name__ == '__main__':
    app = create_app()
    logger.info(f"Starting server on http://localhost:{Config.PORT}")
    app.run(host='0.0.0.0', port=Config.PORT, debug=True)