# app/app.py (updated version)
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
from pathlib import Path

from config import Config

# Import blueprints - using direct imports since we're in the app directory
from app.auth import auth_bp
from app.routes import routes_bp
from app.models import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Ensure instance folder exists
    instance_path = os.path.join(parent_dir, 'instance')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
    
    # Configuration
    app.config['SECRET_KEY'] = Config.SECRET_KEY
    app.config['DEBUG'] = Config.DEBUG
    # Use absolute path for database
    db_path = os.path.join(instance_path, 'shop.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(routes_bp, url_prefix='/')
    
    # Create tables and demo data
    with app.app_context():
        try:
            # Create all tables
            db.create_all()
            logger.info("Database tables created successfully")
            
            # Import and run demo data creation
            from app.demo_data import create_demo_data
            create_demo_data()
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
    
    # Ensure directories exist
    Config.ensure_directories()

    if not (
        Path(Config.SALES_MODEL_PATH).exists()
        and Path(Config.DEMAND_MODEL_PATH).exists()
        and Path(Config.PROFIT_MODEL_PATH).exists()
    ):
        logger.warning(
            "Models not found. Run: python -m scripts.train to train them."
        )

    return app

if __name__ == '__main__':
    app = create_app()
    logger.info(f"Starting server on http://localhost:{Config.PORT}")
    app.run(host='0.0.0.0', port=Config.PORT, debug=True)