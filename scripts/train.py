# scripts/train.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_model import ModelTrainer

if __name__ == '__main__':
    trainer = ModelTrainer()
    metrics = trainer.train_all_models()
    # Demand, sales, and profit are trained as separate targets/feature sets (see src/train_model.py).
    print('Training complete:', metrics)
