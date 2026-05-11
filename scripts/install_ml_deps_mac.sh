#!/usr/bin/env bash
# XGBoost (and many tree libraries) need OpenMP on Apple Silicon / Intel Macs.
# Run from repo root: bash scripts/install_ml_deps_mac.sh

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is for macOS only."
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required: https://brew.sh"
  exit 1
fi

echo "Installing OpenMP (libomp) for XGBoost / LightGBM..."
brew install libomp

echo "Reinstalling Python wheels so they link against system OpenMP..."
pip uninstall -y xgboost lightgbm 2>/dev/null || true
pip install xgboost lightgbm

echo "Done. Verify with: python -c \"from xgboost import XGBRegressor; from lightgbm import LGBMRegressor; print('ok')\""
