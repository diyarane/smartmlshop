import sys
import os
import time
import logging
import warnings

import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except (ImportError, OSError) as _xgb_err:
    HAS_XGBOOST = False
    XGBRegressor = None  # type: ignore
    logger.warning(
        "XGBoost not available (%s). On macOS: brew install libomp; then "
        "pip uninstall -y xgboost && pip install xgboost (or run scripts/install_ml_deps_mac.sh).",
        _xgb_err,
    )

try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except (ImportError, OSError) as _lgb_err:
    HAS_LIGHTGBM = False
    LGBMRegressor = None  # type: ignore
    logger.warning(
        "LightGBM not available (%s). On macOS: brew install libomp && pip install lightgbm",
        _lgb_err,
    )

warnings.filterwarnings("ignore", category=UserWarning)

from config import Config
from src.preprocessing import DataPreprocessor

SALES_FEATURE_COLS = ["unit_price", "category", "demand_pred", "trend"]
PROFIT_FEATURE_COLS = [
    "unit_price",
    "category",
    "demand_pred",
    "trend",
    "sales_pred",
    "cost",
    "margin_hint",
]

RF_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

GBR_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.05, 0.1, 0.15],
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5, 10],
    "subsample": [0.8, 0.9, 1.0],
}


def _inv_log_pred(pred, target_log: bool):
    p = np.asarray(pred, dtype=float).ravel()
    if target_log:
        p = np.clip(p, -48.0, 48.0)
        return np.expm1(p)
    return p


def _fmt_value(v: float, use_dollar: bool) -> str:
    if use_dollar:
        return f"${v:,.2f}"
    return f"{v:,.2f}"


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def print_model_report(name: str, train_m: dict, test_m: dict, train_time: float, use_dollar: bool) -> None:
    print(f"\n=== {name} ===", flush=True)
    print("Training Metrics:", flush=True)
    print(f"  - R² Score: {train_m['r2']:.4f}", flush=True)
    print(f"  - MAE: {_fmt_value(train_m['mae'], use_dollar)}", flush=True)
    print(f"  - RMSE: {_fmt_value(train_m['rmse'], use_dollar)}", flush=True)
    print("Testing Metrics:", flush=True)
    print(f"  - R² Score: {test_m['r2']:.4f}", flush=True)
    print(f"  - MAE: {_fmt_value(test_m['mae'], use_dollar)}", flush=True)
    print(f"  - RMSE: {_fmt_value(test_m['rmse'], use_dollar)}", flush=True)
    print(f"  - Training Time: {train_time:.2f} seconds", flush=True)


def fit_eval_model(
    name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    use_dollar: bool,
    target_log: bool = False,
):
    y_train_orig = np.asarray(y_train, dtype=float).ravel()
    y_test_orig = np.asarray(y_test, dtype=float).ravel()
    y_fit = np.log1p(np.maximum(y_train_orig, 0.0)) if target_log else y_train_orig

    t0 = time.perf_counter()
    model.fit(X_train, y_fit)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    pred_tr = _inv_log_pred(model.predict(X_train), target_log)
    pred_te = _inv_log_pred(model.predict(X_test), target_log)
    train_m = compute_metrics(y_train_orig, pred_tr)
    test_m = compute_metrics(y_test_orig, pred_te)
    print_model_report(name, train_m, test_m, elapsed, use_dollar)
    return model, train_m, test_m, elapsed


def run_rf_grid_search(
    X_train,
    y_train,
    X_test,
    y_test,
    use_dollar: bool,
    target_log: bool = False,
):
    y_train_orig = np.asarray(y_train, dtype=float).ravel()
    y_test_orig = np.asarray(y_test, dtype=float).ravel()
    y_train_fit = np.log1p(np.maximum(y_train_orig, 0.0)) if target_log else y_train_orig

    base = RandomForestRegressor(random_state=Config.RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(
        base,
        RF_PARAM_GRID,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        refit=True,
    )
    t0 = time.perf_counter()
    grid.fit(X_train, y_train_fit)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    best = grid.best_estimator_
    train_m = compute_metrics(
        y_train_orig, _inv_log_pred(best.predict(X_train), target_log)
    )
    test_m = compute_metrics(
        y_test_orig, _inv_log_pred(best.predict(X_test), target_log)
    )

    print("\n=== RANDOM FOREST (GridSearchCV — tuned) ===", flush=True)
    print("Training Metrics:", flush=True)
    print(f"  - R² Score: {train_m['r2']:.4f}", flush=True)
    print(f"  - MAE: {_fmt_value(train_m['mae'], use_dollar)}", flush=True)
    print(f"  - RMSE: {_fmt_value(train_m['rmse'], use_dollar)}", flush=True)
    print("Testing Metrics:", flush=True)
    print(f"  - R² Score: {test_m['r2']:.4f}", flush=True)
    print(f"  - MAE: {_fmt_value(test_m['mae'], use_dollar)}", flush=True)
    print(f"  - RMSE: {_fmt_value(test_m['rmse'], use_dollar)}", flush=True)
    print(f"  - Training Time: {elapsed:.2f} seconds", flush=True)

    print("\nBEST PARAMETERS found:", flush=True)
    bp = grid.best_params_
    print(f"  - Best n_estimators: {bp['n_estimators']}", flush=True)
    print(f"  - Best max_depth: {bp['max_depth']}", flush=True)
    print(f"  - Best min_samples_split: {bp['min_samples_split']}", flush=True)
    print(f"  - Best min_samples_leaf: {bp['min_samples_leaf']}", flush=True)
    print(f"  - Best cross-validation score: {grid.best_score_:.4f}", flush=True)

    return best, train_m, test_m, elapsed, bp, grid.best_score_


def run_gbr_grid_search(
    X_train,
    y_train,
    X_test,
    y_test,
    use_dollar: bool,
    target_log: bool = False,
):
    y_train_orig = np.asarray(y_train, dtype=float).ravel()
    y_test_orig = np.asarray(y_test, dtype=float).ravel()
    y_train_fit = np.log1p(np.maximum(y_train_orig, 0.0)) if target_log else y_train_orig

    base = GradientBoostingRegressor(
        random_state=Config.RANDOM_STATE,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
    )
    grid = GridSearchCV(
        base,
        GBR_PARAM_GRID,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        refit=True,
    )
    t0 = time.perf_counter()
    grid.fit(X_train, y_train_fit)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    best = grid.best_estimator_
    train_m = compute_metrics(
        y_train_orig, _inv_log_pred(best.predict(X_train), target_log)
    )
    test_m = compute_metrics(
        y_test_orig, _inv_log_pred(best.predict(X_test), target_log)
    )

    print("\n=== GRADIENT BOOSTING (GridSearchCV — tuned) ===", flush=True)
    print("Training Metrics:", flush=True)
    print(f"  - R² Score: {train_m['r2']:.4f}", flush=True)
    print(f"  - MAE: {_fmt_value(train_m['mae'], use_dollar)}", flush=True)
    print(f"  - RMSE: {_fmt_value(train_m['rmse'], use_dollar)}", flush=True)
    print("Testing Metrics:", flush=True)
    print(f"  - R² Score: {test_m['r2']:.4f}", flush=True)
    print(f"  - MAE: {_fmt_value(test_m['mae'], use_dollar)}", flush=True)
    print(f"  - RMSE: {_fmt_value(test_m['rmse'], use_dollar)}", flush=True)
    print(f"  - Training Time: {elapsed:.2f} seconds", flush=True)

    print("\nBEST PARAMETERS found:", flush=True)
    bp = grid.best_params_
    print(f"  - Best n_estimators: {bp['n_estimators']}", flush=True)
    print(f"  - Best learning_rate: {bp['learning_rate']}", flush=True)
    print(f"  - Best max_depth: {bp['max_depth']}", flush=True)
    print(f"  - Best min_samples_split: {bp['min_samples_split']}", flush=True)
    print(f"  - Best subsample: {bp['subsample']}", flush=True)
    print(f"  - Best cross-validation score: {grid.best_score_:.4f}", flush=True)

    return best, train_m, test_m, elapsed, bp, grid.best_score_


def print_comparison_table(title: str, rows: list, best_label: str, best_r2: float) -> None:
    print("\n" + "=" * 40, flush=True)
    print(title, flush=True)
    print("=" * 40, flush=True)
    for line in rows:
        print(line, flush=True)
    print("\n" + "=" * 40, flush=True)
    print(f"BEST MODEL: {best_label} with R²={best_r2:.4f}", flush=True)
    print("=" * 40, flush=True)


def _summary_lines(idx: int, label: str, train_m: dict, test_m: dict, use_dollar: bool, extra: str = "") -> str:
    ma_tr = _fmt_value(train_m["mae"], use_dollar)
    rm_tr = _fmt_value(train_m["rmse"], use_dollar)
    ma_te = _fmt_value(test_m["mae"], use_dollar)
    rm_te = _fmt_value(test_m["rmse"], use_dollar)
    block = (
        f"{idx}. {label}\n"
        f"   Training: R²={train_m['r2']:.4f}, MAE={ma_tr}, RMSE={rm_tr}\n"
        f"   Testing:  R²={test_m['r2']:.4f}, MAE={ma_te}, RMSE={rm_te}"
    )
    if extra:
        block += f"\n   {extra}"
    return block


def train_and_compare_target(
    target_title: str,
    X_train,
    X_test,
    y_train,
    y_test,
    use_dollar: bool,
    target_log: bool = False,
) -> tuple:
    """
    Production: tuned GradientBoostingRegressor. Also compares RF, XGBoost, LightGBM, LR, SVR.
    """
    rows_summary: list[str] = []
    candidates: list[tuple[str, float]] = []
    line_no = 1

    gbr_default = GradientBoostingRegressor(
        random_state=Config.RANDOM_STATE,
        n_estimators=500,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=4,
        subsample=0.9,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
    )
    _, tr_m, te_m, _ = fit_eval_model(
        "GRADIENT BOOSTING (Default)",
        gbr_default,
        X_train,
        X_test,
        y_train,
        y_test,
        use_dollar,
        target_log=target_log,
    )
    rows_summary.append(
        _summary_lines(line_no, "GRADIENT BOOSTING (Default)", tr_m, te_m, use_dollar)
    )
    line_no += 1
    candidates.append(("GRADIENT BOOSTING (Default)", te_m["r2"]))

    tuned_gbr, tr_g, te_g, _, gbr_bp, _ = run_gbr_grid_search(
        X_train, y_train, X_test, y_test, use_dollar, target_log=target_log
    )
    gbr_params_str = (
        "Best Params: {"
        f"'n_estimators': {gbr_bp['n_estimators']}, "
        f"'learning_rate': {gbr_bp['learning_rate']}, "
        f"'max_depth': {gbr_bp['max_depth']}, "
        f"'min_samples_split': {gbr_bp['min_samples_split']}, "
        f"'subsample': {gbr_bp['subsample']}"
        "}"
    )
    rows_summary.append(
        _summary_lines(
            line_no,
            "GRADIENT BOOSTING (Tuned)",
            tr_g,
            te_g,
            use_dollar,
            extra=gbr_params_str,
        )
    )
    line_no += 1
    candidates.append(("GRADIENT BOOSTING (Tuned)", te_g["r2"]))

    rf_default = RandomForestRegressor(random_state=Config.RANDOM_STATE, n_jobs=-1)
    _, tr_rf, te_rf, _ = fit_eval_model(
        "RANDOM FOREST (Default)",
        rf_default,
        X_train,
        X_test,
        y_train,
        y_test,
        use_dollar,
        target_log=target_log,
    )
    rows_summary.append(_summary_lines(line_no, "RANDOM FOREST (Default)", tr_rf, te_rf, use_dollar))
    line_no += 1
    candidates.append(("RANDOM FOREST (Default)", te_rf["r2"]))

    tuned_rf, tr_t, te_t, _, best_params, _ = run_rf_grid_search(
        X_train, y_train, X_test, y_test, use_dollar, target_log=target_log
    )
    params_str = (
        "Best Params: {"
        f"'n_estimators': {best_params['n_estimators']}, "
        f"'max_depth': {best_params['max_depth']}, "
        f"'min_samples_split': {best_params['min_samples_split']}, "
        f"'min_samples_leaf': {best_params['min_samples_leaf']}"
        "}"
    )
    rows_summary.append(
        _summary_lines(line_no, "RANDOM FOREST (Tuned)", tr_t, te_t, use_dollar, extra=params_str)
    )
    line_no += 1
    candidates.append(("RANDOM FOREST (Tuned)", te_t["r2"]))

    if HAS_XGBOOST:
        xgb = XGBRegressor(
            random_state=Config.RANDOM_STATE,
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            verbosity=0,
        )
        _, tr_x, te_x, _ = fit_eval_model(
            "XGBOOST",
            xgb,
            X_train,
            X_test,
            y_train,
            y_test,
            use_dollar,
            target_log=target_log,
        )
        rows_summary.append(_summary_lines(line_no, "XGBOOST", tr_x, te_x, use_dollar))
        line_no += 1
        candidates.append(("XGBOOST", te_x["r2"]))
    else:
        print(
            "\n=== XGBOOST ===\n"
            "  (skipped — macOS: brew install libomp; pip uninstall -y xgboost && pip install xgboost)\n"
            "  Or: bash scripts/install_ml_deps_mac.sh",
            flush=True,
        )
        rows_summary.append(
            f"{line_no}. XGBOOST\n"
            "   (skipped — missing OpenMP or package; see scripts/install_ml_deps_mac.sh)\n"
            "   Training: R²=n/a, MAE=n/a, RMSE=n/a\n"
            "   Testing:  R²=n/a, MAE=n/a, RMSE=n/a"
        )
        line_no += 1

    if HAS_LIGHTGBM:
        lgbm = LGBMRegressor(
            random_state=Config.RANDOM_STATE,
            n_estimators=200,
            learning_rate=0.06,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            verbose=-1,
        )
        _, tr_lgb, te_lgb, _ = fit_eval_model(
            "LIGHTGBM",
            lgbm,
            X_train,
            X_test,
            y_train,
            y_test,
            use_dollar,
            target_log=target_log,
        )
        rows_summary.append(_summary_lines(line_no, "LIGHTGBM", tr_lgb, te_lgb, use_dollar))
        line_no += 1
        candidates.append(("LIGHTGBM", te_lgb["r2"]))
    else:
        print(
            "\n=== LIGHTGBM ===\n"
            "  (skipped — pip install lightgbm; on macOS ensure: brew install libomp)",
            flush=True,
        )
        rows_summary.append(
            f"{line_no}. LIGHTGBM\n"
            "   (skipped — install lightgbm or fix OpenMP on Mac)\n"
            "   Training: R²=n/a, MAE=n/a, RMSE=n/a\n"
            "   Testing:  R²=n/a, MAE=n/a, RMSE=n/a"
        )
        line_no += 1

    lr = LinearRegression()
    _, tr_l, te_l, _ = fit_eval_model(
        "LINEAR REGRESSION",
        lr,
        X_train,
        X_test,
        y_train,
        y_test,
        use_dollar,
        target_log=target_log,
    )
    rows_summary.append(_summary_lines(line_no, "LINEAR REGRESSION", tr_l, te_l, use_dollar))
    line_no += 1
    candidates.append(("LINEAR REGRESSION", te_l["r2"]))

    svr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
        ]
    )
    _, tr_s, te_s, _ = fit_eval_model(
        "SVR",
        svr_pipe,
        X_train,
        X_test,
        y_train,
        y_test,
        use_dollar,
        target_log=target_log,
    )
    rows_summary.append(_summary_lines(line_no, "SVR", tr_s, te_s, use_dollar))
    line_no += 1
    candidates.append(("SVR", te_s["r2"]))

    best_name, best_r2 = max(candidates, key=lambda x: x[1])
    print_comparison_table(
        f"MULTIPLE MODEL COMPARISON — {target_title}",
        rows_summary,
        best_name,
        best_r2,
    )

    return tuned_gbr, best_name, best_r2, tuned_rf


class ModelTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {"sales": None, "demand": None, "profit": None}

    def train_all_models(self):
        """Train demand → sales → profit; persist tuned Gradient Boosting + RF fallbacks."""
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

        print("\n########## DEMAND (daily store-level) ##########", flush=True)
        demand_gbr, _, _, demand_rf_fb = train_and_compare_target(
            "DEMAND",
            Xd_train,
            Xd_test,
            yd_train,
            yd_test,
            use_dollar=False,
            target_log=False,
        )

        Config.DEMAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(demand_gbr, Config.DEMAND_MODEL_PATH, compress=3)
        joblib.dump(demand_rf_fb, Config.DEMAND_MODEL_RF_FALLBACK_PATH, compress=3)
        joblib.dump(list(Config.DEMAND_FEATURES), Config.DEMAND_FEATURE_NAMES_PATH)
        logger.info("Demand model (tuned Gradient Boosting) saved to %s", Config.DEMAND_MODEL_PATH)

        d_pred_all = demand_gbr.predict(X_demand)
        dnorm = pd.to_datetime(demand_dates).dt.normalize()
        daily_map = dict(zip(dnorm, d_pred_all))

        tx_dates = pd.to_datetime(dfp["date"]).dt.normalize()
        demand_pred_col = tx_dates.map(daily_map)
        demand_pred_col = demand_pred_col.fillna(dfp[Config.TARGET_DEMAND].astype(float))

        ordinals = tx_dates.map(pd.Timestamp.toordinal)
        tmin = int(ordinals.min())
        tmax = int(ordinals.max())
        trend_col = (ordinals - tmin) / max(1, tmax - tmin)

        rng = np.random.default_rng(Config.RANDOM_STATE)
        noise = 1.0 + rng.normal(0, 0.028, size=len(demand_pred_col))
        demand_sales_feat = demand_pred_col.values.astype(float) * noise

        X_sales = pd.DataFrame(
            {
                "unit_price": dfp["unit_price"].astype(float).values,
                "category": dfp["category"].astype(float).values,
                "demand_pred": demand_sales_feat,
                "trend": trend_col.values.astype(float),
            }
        )

        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_sales, y_sales, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )

        print("\n########## SALES (revenue) ##########", flush=True)
        sales_gbr, _, _, sales_rf_fb = train_and_compare_target(
            "SALES",
            X_train_s,
            X_test_s,
            y_train_s,
            y_test_s,
            use_dollar=True,
            target_log=True,
        )

        Config.SALES_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(sales_gbr, Config.SALES_MODEL_PATH, compress=3)
        joblib.dump(sales_rf_fb, Config.SALES_MODEL_RF_FALLBACK_PATH, compress=3)
        logger.info("Sales model (tuned Gradient Boosting, log1p target) saved to %s", Config.SALES_MODEL_PATH)

        sales_hat = _inv_log_pred(sales_gbr.predict(X_sales), True)

        if "cost" in dfp.columns:
            cost_col = dfp["cost"].astype(float).values
        else:
            cost_col = (
                dfp["unit_price"].astype(float) * dfp[Config.TARGET_DEMAND].astype(float) * 0.6
            ).values

        revenue = dfp[Config.TARGET_SALES].astype(float)
        margin_col = (y_profit / revenue.replace(0, np.nan)).fillna(0.35).clip(0.08, 0.72)

        X_profit = pd.DataFrame(
            {
                "unit_price": dfp["unit_price"].astype(float).values,
                "category": dfp["category"].astype(float).values,
                "demand_pred": demand_pred_col.values.astype(float),
                "trend": trend_col.values.astype(float),
                "sales_pred": sales_hat,
                "cost": cost_col,
                "margin_hint": margin_col.values.astype(float),
            }
        )

        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
            X_profit, y_profit, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )

        print("\n########## PROFIT ##########", flush=True)
        profit_gbr, _, _, profit_rf_fb = train_and_compare_target(
            "PROFIT",
            X_train_p,
            X_test_p,
            y_train_p,
            y_test_p,
            use_dollar=True,
            target_log=True,
        )

        Config.PROFIT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(profit_gbr, Config.PROFIT_MODEL_PATH, compress=3)
        joblib.dump(profit_rf_fb, Config.PROFIT_MODEL_RF_FALLBACK_PATH, compress=3)
        logger.info("Profit model (tuned Gradient Boosting, log1p target) saved to %s", Config.PROFIT_MODEL_PATH)

        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        s_hat_all = _inv_log_pred(sales_gbr.predict(X_sales), True)
        sales_scale = float(y_sales.mean() / max(1e-6, np.mean(s_hat_all)))

        joblib.dump(SALES_FEATURE_COLS, Config.SALES_FEATURE_NAMES_PATH)
        joblib.dump(PROFIT_FEATURE_COLS, Config.PROFIT_FEATURE_NAMES_PATH)
        joblib.dump(
            {
                "trend_ord_min": tmin,
                "trend_ord_max": tmax,
                "sales_scale": sales_scale,
                "sales_log1p": True,
                "profit_log1p": True,
            },
            Config.FORECAST_TRAIN_META_PATH,
        )

        X_legacy, _, _, _ = self.preprocessor.prepare_features(df, include_target=True)
        joblib.dump(X_legacy.columns.tolist(), Config.MODELS_DIR / "feature_names.pkl")

        self.models = {
            "sales": sales_gbr,
            "demand": demand_gbr,
            "profit": profit_gbr,
        }

        logger.info("All models trained (production = tuned Gradient Boosting per target).")

        te_d = compute_metrics(yd_test, demand_gbr.predict(Xd_test))
        te_s = compute_metrics(
            y_test_s, _inv_log_pred(sales_gbr.predict(X_test_s), True)
        )
        te_p = compute_metrics(
            y_test_p, _inv_log_pred(profit_gbr.predict(X_test_p), True)
        )

        return {
            "sales_metrics": {"mae": te_s["mae"], "rmse": te_s["rmse"], "r2": te_s["r2"]},
            "demand_metrics": {"mae": te_d["mae"], "rmse": te_d["rmse"], "r2": te_d["r2"]},
            "profit_metrics": {"mae": te_p["mae"], "rmse": te_p["rmse"], "r2": te_p["r2"]},
        }

    def load_models(self):
        try:
            self.models["sales"] = joblib.load(Config.SALES_MODEL_PATH)
            self.models["demand"] = joblib.load(Config.DEMAND_MODEL_PATH)
            self.models["profit"] = joblib.load(Config.PROFIT_MODEL_PATH)
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False


if __name__ == "__main__":
    trainer = ModelTrainer()
    metrics = trainer.train_all_models()
    print("\nTraining Results (tuned Gradient Boosting — test set):")
    if metrics:
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name}:")
            for metric, value in model_metrics.items():
                print(f"  {metric}: {value:.4f}")
    else:
        print("Training failed.")
