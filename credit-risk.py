import os
import zipfile
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.calibration import calibration_curve
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import optuna
except ImportError:
    optuna = None
try:
    import shap
except ImportError:
    shap = None

import kaggle

COMPETITION = "home-credit-default-risk"
DATA_DIR ='data_raw'
DB_NAME = 'CreditRisk.db'
ENCODING_BY_FILE = {
    "HomeCredit_columns_description.csv": "cp1252"
}
FORCE_RELOAD = False
VALIDATE_COLUMNS = False
RELOAD_ON_SCHEMA_MISMATCH = False
FILES_TO_LOAD = [
    'application_train.csv',
    'application_test.csv',
    'bureau.csv',
    'bureau_balance.csv',
    'credit_card_balance.csv',
    'installments_payments.csv',
    'POS_CASH_balance.csv',
    'previous_application.csv',
    'sample_submission.csv',
    'HomeCredit_columns_description.csv'
]
IMPORTANCE_TOP_N = None
TOP_N_BASE_FEATURES = 50
SQL_READ_CHUNK_SIZE = 50000
HOLDOUT_SIZE = 0.2
HOLDOUT_RANDOM_STATE = 42
USE_OPTUNA = False
OPTUNA_TRIALS = 30
OPTUNA_TIMEOUT = None
OPTUNA_METRIC = "auc"
OPTUNA_SEED = 42
SHOW_HOLDOUT_METRICS = True
RUN_SHAP = True
SHAP_SAMPLE_SIZE = 5000
SHAP_RANDOM_STATE = 42
SHAP_MAX_DISPLAY = 25
RISK_BAND_THRESHOLDS = (0.10, 0.15)
RISK_BAND_LABELS = ("Auto approve <=10%", "Manual review 10-15%", "Decline >15%")
LGBM_TUNED_PARAMS = {
    "n_estimators": 771,
    "learning_rate": 0.017189458732575952,
    "num_leaves": 121,
    "max_depth": -1,
    "min_child_samples": 48,
    "subsample": 0.8797198227571087,
    "colsample_bytree": 0.6551096139918666,
    "reg_alpha": 4.987349142404867,
    "reg_lambda": 1.9673168791130142,
}
CATEGORICAL_COLS = [
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
]
BINARY_MAPS = {
    "FLAG_OWN_CAR": {"Y": 1, "N": 0},
    "FLAG_OWN_REALTY": {"Y": 1, "N": 0},
    "NAME_CONTRACT_TYPE": {"Cash loans": 0, "Revolving loans": 1},
}
ALWAYS_INCLUDE_FEATURES = list(dict.fromkeys(list(BINARY_MAPS.keys()) + CATEGORICAL_COLS))

def has_kaggle_credentials():
    return bool(
        os.environ.get("KAGGLE_API_TOKEN")
        or (
            os.environ.get("KAGGLE_USERNAME")
            and os.environ.get("KAGGLE_KEY")
        )
    )

kaggle_credentials = has_kaggle_credentials()
if not kaggle_credentials:
    print(
        "WARNING: Kaggle credentials not found. Set KAGGLE_API_TOKEN or "
        "KAGGLE_USERNAME/KAGGLE_KEY if you need to download data."
    )

def is_valid_zip(path):
    return os.path.isfile(path) and zipfile.is_zipfile(path)

def is_zip_intact(path):
    if not is_valid_zip(path):
        return False
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False

def extracted_files_present(data_dir, filenames):
    return all(os.path.exists(os.path.join(data_dir, f)) for f in filenames)

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
zip_path = os.path.join(DATA_DIR, f"{COMPETITION}.zip")

if VALIDATE_COLUMNS:
    if not kaggle_credentials:
        print("Skipping download/unzip checks because Kaggle credentials are missing.")
    else:
        print(f"Authenticating with Kaggle and downloading {COMPETITION}...")

        try:
            if not is_zip_intact(zip_path):
                if os.path.exists(zip_path):
                    print("Found a corrupted zip. Deleting and re-downloading...")
                    os.remove(zip_path)
                kaggle.api.competition_download_files(COMPETITION, path=DATA_DIR, force=True)
            else:
                print("Zip already present and valid. Skipping download.")

            if not is_zip_intact(zip_path):
                raise RuntimeError("Download did not produce a valid zip file.")

            if not extracted_files_present(DATA_DIR, FILES_TO_LOAD):
                print("Unziping files...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(DATA_DIR)
                except Exception:
                    print("Extraction failed. Re-downloading zip and retrying...")
                    os.remove(zip_path)
                    kaggle.api.competition_download_files(COMPETITION, path=DATA_DIR, force=True)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(DATA_DIR)
                print("Download and extraction complete.")
            else:
                print("Files already extracted. Skipping unzip.")
        except Exception as e:
            print(f"ERROR: {e}")
            print("If this failed with 403 forbidden: Go to the competition page and accept the rules.")
            print("If it failed with 'Unauthorized': Set KAGGLE_USERNAME and KAGGLE_KEY.")
else:
    print("Skipping download/unzip checks because VALIDATE_COLUMNS is False.")

conn = sqlite3.connect(DB_NAME)
print(f'Connected to database {DB_NAME}')
files_to_load = FILES_TO_LOAD

def format_column_diff(cols, limit=8):
    cols_sorted = sorted(cols)
    if len(cols_sorted) <= limit:
        return ", ".join(cols_sorted)
    return ", ".join(cols_sorted[:limit]) + f", ... (+{len(cols_sorted) - limit} more)"

def get_table_columns(conn, table_name):
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()]

def get_csv_columns(file_path, encoding):
    try:
        return pd.read_csv(file_path, nrows=0, encoding=encoding).columns.tolist()
    except Exception as e:
        print(f" Could not read columns from {file_path}: {e}")
        return []

def validate_table_columns(conn, table_name, file_path, encoding):
    db_cols = get_table_columns(conn, table_name)
    if not db_cols:
        print(f" Table {table_name} has no columns.")
        return False

    csv_cols = get_csv_columns(file_path, encoding)
    if not csv_cols:
        print(f" No columns found in {file_path}.")
        return False

    clean_csv_cols = [c.replace(' ', '_').replace(':', '').replace('-', '_') for c in csv_cols]
    db_set = set(db_cols)
    csv_set = set(clean_csv_cols)
    missing_in_db = csv_set - db_set
    extra_in_db = db_set - csv_set

    if missing_in_db or extra_in_db:
        print(f" Column mismatch for {table_name}.")
        if missing_in_db:
            print(f"  Missing in DB: {format_column_diff(missing_in_db)}")
        if extra_in_db:
            print(f"  Extra in DB: {format_column_diff(extra_in_db)}")
        return False

    print(f" Columns OK for {table_name}.")
    return True

def load_numeric_df(conn, table_name, chunk_size=SQL_READ_CHUNK_SIZE):
    query = f'SELECT * FROM "{table_name}"'
    if chunk_size:
        chunk_iter = pd.read_sql_query(query, conn, chunksize=chunk_size)
        try:
            first_chunk = next(chunk_iter)
        except StopIteration:
            return pd.DataFrame()
        chunks = [first_chunk]
        chunks.extend(chunk_iter)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_sql_query(query, conn)
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
        df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    if "PAYMENT_RATE" not in df.columns:
        if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
            denom = df["AMT_CREDIT"]
            df["PAYMENT_RATE"] = np.where(denom > 0, df["AMT_ANNUITY"] / denom, np.nan)
    if "ANNUITY_INCOME_PERC" not in df.columns:
        if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            denom = df["AMT_INCOME_TOTAL"]
            df["ANNUITY_INCOME_PERC"] = np.where(denom > 0, df["AMT_ANNUITY"] / denom, np.nan)
    if "CREDIT_TO_INCOME_RATIO" not in df.columns:
        if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            denom = df["AMT_INCOME_TOTAL"]
            df["CREDIT_TO_INCOME_RATIO"] = np.where(denom > 0, df["AMT_CREDIT"] / denom, np.nan)
    if "GOODS_TO_CREDIT_RATIO" not in df.columns:
        if "AMT_GOODS_PRICE" in df.columns and "AMT_CREDIT" in df.columns:
            denom = df["AMT_CREDIT"]
            df["GOODS_TO_CREDIT_RATIO"] = np.where(denom > 0, df["AMT_GOODS_PRICE"] / denom, np.nan)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    keep_cols = list(dict.fromkeys(numeric_cols + cat_cols))
    return df[keep_cols]

def build_bureau_aggregates(conn):
    aggregations = {
        "DAYS_CREDIT": ["min", "max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "sum"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "is_active": ["sum", "mean"],
    }
    available_cols = set(get_table_columns(conn, "bureau"))
    select_cols = ["SK_ID_CURR"] + [c for c in aggregations.keys() if c != "is_active"]
    if "CREDIT_ACTIVE" in available_cols:
        select_cols.append("CREDIT_ACTIVE")
    select_cols = [c for c in dict.fromkeys(select_cols) if c in available_cols]
    if "SK_ID_CURR" not in select_cols:
        print("SK_ID_CURR not found in bureau table.")
        return None
    select_sql = ", ".join(f'"{c}"' for c in select_cols)
    df = pd.read_sql_query(f'SELECT {select_sql} FROM "bureau"', conn)
    if df.empty:
        print("No data found in bureau table.")
        return None

    if "CREDIT_ACTIVE" in df.columns:
        df["is_active"] = (df["CREDIT_ACTIVE"] == "Active").astype(int)

    available_aggs = {
        col: stats for col, stats in aggregations.items() if col in df.columns
    }
    if not available_aggs:
        print("No bureau columns available for aggregation.")
        return None

    agg_df = df.groupby("SK_ID_CURR").agg(available_aggs)
    agg_df.columns = [f"BURO_{col}_{stat}" for col, stat in agg_df.columns]
    return agg_df.reset_index()

def build_installments_aggregates(conn):
    available_cols = set(get_table_columns(conn, "installments_payments"))
    required = {
        "SK_ID_CURR",
        "DAYS_ENTRY_PAYMENT",
        "DAYS_INSTALMENT",
        "AMT_PAYMENT",
        "AMT_INSTALMENT",
    }
    if not required.issubset(available_cols):
        missing = sorted(required - available_cols)
        print(f"Missing installments columns: {', '.join(missing)}")
        return None

    select_cols = list(required)
    select_sql = ", ".join(f'"{c}"' for c in select_cols)
    df = pd.read_sql_query(f'SELECT {select_sql} FROM "installments_payments"', conn)
    if df.empty:
        print("No data found in installments_payments table.")
        return None

    days_late = (df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]).clip(lower=0)
    valid_payment = df[["AMT_PAYMENT", "AMT_INSTALMENT"]].notna().all(axis=1)
    payment_discrepancy = np.where(
        valid_payment,
        df["AMT_PAYMENT"] < df["AMT_INSTALMENT"],
        np.nan,
    )

    agg_df = df[["SK_ID_CURR"]].copy()
    agg_df["AVG_DAYS_LATE"] = days_late
    agg_df["MAX_DAYS_LATE"] = days_late
    agg_df["PAYMENT_DISCREPANCY"] = payment_discrepancy

    agg_df = agg_df.groupby("SK_ID_CURR").agg(
        AVG_DAYS_LATE=("AVG_DAYS_LATE", "mean"),
        MAX_DAYS_LATE=("MAX_DAYS_LATE", "max"),
        PAYMENT_DISCREPANCY=("PAYMENT_DISCREPANCY", "mean"),
    )
    return agg_df.reset_index()

def build_previous_app_aggregates(conn):
    aggregations = {
        "AMT_ANNUITY": ["max", "mean"],
        "AMT_APPLICATION": ["max", "mean"],
        "AMT_DOWN_PAYMENT": ["max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }
    available_cols = set(get_table_columns(conn, "previous_application"))
    required = {"SK_ID_CURR"}
    if not required.issubset(available_cols):
        print("SK_ID_CURR not found in previous_application table.")
        return None

    select_cols = ["SK_ID_CURR"]
    for col in aggregations.keys():
        if col in available_cols:
            select_cols.append(col)
    if "NAME_CONTRACT_STATUS" in available_cols:
        select_cols.append("NAME_CONTRACT_STATUS")
    if "APP_CREDIT_PERC" not in available_cols:
        if "AMT_APPLICATION" in available_cols and "AMT_CREDIT" in available_cols:
            select_cols.append("AMT_CREDIT")

    select_cols = list(dict.fromkeys(select_cols))
    select_sql = ", ".join(f'"{c}"' for c in select_cols)
    df = pd.read_sql_query(f'SELECT {select_sql} FROM "previous_application"', conn)
    if df.empty:
        print("No data found in previous_application table.")
        return None

    if "APP_CREDIT_PERC" not in df.columns:
        if "AMT_APPLICATION" in df.columns and "AMT_CREDIT" in df.columns:
            denom = df["AMT_CREDIT"]
            df["APP_CREDIT_PERC"] = np.where(denom > 0, df["AMT_APPLICATION"] / denom, np.nan)

    available_aggs = {
        col: stats for col, stats in aggregations.items() if col in df.columns
    }
    if not available_aggs:
        print("No previous_application columns available for aggregation.")
        return None

    agg_df = df.groupby("SK_ID_CURR").agg(available_aggs)
    agg_df.columns = [f"PREV_{col}_{stat}" for col, stat in agg_df.columns]

    count_df = df.groupby("SK_ID_CURR").size().rename("PREV_APP_COUNT")
    agg_df = agg_df.merge(count_df, left_index=True, right_index=True, how="left")

    if "NAME_CONTRACT_STATUS" in df.columns:
        df["PREV_APPROVED_FLAG"] = (df["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
        df["PREV_REFUSED_FLAG"] = (df["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
        status_df = df.groupby("SK_ID_CURR").agg(
            PREV_APPROVED_COUNT=("PREV_APPROVED_FLAG", "sum"),
            PREV_REFUSED_COUNT=("PREV_REFUSED_FLAG", "sum"),
        )
        agg_df = agg_df.merge(status_df, left_index=True, right_index=True, how="left")
        denom = agg_df["PREV_APP_COUNT"]
        agg_df["PREV_APPROVED_RATE"] = np.where(
            denom > 0,
            agg_df["PREV_APPROVED_COUNT"] / denom,
            np.nan,
        )
        agg_df["PREV_REFUSED_RATE"] = np.where(
            denom > 0,
            agg_df["PREV_REFUSED_COUNT"] / denom,
            np.nan,
        )

    return agg_df.reset_index()

def get_categorical_features(df, features):
    return [f for f in features if f in df.columns and pd.api.types.is_categorical_dtype(df[f])]

def get_top_features_by_importance(numeric_df, target_col="TARGET", top_n=50, random_state=42):
    if lgb is None:
        print("LightGBM is not installed. Install with: pip install lightgbm")
        return None
    if target_col not in numeric_df.columns:
        print(f"Target column {target_col} not found or not numeric.")
        return None

    model_df = numeric_df.dropna(subset=[target_col])
    X = model_df.drop(columns=[target_col])
    if "SK_ID_CURR" in X.columns:
        X = X.drop(columns=["SK_ID_CURR"])
    y = model_df[target_col]

    if X.empty:
        print("No features available for importance ranking.")
        return None

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        importance_type="gain",
    )
    cat_features = get_categorical_features(X, X.columns.tolist())
    fit_kwargs = {}
    if cat_features:
        fit_kwargs["categorical_feature"] = cat_features
    model.fit(X, y, **fit_kwargs)
    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    top_features = importance_df["feature"].head(top_n).tolist()
    return {"features": top_features, "importance_df": importance_df}

def select_top_features_lgbm(train_df, feature_pool, target_col="TARGET", top_n=50, random_state=42):
    if lgb is None:
        print("LightGBM is not installed. Install with: pip install lightgbm")
        return []
    if target_col not in train_df.columns:
        print(f"Target column {target_col} not found or not numeric.")
        return []

    candidate_features = [f for f in feature_pool if f in train_df.columns]
    if not candidate_features:
        print("No candidate features available for selection.")
        return []

    model_df = train_df.dropna(subset=[target_col])
    X = model_df[candidate_features]
    y = model_df[target_col]

    if X.empty:
        print("No features available for fold-level importance ranking.")
        return []

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        importance_type="gain",
    )
    cat_features = get_categorical_features(X, X.columns.tolist())
    fit_kwargs = {}
    if cat_features:
        fit_kwargs["categorical_feature"] = cat_features
    model.fit(X, y, **fit_kwargs)
    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return importance_df["feature"].head(top_n).tolist()

def compute_ks(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return np.max(tpr - fpr)

def safe_show(plot_name, output_dir="plots", show=None):
    if plt is None:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{plot_name}.png"
    if show is None:
        show = os.environ.get("SHOW_PLOTS", "").lower() in ("1", "true", "yes")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        try:
            plt.show()
        except Exception as exc:
            print(f"Plot display failed ({exc}). Saved to {output_path}")
    else:
        print(f"Saved plot to {output_path}")
    plt.close()

def compute_psi(expected, actual, bins=10, strategy="quantile"):
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    if expected.empty or actual.empty:
        return None

    if strategy == "quantile":
        try:
            _, bin_edges = pd.qcut(expected, q=bins, retbins=True, duplicates="drop")
        except ValueError:
            return None
    elif strategy == "uniform":
        bin_edges = np.linspace(expected.min(), expected.max(), bins + 1)
    else:
        return None

    if len(bin_edges) < 2 or np.allclose(bin_edges, bin_edges[0]):
        return None

    expected_bins = pd.cut(expected, bins=bin_edges, include_lowest=True)
    actual_bins = pd.cut(actual, bins=bin_edges, include_lowest=True)
    expected_pct = expected_bins.value_counts(normalize=True).sort_index()
    actual_pct = actual_bins.value_counts(normalize=True).sort_index()

    psi_df = pd.DataFrame(
        {
            "bin": expected_pct.index.astype(str),
            "expected_pct": expected_pct.values,
            "actual_pct": actual_pct.values,
        }
    )
    eps = 1e-6
    expected_adj = np.clip(psi_df["expected_pct"].to_numpy(), eps, None)
    actual_adj = np.clip(psi_df["actual_pct"].to_numpy(), eps, None)
    psi_df["psi"] = (actual_adj - expected_adj) * np.log(actual_adj / expected_adj)
    psi_value = float(psi_df["psi"].sum())
    return psi_value, psi_df

def plot_psi_distribution(psi_df, title="PSI - distribution"):
    if plt is None:
        return
    idx = np.arange(len(psi_df))
    width = 0.4
    plt.figure(figsize=(9, 4))
    plt.bar(idx - width / 2, psi_df["expected_pct"], width, label="Train")
    plt.bar(idx + width / 2, psi_df["actual_pct"], width, label="Holdout")
    plt.xticks(idx, psi_df["bin"], rotation=45, ha="right")
    plt.ylabel("Proportion")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def plot_calibration_curve(y_true, y_proba, bins=10, strategy="quantile"):
    if plt is None:
        return
    frac_pos, mean_pred = calibration_curve(
        y_true, y_proba, n_bins=bins, strategy=strategy
    )
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Ideal")
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout()

def compute_decile_table(y_true, y_score, n_bins=10):
    df = pd.DataFrame({"y_true": y_true, "score": y_score}).dropna()
    if df.empty:
        return None
    try:
        df["decile"] = pd.qcut(df["score"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    max_decile = df["decile"].max()
    if pd.isna(max_decile):
        return None
    df["decile"] = (max_decile - df["decile"]).astype(int) + 1
    summary = (
        df.groupby("decile")
        .agg(
            count=("y_true", "size"),
            bads=("y_true", "sum"),
            mean_score=("score", "mean"),
            bad_rate=("y_true", "mean"),
        )
        .sort_index()
    )
    summary["cum_bads"] = summary["bads"].cumsum()
    summary["cum_count"] = summary["count"].cumsum()
    summary["bad_rate_cum"] = summary["cum_bads"] / summary["cum_count"]
    return summary.reset_index()

def compute_risk_band_table(y_true, y_score, thresholds=(0.10, 0.15), labels=None):
    df = pd.DataFrame({"y_true": y_true, "score": y_score}).dropna()
    if df.empty:
        return None
    thresholds = sorted(thresholds)
    if labels is None:
        labels = [f"<= {thresholds[0]:.2%}"]
        for low, high in zip(thresholds, thresholds[1:]):
            labels.append(f"{low:.2%}-{high:.2%}")
        labels.append(f"> {thresholds[-1]:.2%}")
    if len(labels) != len(thresholds) + 1:
        return None
    bins = [-np.inf] + thresholds + [np.inf]
    df["band"] = pd.cut(df["score"], bins=bins, labels=labels, include_lowest=True)
    if df["band"].isna().all():
        return None
    summary = (
        df.groupby("band", observed=False)
        .agg(
            count=("y_true", "size"),
            bads=("y_true", "sum"),
            mean_score=("score", "mean"),
            bad_rate=("y_true", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reset_index()
    )
    total = summary["count"].sum()
    if total > 0:
        summary["share"] = summary["count"] / total
    total_bads = summary["bads"].sum()
    if total_bads > 0:
        summary["bad_share"] = summary["bads"] / total_bads
    return summary

def run_lgbm_cv(
    numeric_df,
    features=None,
    feature_pool=None,
    force_features=None,
    select_top_n=None,
    lgbm_params=None,
    target_col="TARGET",
    n_splits=5,
    random_state=42,
):
    if lgb is None:
        print("LightGBM is not installed. Install with: pip install lightgbm")
        return None
    if target_col not in numeric_df.columns:
        print(f"Target column {target_col} not found or not numeric.")
        return None
    use_selection = feature_pool is not None and select_top_n is not None
    if not use_selection and not features:
        print("Feature list is empty.")
        return None
    if use_selection:
        feature_pool = [f for f in feature_pool if f not in (target_col, "SK_ID_CURR")]
        if not feature_pool:
            print("Feature pool is empty.")
            return None

    if not use_selection:
        clean_features = [f for f in features if f != "SK_ID_CURR"]
        if not clean_features:
            print("Feature list is empty after removing SK_ID_CURR.")
            return None

    model_df = numeric_df.dropna(subset=[target_col])
    y = model_df[target_col]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc_scores = []
    train_auc_scores = []
    ks_scores = []
    f1_scores = []
    fpr_folds = []
    tpr_folds = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(model_df, y), start=1):
        train_df = model_df.iloc[train_idx]
        valid_df = model_df.iloc[valid_idx]

        if use_selection:
            fold_features = select_top_features_lgbm(
                train_df,
                feature_pool=feature_pool,
                target_col=target_col,
                top_n=select_top_n,
                random_state=random_state,
            )
            if force_features:
                fold_features = list(
                    dict.fromkeys(
                        fold_features + [f for f in force_features if f in train_df.columns]
                    )
                )
            clean_features = [
                f
                for f in fold_features
                if f not in ("SK_ID_CURR", target_col) and f in train_df.columns
            ]
            if not clean_features:
                print("Feature list is empty after fold-level selection.")
                return None

        X_train = train_df[clean_features]
        y_train = train_df[target_col]
        X_valid = valid_df[clean_features]
        y_valid = valid_df[target_col]

        model_params = dict(LGBM_TUNED_PARAMS)
        model_params.update(
            {
                "random_state": random_state,
                "n_jobs": -1,
                "importance_type": "gain",
            }
        )
        if lgbm_params:
            model_params.update(lgbm_params)
        if "random_state" not in model_params:
            model_params["random_state"] = random_state
        model = lgb.LGBMClassifier(**model_params)
        cat_features = get_categorical_features(X_train, X_train.columns.tolist())
        fit_kwargs = {}
        if cat_features:
            fit_kwargs["categorical_feature"] = cat_features
        model.fit(X_train, y_train, **fit_kwargs)
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba = model.predict_proba(X_valid)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y_valid, y_proba)
        fpr_folds.append(fpr)
        tpr_folds.append(tpr)
        auc_scores.append(roc_auc_score(y_valid, y_proba))
        train_auc_scores.append(roc_auc_score(y_train, y_proba_train))
        ks_scores.append(compute_ks(y_valid, y_proba))
        f1_scores.append(f1_score(y_valid, y_pred))

    mean_fpr = np.linspace(0, 1, 100)
    tpr_interps = []
    for fpr, tpr in zip(fpr_folds, tpr_folds):
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        tpr_interps.append(tpr_interp)
    mean_tpr = np.mean(tpr_interps, axis=0)
    std_tpr = np.std(tpr_interps, axis=0)

    return {
        "top_n": len(clean_features),
        "train_auc_mean": float(np.mean(train_auc_scores)),
        "train_auc_std": float(np.std(train_auc_scores)),
        "auc_mean": float(np.mean(auc_scores)),
        "auc_std": float(np.std(auc_scores)),
        "ks_mean": float(np.mean(ks_scores)),
        "ks_std": float(np.std(ks_scores)),
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "folds": n_splits,
        "roc_curves": {
            "fpr_folds": fpr_folds,
            "tpr_folds": tpr_folds,
            "mean_fpr": mean_fpr,
            "mean_tpr": mean_tpr,
            "std_tpr": std_tpr,
            "auc_folds": auc_scores,
        },
    }

def tune_lgbm_optuna(
    train_val_df,
    feature_pool,
    force_features,
    select_top_n,
    target_col="TARGET",
    n_splits=5,
    random_state=42,
    n_trials=30,
    timeout=None,
    metric="auc",
    optuna_seed=42,
):
    if optuna is None:
        print("Optuna is not installed. Install with: pip install optuna")
        return None
    if metric not in ("auc", "ks", "f1"):
        print(f"Unsupported metric {metric}. Use 'auc', 'ks', or 'f1'.")
        return None

    sampler = optuna.samplers.TPESampler(seed=optuna_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        cv_result = run_lgbm_cv(
            train_val_df,
            feature_pool=feature_pool,
            force_features=force_features,
            select_top_n=select_top_n,
            lgbm_params=params,
            target_col=target_col,
            n_splits=n_splits,
            random_state=random_state,
        )
        if cv_result is None:
            return 0.0
        if metric == "auc":
            return cv_result["auc_mean"]
        if metric == "ks":
            return cv_result["ks_mean"]
        return cv_result["f1_mean"]

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "metric": metric,
    }

def fit_lgbm_full(numeric_df, features, target_col="TARGET", random_state=42, lgbm_params=None):
    if lgb is None:
        print("LightGBM is not installed. Install with: pip install lightgbm")
        return None
    if target_col not in numeric_df.columns:
        print(f"Target column {target_col} not found or not numeric.")
        return None
    clean_features = [f for f in features if f != "SK_ID_CURR"]
    if not clean_features:
        print("Feature list is empty.")
        return None

    model_df = numeric_df[[target_col] + clean_features].dropna(subset=[target_col])
    X = model_df[clean_features]
    y = model_df[target_col]

    model_params = dict(LGBM_TUNED_PARAMS)
    model_params.update(
        {
            "random_state": random_state,
            "n_jobs": -1,
            "importance_type": "gain",
        }
    )
    if lgbm_params:
        model_params.update(lgbm_params)
    if "random_state" not in model_params:
        model_params["random_state"] = random_state
    model = lgb.LGBMClassifier(**model_params)
    cat_features = get_categorical_features(X, X.columns.tolist())
    fit_kwargs = {}
    if cat_features:
        fit_kwargs["categorical_feature"] = cat_features
    model.fit(X, y, **fit_kwargs)
    return {"model": model, "features": clean_features}

for filename in files_to_load:
    table_name = filename.replace('.csv','')
    file_path = os.path.join(DATA_DIR, filename)
    encoding = ENCODING_BY_FILE.get(filename, "utf-8")

    query_check = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
    table_exists = conn.execute(query_check).fetchone() is not None
    if FORCE_RELOAD and table_exists:
        print(f' Table {table_name} exists. Dropping for reload.')
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.commit()
        table_exists = False

    if table_exists and VALIDATE_COLUMNS:
        is_valid = validate_table_columns(conn, table_name, file_path, encoding)
        if not is_valid and RELOAD_ON_SCHEMA_MISMATCH:
            print(f" Schema mismatch detected. Dropping {table_name} for reload.")
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            conn.commit()
            table_exists = False

    if not table_exists:
        print(f' Loading {filename} into SQL... (This may take a moment)')

        chunk_size = 50000
        if encoding != "utf-8":
            print(f" Using encoding {encoding} for {filename}")
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, encoding=encoding)):
            chunk.columns = [c.replace(' ', '_').replace(':', '').replace('-', '_') for c in chunk.columns]
            chunk.to_sql(table_name, conn, if_exists='append', index=False)
            print(f' - Table {table_name} ready')
    else:
        print(f' Table {table_name} already exists. Skipping load.')

application_train_df = load_numeric_df(conn, "application_train")
base_feature_pool = [
    c for c in application_train_df.columns if c not in ("TARGET", "SK_ID_CURR")
]

model_df = application_train_df.dropna(subset=["TARGET"])
feature_result = None
train_ids = None
holdout_ids = None
if model_df.empty:
    print("No training rows found after dropping missing TARGET.")
else:
    train_ids, holdout_ids = train_test_split(
        model_df["SK_ID_CURR"],
        test_size=HOLDOUT_SIZE,
        random_state=HOLDOUT_RANDOM_STATE,
        stratify=model_df["TARGET"],
    )
    train_val_base = model_df[model_df["SK_ID_CURR"].isin(train_ids)][
        ["TARGET"] + base_feature_pool
    ]
    feature_result = get_top_features_by_importance(
        train_val_base,
        target_col="TARGET",
        top_n=TOP_N_BASE_FEATURES,
        random_state=42,
    )
if feature_result:
    base_features = feature_result["features"]

    bureau_agg = build_bureau_aggregates(conn)
    bureau_features = []
    if bureau_agg is not None:
        application_train_df = application_train_df.merge(bureau_agg, on="SK_ID_CURR", how="left")
        bureau_features = [c for c in bureau_agg.columns if c != "SK_ID_CURR"]
        if (
            "AMT_INCOME_TOTAL" in application_train_df.columns
            and "BURO_AMT_CREDIT_SUM_DEBT_sum" in application_train_df.columns
            and "RATIO_BUREAU_DEBT_TO_INCOME" not in application_train_df.columns
        ):
            denom = application_train_df["AMT_INCOME_TOTAL"]
            application_train_df["RATIO_BUREAU_DEBT_TO_INCOME"] = np.where(
                denom > 0,
                application_train_df["BURO_AMT_CREDIT_SUM_DEBT_sum"] / denom,
                np.nan,
            )
            bureau_features.append("RATIO_BUREAU_DEBT_TO_INCOME")

    installments_agg = build_installments_aggregates(conn)
    installments_features = []
    if installments_agg is not None:
        application_train_df = application_train_df.merge(installments_agg, on="SK_ID_CURR", how="left")
        installments_features = [c for c in installments_agg.columns if c != "SK_ID_CURR"]

    prev_agg = build_previous_app_aggregates(conn)
    prev_features = []
    if prev_agg is not None:
        application_train_df = application_train_df.merge(prev_agg, on="SK_ID_CURR", how="left")
        prev_features = [c for c in prev_agg.columns if c != "SK_ID_CURR"]
        if "AMT_ANNUITY" in application_train_df.columns and "PREV_AMT_ANNUITY_mean" in application_train_df.columns:
            denom = application_train_df["PREV_AMT_ANNUITY_mean"]
            application_train_df["RATIO_CURR_TO_PREV_ANNUITY"] = np.where(
                denom > 0,
                application_train_df["AMT_ANNUITY"] / denom,
                np.nan,
            )
            prev_features.append("RATIO_CURR_TO_PREV_ANNUITY")

    always_features = [c for c in ALWAYS_INCLUDE_FEATURES if c in application_train_df.columns]
    train_val_df = application_train_df[
        application_train_df["SK_ID_CURR"].isin(train_ids)
    ].dropna(subset=["TARGET"])
    holdout_df = application_train_df[
        application_train_df["SK_ID_CURR"].isin(holdout_ids)
    ].dropna(subset=["TARGET"])
    final_features = list(
        dict.fromkeys(
            base_features
            + always_features
            + bureau_features
            + installments_features
            + prev_features
        )
    )
    print(
        f"Using {len(final_features)} features "
        f"({len(base_features)} base + {len(always_features)} forced + {len(bureau_features)} bureau + "
        f"{len(installments_features)} installments + {len(prev_features)} previous_app)."
    )
    print(
        f"Train/validation rows: {len(train_val_df)}; holdout rows: {len(holdout_df)} "
        f"({HOLDOUT_SIZE:.0%})."
    )

    tuned_params = None
    if USE_OPTUNA:
        optuna_result = tune_lgbm_optuna(
            train_val_df,
            feature_pool=base_feature_pool,
            force_features=always_features + bureau_features + installments_features + prev_features,
            select_top_n=TOP_N_BASE_FEATURES,
            target_col="TARGET",
            n_splits=5,
            random_state=42,
            n_trials=OPTUNA_TRIALS,
            timeout=OPTUNA_TIMEOUT,
            metric=OPTUNA_METRIC,
            optuna_seed=OPTUNA_SEED,
        )
        if optuna_result:
            tuned_params = optuna_result["best_params"]
            print(
                f"Optuna best {optuna_result['metric']} (CV): {optuna_result['best_value']:.6f}"
            )
            print(f"Optuna best params: {tuned_params}")

    cv_result = run_lgbm_cv(
        train_val_df,
        feature_pool=base_feature_pool,
        force_features=always_features + bureau_features + installments_features + prev_features,
        select_top_n=TOP_N_BASE_FEATURES,
        lgbm_params=tuned_params,
        target_col="TARGET",
        n_splits=5,
        random_state=42,
    )
    if cv_result:
        metrics_df = pd.DataFrame(
            [
                {
                    "model": "LightGBM",
                    "features": cv_result["top_n"],
                    "train_auc_mean": cv_result["train_auc_mean"],
                    "auc_mean": cv_result["auc_mean"],
                    "auc_std": cv_result["auc_std"],
                    "ks_mean": cv_result["ks_mean"],
                    "ks_std": cv_result["ks_std"],
                    "f1_mean": cv_result["f1_mean"],
                    "f1_std": cv_result["f1_std"],
                }
            ]
        )
        print("Validation metrics (CV):")
        print(metrics_df.to_string(index=False))

    fit_result = fit_lgbm_full(
        train_val_df,
        final_features,
        target_col="TARGET",
        random_state=42,
        lgbm_params=tuned_params,
    )
    if fit_result:
        model = fit_result["model"]
        holdout_pred = None
        if holdout_df.empty:
            if SHOW_HOLDOUT_METRICS:
                print("Holdout set is empty. Skipping final evaluation.")
        else:
            X_holdout = holdout_df[fit_result["features"]]
            y_holdout = holdout_df["TARGET"]
            holdout_pred = model.predict_proba(X_holdout)[:, 1]
            if SHOW_HOLDOUT_METRICS:
                y_pred = (holdout_pred >= 0.5).astype(int)
                holdout_metrics = pd.DataFrame(
                    [
                        {
                            "model": "LightGBM",
                            "features": len(fit_result["features"]),
                            "auc": roc_auc_score(y_holdout, holdout_pred),
                            "ks": compute_ks(y_holdout, holdout_pred),
                            "f1": f1_score(y_holdout, y_pred),
                        }
                    ]
                )
                print("Holdout metrics (intocable):")
                print(holdout_metrics.to_string(index=False))

        if holdout_pred is None:
            print("Holdout set is empty. Skipping PSI.")
        else:
            dev_pred = model.predict_proba(train_val_df[fit_result["features"]])[:, 1]
            psi_result = compute_psi(dev_pred, holdout_pred, bins=10, strategy="quantile")
            if psi_result is None:
                print("PSI could not be computed.")
            else:
                psi_value, psi_table = psi_result
                print(f"PSI (train vs holdout, score): {psi_value:.6f}")
                print(psi_table.to_string(index=False))
                if plt is None:
                    print("Matplotlib is not installed. Install with: pip install matplotlib")
                else:
                    plot_psi_distribution(
                        psi_table, title="PSI - score distribution (train vs holdout)"
                    )
                    safe_show("psi_score")

        if holdout_pred is None:
            print("Holdout set is empty. Skipping calibration curve.")
        else:
            plot_calibration_curve(y_holdout, holdout_pred, bins=10, strategy="quantile")
            safe_show("calibration_curve")

        if holdout_pred is None:
            print("Holdout set is empty. Skipping decile analysis.")
        else:
            decile_table = compute_decile_table(y_holdout, holdout_pred, n_bins=10)
            if decile_table is None:
                print("Decile table could not be computed.")
            else:
                view = decile_table[
                    ["decile", "count", "mean_score", "bad_rate", "bad_rate_cum"]
                ]
                print("Decile analysis (holdout):")
                print(view.to_string(index=False))

        if holdout_pred is None:
            print("Holdout set is empty. Skipping risk bands.")
        else:
            risk_table = compute_risk_band_table(
                y_holdout,
                holdout_pred,
                thresholds=RISK_BAND_THRESHOLDS,
                labels=RISK_BAND_LABELS,
            )
            if risk_table is None:
                print("Risk band table could not be computed.")
            else:
                view = risk_table[
                    [
                        "band",
                        "count",
                        "share",
                        "mean_score",
                        "bad_rate",
                        "min_score",
                        "max_score",
                    ]
                ]
                print("Risk bands (holdout):")
                print(view.to_string(index=False))

        coef = model.feature_importances_
        importance_df = pd.DataFrame(
            {"feature": fit_result["features"], "importance": coef}
        ).sort_values("importance", ascending=False)
        importance_view = importance_df.copy()
        if IMPORTANCE_TOP_N:
            importance_view = importance_view.head(IMPORTANCE_TOP_N)
        importance_view.insert(0, "rank", range(1, len(importance_view) + 1))
        if IMPORTANCE_TOP_N:
            title = f"Top feature importances (LightGBM gain, top {IMPORTANCE_TOP_N}):"
        else:
            title = "Feature importances (LightGBM gain, all features):"
        print(title)
        print(importance_view.to_string(index=False))
        if RUN_SHAP:
            if shap is None:
                print("SHAP is not installed. Install with: pip install shap")
            elif plt is None:
                print("Matplotlib is not installed. Install with: pip install matplotlib")
            else:
                sample_size = min(len(train_val_df), SHAP_SAMPLE_SIZE)
                if sample_size < 1:
                    print("Not enough rows for SHAP.")
                else:
                    sample_df = train_val_df[fit_result["features"]].sample(
                        n=sample_size, random_state=SHAP_RANDOM_STATE
                    )
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(sample_df)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    shap.summary_plot(
                        shap_values,
                        sample_df,
                        max_display=SHAP_MAX_DISPLAY,
                        show=False,
                    )
                    plt.tight_layout()
                    safe_show("shap_summary")
    if plt is None:
        print("Matplotlib is not installed. Install with: pip install matplotlib")
    elif cv_result and "roc_curves" in cv_result:
        roc_data = cv_result["roc_curves"]
        fpr_folds = roc_data["fpr_folds"]
        tpr_folds = roc_data["tpr_folds"]
        mean_fpr = roc_data["mean_fpr"]
        mean_tpr = roc_data["mean_tpr"]
        std_tpr = roc_data["std_tpr"]
        auc_folds = roc_data["auc_folds"]

        plt.figure(figsize=(7, 6))
        for idx, (fpr, tpr, auc_value) in enumerate(
            zip(fpr_folds, tpr_folds, auc_folds), start=1
        ):
            plt.plot(fpr, tpr, alpha=0.3, label=f"Fold {idx} AUC = {auc_value:.4f}")

        mean_auc = cv_result["auc_mean"]
        std_auc = cv_result["auc_std"]
        plt.plot(
            mean_fpr,
            mean_tpr,
            color="blue",
            linewidth=2,
            label=f"Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}",
        )
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color="blue", alpha=0.15)

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Cross-Validation)")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        safe_show("roc_cv")
            
