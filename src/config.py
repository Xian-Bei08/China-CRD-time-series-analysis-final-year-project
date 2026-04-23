from pathlib import Path

# =========================
# Project paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# =========================
# Input files
# =========================
CRD_FILE = DATA_RAW / "MainDependentVariable.csv"
PM25_FILE = DATA_RAW / "AIR_POLLUTION_1990_2023_PM.CSV"
OZONE_FILE = DATA_RAW / "AIR_POLLUTION_1990_2021_OZONE.csv"
HAP_FILE = DATA_RAW / "AIR_POLLUTION_1990_2023_HAP_PM.CSV"
AGEING_FILE = DATA_RAW / "fuels_gdp_ageing.csv"
GOV_HEALTH_EXP_FILE = DATA_RAW / "Domestic general government health expenditure.csv"

# =========================
# Interim output files
# =========================
CRD_MAIN_FILE = DATA_INTERIM / "crd_main.csv"
PM25_MAIN_FILE = DATA_INTERIM / "pm25_main.csv"
OZONE_MAIN_FILE = DATA_INTERIM / "ozone_main.csv"
HOUSEHOLD_PM_MAIN_FILE = DATA_INTERIM / "household_pm_main.csv"
AGEING_HEALTH_MAIN_FILE = DATA_INTERIM / "ageing_health_main.csv"

# =========================
# Final processed output
# =========================
FINAL_ANALYSIS_FILE = DATA_PROCESSED / "final_analysis_dataset.csv"
FEATURE_FILE = DATA_PROCESSED / "feature_candidates_with_lags.csv"
MODELLING_DIFF_FILE = DATA_PROCESSED / "modelling_dataset_diff.csv"
MODELLING_LEVEL_FILE = DATA_PROCESSED / "modelling_dataset_level.csv"

# =========================
# Analysis settings
# =========================
LOCATION = "China"
LOCATION_ID = 6
SEX = "Both"
AGE_NAME = "Age-standardized"
METRIC = "Rate"
CRD_MEASURE = "DALYs (Disability-Adjusted Life Years)"
CRD_CAUSE = "Chronic respiratory diseases"

START_YEAR = 1990
END_YEAR = 2023
TRAIN_START_YEAR = 1990
TRAIN_END_YEAR = 2015
TEST_START_YEAR = 2016
TEST_END_YEAR = 2020

# =========================
# Variable settings for the new dissertation pipeline
# =========================
YEAR_COL = "year"
Y_COL = "d_crd_daly_rate"
LEVEL_Y_COL = "crd_daly_rate"
BASE_LEVEL_X_COLS = [
    "pm25",
    "ozone",
    "household_pm",
    "ageing_65_plus",
    "gov_health_exp_pct_gdp",
]
DIFF_X_COLS = [
    "d_pm25",
    "d_ozone",
    "d_household_pm",
    "d_ageing_65_plus",
    "d_gov_health_exp_pct_gdp",
]
SCREENING_CANDIDATE_X_COLS = [
    "d_pm25_lag0",
    "d_pm25_lag1",
    "d_pm25_lag2",
    "d_gov_health_exp_pct_gdp_lag0",
    "d_gov_health_exp_pct_gdp_lag1",
    "d_gov_health_exp_pct_gdp_lag2",
    "d_household_pm_lag1",
    "d_household_pm_lag2",
    "d_ageing_65_plus_lag1",
    "trend",
]
TREND_COL = "trend"

# =========================
# Model tuning settings
# =========================
HEALTH_EXP_IMPUTATION_METHOD = "linear_trend_extrapolation"
VIF_THRESHOLD = 10.0
ELASTIC_NET_L1_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
ELASTIC_NET_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_STATE = 42
LOOCV_MIN_FEATURES = 1
BACKWARD_PVALUE_THRESHOLD = 0.10
INNER_CV_SPLITS = 5

# =========================
# Results paths
# =========================
RESULTS_DIR = PROJECT_ROOT / "results"
PRECHECK_DIR = RESULTS_DIR / "prechecks"
BASELINE_OLS_DIR = RESULTS_DIR / "baseline_ols"
LASSO_DIR = RESULTS_DIR / "lasso"
REFINED_OLS_DIR = RESULTS_DIR / "refined_ols"
ARDL_DIR = RESULTS_DIR / "ardl"
BOUNDS_ECM_DIR = RESULTS_DIR / "bounds_ecm"
VALIDATION_DIR = RESULTS_DIR / "validation"
FEATURE_SCREENING_DIR = RESULTS_DIR / "feature_screening"
OLS_LOOCV_DIR = RESULTS_DIR / "ols_loocv"
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"

# =========================
# Output control
# =========================
MINIMAL_OUTPUT_MODE = True
RUN_ROBUSTNESS_CHECKS = False
RUN_BOUNDS_ECM = True
SAVE_OLS_SUMMARY_TEXT = True
SAVE_ARDL_SUMMARY_TEXT = True
SAVE_ACTUAL_VS_FITTED_SERIES = True
SAVE_DETAILED_VALIDATION_PATHS = True
SAVE_LASSO_SELECTION_FREQUENCY = False
SAVE_STATIONARITY_TABLES = True
