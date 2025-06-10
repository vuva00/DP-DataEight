import os
import logging
from datetime import datetime
import pandas as pd
from google.cloud import bigquery, storage
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
from pandas_gbq import to_gbq
from collections import defaultdict
from datetime import date as dt_date

# Setup
project_id = "behavio-test"
bucket_name = "withdraws-model"
gcs_folder = "xgb_model"
dataset_id = "Data"
summary_table_id = "WithdrawalForecastSummary"
importance_table_id = "WithdrawalModelFeatureImportances"
summary_bq_table = f"{project_id}.{dataset_id}.{summary_table_id}"
importance_bq_table = f"{project_id}.{dataset_id}.{importance_table_id}"

# -------------------- Logging --------------------

date = datetime.now().strftime("%Y-%m-%d")

log_filename = f'monthly_model_run_{date}.log'
log_file = f"/tmp/{log_filename}.log"

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# -------------------- Load Data --------------------
client = bigquery.Client(project=project_id)
logger.info("Loading AtlasCechu and UserCredits...")

df_atlas = client.query("SELECT * FROM `behavio-test.Data.AtlasCechu`").to_dataframe()
df_payments = client.query("SELECT * FROM `behavio-test.Data.Payments`").to_dataframe()
df_credits = client.query("SELECT * FROM `behavio-test.Data.UserCredits`").to_dataframe()

# Clean and preprocess
df_credits_cleaned = df_credits[df_credits['credits'] > 0]
df_payments_cleaned = df_payments[df_payments['user'].notna()]

# -------------------- Atlas preprocessing --------------------

logger.info("Processing AtlasCechu...")
grouped_cols = defaultdict(dict)
for col in df_atlas.columns:
    if "-" in col:
        group, key = col.split('-', 1)
        grouped_cols[group][key] = col
    else:
        grouped_cols[col][col] = col

structured_data = []
for _, row in df_atlas.iterrows():
    entry = {}
    for group, mapping in grouped_cols.items():
        entry[group] = [key for key, col in mapping.items() if row[col] == 1]
    structured_data.append(entry)
    
df_atlas_numeric_values = pd.DataFrame(structured_data).drop(columns=['user_id'], errors='ignore')
df_atlas_nv = pd.concat([df_atlas[['user_id']], df_atlas_numeric_values], axis=1)

mapping_dicts = {}
for col in df_atlas_nv.columns:
    if col == 'user_id':
        continue
    unique_lists = df_atlas_nv[col].apply(lambda x: tuple(sorted(x))).unique()
    mapping_dicts[col] = {lst: idx + 1 for idx, lst in enumerate(unique_lists)}
    df_atlas_nv[col] = df_atlas_nv[col].apply(lambda x: mapping_dicts[col][tuple(sorted(x))])

withdrawals = df_payments[(df_payments['credits'] >= 500) & (df_payments['state'].isin(['PAID', 'APPROVED']))]
withdrawal_stats = withdrawals.groupby('user').agg(
    num_withdrawals=('credits', 'count'),
    avg_withdrawal=('credits', 'mean'),
    total_withdrawn=('credits', 'sum')
).reset_index()

eligible_now = df_credits[df_credits['credits'] >= 480].copy()
eligible_now = eligible_now.merge(withdrawal_stats, on='user', how='left')
eligible_now[['num_withdrawals', 'avg_withdrawal', 'total_withdrawn']] = eligible_now[
    ['num_withdrawals', 'avg_withdrawal', 'total_withdrawn']
].fillna(0)

def assign_behavior(row):
    if row['num_withdrawals'] == 0:
        return 'new'
    elif row['num_withdrawals'] <= 2:
        return 'occasional'
    else:
        return 'regular'

eligible_now['withdrawal_segment'] = eligible_now.apply(assign_behavior, axis=1)
segment_mapping = {'new': 0, 'occasional': 1, 'regular': 2}
eligible_now['withdrawal_segment_code'] = eligible_now['withdrawal_segment'].map(segment_mapping)
eligible_now.drop(columns=['withdrawal_segment', 'wage'], inplace=True, errors='ignore')

# -------------------- Merge + Feature Prep --------------------
logger.info("Merging and preparing features...")
df_atlas_clean_nv = df_atlas_nv.copy()
df_atlas_clean_nv.rename(columns={'user_id': 'user'}, inplace=True)
df_model = pd.merge(eligible_now, df_atlas_clean_nv, on='user', how='left')
df_model.fillna(0, inplace=True)

# -------------------- Load Model & Artifacts --------------------
logger.info("Loading model and artifacts from GCS...")
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

def download_blob(filename):
    blob = bucket.blob(f"{gcs_folder}/{filename}")
    path = f"/tmp/{filename}"
    blob.download_to_filename(path)
    return path

model_path = download_blob("best_xgb_model.json")
scaler_path = download_blob("standard_scaler.pkl")
threshold_path = download_blob("best_threshold.pkl")
features_path = download_blob("feature_names.pkl")

booster = xgb.Booster()
booster.load_model(model_path)
scaler = joblib.load(scaler_path)
best_threshold = joblib.load(threshold_path)
feature_names = joblib.load(features_path)

# -------------------- Predict --------------------
X_pred = df_model[feature_names]
X_pred_scaled = scaler.transform(X_pred)
dpred = xgb.DMatrix(X_pred_scaled)

logger.info("Running prediction...")
y_proba = booster.predict(dpred)
df_model['prediction'] = (y_proba > best_threshold).astype(int)

predicted = df_model[df_model['prediction'] == 1]
predicted_credits = int(predicted['credits'].sum())
predicted_users = int(predicted.shape[0])
interval_low = int(predicted_credits * 0.95 // 1000 * 1000)
interval_high = int(predicted_credits * 1.1 // 1000 * 1000)

# -------------------- Log Predictions --------------------
logger.info(f"Predicted users to withdraw: {predicted_users}")
logger.info(f"Predicted withdrawal credits: {predicted_credits:,} CZK")
logger.info(f"Estimated range: {interval_low:,} CZK – {interval_high:,} CZK")

# -------------------- Save Results to BigQuery --------------------
summary_df = pd.DataFrame([{
    "run_date": dt_date.today(),
    "predicted_users": predicted_users,
    "predicted_credits": predicted_credits,
    "predicted_credits_lower": interval_low,
    "predicted_credits_upper": interval_high
}])

def check_if_exists(table_id, date_column="run_date"):
    query = f"""
    SELECT COUNT(*) as count
    FROM `{table_id}`
    WHERE {date_column} = DATE('{dt_date.today()}')
    """
    return client.query(query).to_dataframe().iloc[0]["count"] > 0

summary_exists = check_if_exists(summary_bq_table)
summary_mode = "replace" if summary_exists else "append"
to_gbq(summary_df, summary_bq_table, project_id=project_id, if_exists=summary_mode)
logger.info(f"{'Replaced' if summary_exists else 'Appended'} forecast to BigQuery table {summary_bq_table}")


# -------------------- Save Top 20 Feature Importances to BigQuery --------------------
logger.info("Saving top 20 feature importances to BigQuery...")

# Get feature importances by gain
booster_feature_map = {f"f{idx}": col for idx, col in enumerate(feature_names)}
importance_dict = booster.get_score(importance_type='gain')

# Rename f0, f1... to actual column names
importance_named = {
    booster_feature_map.get(k, k): v for k, v in importance_dict.items()
}

# Create DataFrame (skip the withdrawal_segment_code column)
importance_df = pd.DataFrame(
    list(importance_named.items()), columns=["feature", "importance"]
).sort_values(by="importance", ascending=False).iloc[1:21]

importance_df["run_date"] = dt_date.today()
importance_df = importance_df[["run_date", "feature", "importance"]]

importance_exists = check_if_exists(importance_bq_table)
importance_mode = "replace" if importance_exists else "append"

feature_table_id = "behavio-test.Data.WithdrawalModelFeatureImportances"

to_gbq(importance_df, importance_bq_table, project_id=project_id, if_exists=importance_mode)
logger.info(f"{'Replaced' if importance_exists else 'Appended'} top 20 feature importances to BigQuery table `{importance_bq_table}`")

# -------------------- Upload Log --------------------
def upload_log():
    blob = bucket.blob(f"logs/weekly_model_run_{datetime.now().strftime('%Y-%m-%d')}.log")
    blob.upload_from_filename(log_file)
    logger.info("Uploaded log to GCS")

upload_log()