import logging
import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterSampler
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import xgboost as xgb
from collections import defaultdict
from datetime import datetime
from pandas_gbq import to_gbq
from google.cloud import storage
import joblib
import os
from datetime import date as dt_date

# -------------------- Set Up Logging --------------------
date = datetime.now().strftime("%Y-%m-%d")

log_filename = f'monthly_model_run_{date}.log'
log_file_path = f"/tmp/{log_filename}.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# -------------------- Connect & Load Data From GBQ --------------------
project_id = "behavio-test"
client = bigquery.Client(project=project_id)

logger.info("Reading data from BigQuery...")

df_atlas = client.query("SELECT * FROM `behavio-test.Data.AtlasCechu`").to_dataframe()
df_payments = client.query("SELECT * FROM `behavio-test.Data.Payments`").to_dataframe()
df_credits = client.query("SELECT * FROM `behavio-test.Data.UserCredits`").to_dataframe()

# Clean and preprocess
df_credits_cleaned = df_credits[df_credits['credits'] > 0]
df_payments_cleaned = df_payments[df_payments['user'].notna()]

# -------------------- Processing Atlas columns --------------------
logger.info("Encoding structured columns from Atlas...")
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
    
df_atlas_numeric_values = pd.DataFrame(structured_data).drop(columns=['user_id'])
df_atlas_nv = pd.concat([df_atlas[['user_id']], df_atlas_numeric_values], axis=1)

mapping_dicts = {}
for col in df_atlas_nv.columns:
    if col == 'user_id':
        continue
    unique_lists = df_atlas_nv[col].apply(lambda x: tuple(sorted(x))).unique()
    mapping_dicts[col] = {lst: idx + 1 for idx, lst in enumerate(unique_lists)}
    df_atlas_nv[col] = df_atlas_nv[col].apply(lambda x: mapping_dicts[col][tuple(sorted(x))])
    
# -------------------- Logic for target Columns --------------------
logger.info("Aggregating withdrawal stats")
withdrawals = df_payments[(df_payments['credits'] >= 500) & (df_payments['state'].isin(['PAID', 'APPROVED']))]
withdrawal_stats = withdrawals.groupby('user').agg(
    num_withdrawals=('credits', 'count'),
    avg_withdrawal=('credits', 'mean'),
    total_withdrawn=('credits', 'sum')
).reset_index()

# Identify currently eligible users
logger.info("Building eligible user set")
# Users with 480 credits can be eligible in the near future for cash out, even tho the withdrawal needs to be above or equal to the 500
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

def likely_to_withdraw_now(row):
    if row['num_withdrawals'] == 0:
        return 0
    expected_threshold = max(0.9 * row['avg_withdrawal'], 450)
    return int(row['credits'] >= expected_threshold)

eligible_now['target'] = eligible_now.apply(likely_to_withdraw_now, axis=1)

# -------------------- Merge Datasets & Feature Engineering --------------------
logger.info("Merging structured Atlas features")
df_atlas_clean_nv = df_atlas_nv.copy()
df_atlas_clean_nv.rename(columns={'user_id': 'user'}, inplace=True)
df_model = pd.merge(eligible_now, df_atlas_clean_nv, on='user', how='left')

X = df_model.drop(columns=[
    'target', 'user', 'credits',
    'num_withdrawals', 'avg_withdrawal', 'total_withdrawn', 
    'is_active', 'is_verified', 'is_locked'
])
y = df_model['target']
X = X.fillna(0)

# Normalize features
logger.info("Scaling features")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

# -------------------- Cross-validation Set up & Hyperparameter tuning --------------------
logger.info("Starting cross-validated hyperparameter search")
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

best_f1 = 0
best_threshold = 0.5
best_params = None
best_booster = None
best_report = ""

param_dist = {
    'learning_rate': [0.1, 0.15, 0.2, 0.25],
    'max_depth': [6, 7, 8],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.9, 1.0],
    'scale_pos_weight': [1.0, 1.5, 2.0],
}
param_list = list(ParameterSampler(param_dist, n_iter=200, random_state=42))

def f1_eval(preds, dtrain):
    labels = dtrain.get_label()
    preds_binary = (preds > 0.5).astype(int)
    return 'f1', f1_score(labels, preds_binary)

# Run training loop with cross-validation
for params in tqdm(param_list, desc="CV Search"):
    full_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 42,
        **params
    }
    cv_f1_scores = []

    for train_idx, valid_idx in cv.split(X_scaled, y):
        X_train_cv, X_valid_cv = X_scaled[train_idx], X_scaled[valid_idx]
        y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[valid_idx]

        dtrain_cv = xgb.DMatrix(X_train_cv, label=y_train_cv)
        dvalid_cv = xgb.DMatrix(X_valid_cv, label=y_valid_cv)

        booster = xgb.train(
            full_params,
            dtrain_cv,
            num_boost_round=200,
            evals=[(dvalid_cv, 'eval')],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        y_proba = booster.predict(dvalid_cv)

        # Threshold tuning
        best_f1_cv = 0
        for thresh in np.linspace(0.3, 0.55, 30):
            y_pred_cv = (y_proba > thresh).astype(int)
            f1_cv = f1_score(y_valid_cv, y_pred_cv)
            best_f1_cv = max(best_f1_cv, f1_cv)

        cv_f1_scores.append(best_f1_cv)

    mean_f1 = np.mean(cv_f1_scores)
    logger.info(f"Params: {params}, Mean F1: {mean_f1:.4f}")

    if mean_f1 > best_f1:
        best_f1 = mean_f1
        best_params = full_params
        best_booster = booster
        best_threshold = thresh
        y_final_pred = (y_proba > best_threshold).astype(int)
        best_report = classification_report(y_valid_cv, y_final_pred, output_dict=False)

logger.info(f"Best Mean F1: {best_f1:.4f} at threshold={best_threshold:.2f}")
logger.info(f"Best Params: {best_params}")
logger.info(f"Best CV Fold Report:\n{best_report}")

# Define GCS bucket and target path
bucket_name = "withdraws-model"
gcs_folder = "xgb_model"
model_dir = "/tmp/model_export"

# Create local folder to save model files
os.makedirs(model_dir, exist_ok=True)

# Save model artifacts locally
model_path = os.path.join(model_dir, "best_xgb_model.json")
best_booster.save_model(model_path)

joblib.dump(scaler, os.path.join(model_dir, "standard_scaler.pkl"))
joblib.dump(best_threshold, os.path.join(model_dir, "best_threshold.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(model_dir, "feature_names.pkl"))

# -------------------- Save Model to the GCS --------------------
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

def upload_to_gcs(local_path, blob_name):
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{blob_name}")

for filename in os.listdir(model_dir):
    local_path = os.path.join(model_dir, filename)
    upload_to_gcs(local_path, f"{gcs_folder}/{filename}")

# ---------- SAVE TOP 20 FEATURE IMPORTANCES TO GBQ ----------
logger.info("Extracting top 20 feature importances...")

# Map f0, f1... to actual column names
booster_feature_map = {f"f{idx}": col for idx, col in enumerate(X.columns)}
importance_dict = best_booster.get_score(importance_type='gain')

# Remap keys
importance_named = {
    booster_feature_map.get(k, k): v for k, v in importance_dict.items()
}

# Create DataFrame (skip the withdrawal_segment_code column)
importance_df = pd.DataFrame(
    list(importance_named.items()), columns=["feature", "importance"]
).sort_values(by="importance", ascending=False).iloc[1:21]

# Add timestamp
importance_df["date"] = dt_date(2025, 4, 1)

# Reorder columns
importance_df = importance_df[["date", "feature", "importance"]]

# Define table name
feature_table_id = "behavio-test.Data.WithdrawalModelFeatureImportances"

# Save to BigQuery
to_gbq(
    dataframe=importance_df,
    destination_table=feature_table_id,
    project_id="behavio-test",
    if_exists="append"
)
logger.info(f"Saved top 20 feature importances to BigQuery table `{feature_table_id}`")

#----------Predict the cash outs------------
X_pred = df_model.drop(columns=[
    'target', 'user', 'credits',
    'num_withdrawals', 'avg_withdrawal', 'total_withdrawn',
    'is_active', 'is_verified', 'is_locked'
], errors='ignore')

X_pred = X_pred.fillna(0)

X_pred_scaled = scaler.transform(X_pred)
dpred = xgb.DMatrix(X_pred_scaled)

y_pred_proba_final = best_booster.predict(dpred)
df_model['prediction'] = (y_pred_proba_final > best_threshold).astype(int)

predicted_cashouts = df_model[df_model['prediction'] == 1]
total_predicted_credits = predicted_cashouts['credits'].sum()

# Round to the nearest thousand
def round_thousand(x):
    return int(round(x, -3))

# Calculate ±5% bounds
lower_bound = round_thousand(total_predicted_credits * 0.95)
upper_bound = round_thousand(total_predicted_credits * 1.05)

# Count users predicted to withdraw
num_predicted_users = df_model['prediction'].sum()
logger.info(f"Estimated number of users who will cash out: {num_predicted_users}")
logger.info(f"Estimated reserve needed for upcoming withdrawals: {total_predicted_credits:,.0f} CZK")
logger.info(f"Estimated reserve range: {lower_bound:,} CZK – {upper_bound:,} CZK")

#---------- Save the Predictes Values to the Dataset in GBQ ------------
summary_df = pd.DataFrame([{
    # Used fixed date, because the available dataset of payments is to the 31st of March 2025,
    # so the run date is set to the 1st of April 2025. Uncomment the line below to use today's date.
    #"run_date": dt_date.today(),
    "run_date": dt_date(2025, 4, 1),
    "predicted_users": int(num_predicted_users),
    "predicted_credits": int(total_predicted_credits),
    "predicted_credits_lower": int(lower_bound),
    "predicted_credits_upper": int(upper_bound),
}])

# Write to BigQuery
summary_table_id = "behavio-test.Data.WithdrawalForecastSummary"

to_gbq(
    dataframe=summary_df,
    destination_table=summary_table_id,
    project_id="behavio-test",
    if_exists="append"
)
logger.info(f"Saved forecast summary to BigQuery table `{summary_table_id}`")

#---------- Save the Log of the Run to the GCS ------------
def upload_log_to_gcs(local_log_path, bucket_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_log_path)
    logger.info(f"Uploaded log to GCS: gs://{bucket_name}/{destination_blob_name}")

upload_log_to_gcs(
    local_log_path=log_file_path,
    bucket_name=bucket_name,
    destination_blob_name=f"logs/{log_filename}"
)