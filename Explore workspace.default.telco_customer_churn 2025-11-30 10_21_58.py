# Databricks notebook source
import pyspark.sql.functions as F 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix as cm, f1_score, precision_score, recall_score, roc_auc_score, classification_report, roc_curve
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
import matplotlib.pyplot as plt

# COMMAND ----------

path = "workspace.default.telco_customer_churn"
df = spark.table(path)

# COMMAND ----------



# COMMAND ----------

display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.select('Churn').distinct())

# COMMAND ----------

df = df.drop("customerID")

# Churn
df = df.replace("Yes", "1", subset=['Churn']).replace("No", "0", subset=['Churn'])
df = df.withColumn("Churn", F.col("Churn").cast("int"))

# Gender
df = df.replace("Male", "1", subset=['gender']).replace("Female", "0", subset=['gender'])
df = df.withColumn("gender", F.col("gender").cast("int"))

# totalcharges
df = df.withColumn("TotalCharges", 
                   F.when(F.trim(F.col("TotalCharges")) == "", None)
                   .otherwise(F.col("TotalCharges")))
df = df.withColumn("TotalCharges", F.col("TotalCharges").cast("float"))


# monthlycharges
df = df.withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("float"))

# COMMAND ----------

pandas_df = df.toPandas()

# COMMAND ----------

# Handle missing values in TotalCharges
pandas_df['TotalCharges'] = pandas_df['TotalCharges'].fillna(pandas_df['TotalCharges'].median())

# Separate features and target
X = pandas_df.drop('Churn', axis=1)
y = pandas_df['Churn']

# Identify categorical columns
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                    'PaperlessBilling', 'PaymentMethod']

# COMMAND ----------

# Encode all categorical variables using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks")

with mlflow.start_run():
    # Train Model
    rf = RandomForestClassifier(
        n_estimators=200,  # Increased for better performance
        random_state=42,
        class_weight='balanced',
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt'
    )
    rf.fit(X_train, y_train)

    # Get prediction probabilities
    y_pred_proba = rf.predict_proba(X_test)[:,1]
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Calculate Youden's J statistic for each threshold
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Make predictions with optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Also test with business-oriented threshold (favor recall)
    business_threshold = 0.35
    y_pred_business = (y_pred_proba >= business_threshold).astype(int)
    
    # Calculate metrics for both thresholds
    # Optimal threshold metrics
    acc_opt = accuracy_score(y_test, y_pred_optimal)
    precision_opt = precision_score(y_test, y_pred_optimal)
    recall_opt = recall_score(y_test, y_pred_optimal)
    f1_opt = f1_score(y_test, y_pred_optimal)
    conf_matrix_opt = cm(y_test, y_pred_optimal)
    
    # Business threshold metrics
    acc_bus = accuracy_score(y_test, y_pred_business)
    precision_bus = precision_score(y_test, y_pred_business)
    recall_bus = recall_score(y_test, y_pred_business)
    f1_bus = f1_score(y_test, y_pred_business)
    conf_matrix_bus = cm(y_test, y_pred_business)
    
    # ROC AUC (threshold-independent)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Log metrics for optimal threshold
    mlflow.log_metric("accuracy_optimal", acc_opt)
    mlflow.log_metric("precision_optimal", precision_opt)
    mlflow.log_metric("recall_optimal", recall_opt)
    mlflow.log_metric("f1_score_optimal", f1_opt)
    
    # Log metrics for business threshold
    mlflow.log_metric("accuracy_business", acc_bus)
    mlflow.log_metric("precision_business", precision_bus)
    mlflow.log_metric("recall_business", recall_bus)
    mlflow.log_metric("f1_score_business", f1_bus)
    
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("optimal_threshold", optimal_threshold)
    
    # Log parameters
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("business_threshold", business_threshold)
    
    # Create signature and log model
    signature = infer_signature(X_train, rf.predict_proba(X_train))
    mlflow.sklearn.log_model(rf, "random_forest_model", signature=signature)

    # Print results
    print("=" * 70)
    print("MODEL TRAINING COMPLETED - THRESHOLD COMPARISON")
    print("=" * 70)
    
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    print(f"Optimal Threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"Business Threshold: {business_threshold:.4f}")
    
    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLD RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {acc_opt:.4f}")
    print(f"Precision: {precision_opt:.4f}")
    print(f"Recall:    {recall_opt:.4f}")
    print(f"F1 Score:  {f1_opt:.4f}")
    print("\nConfusion Matrix:")
    print(f"                Predicted No Churn  Predicted Churn")
    print(f"Actual No Churn      {conf_matrix_opt[0][0]:>6}            {conf_matrix_opt[0][1]:>6}")
    print(f"Actual Churn         {conf_matrix_opt[1][0]:>6}            {conf_matrix_opt[1][1]:>6}")
    
    print("\n" + "=" * 70)
    print("BUSINESS THRESHOLD RESULTS (Optimized for Recall)")
    print("=" * 70)
    print(f"Accuracy:  {acc_bus:.4f}")
    print(f"Precision: {precision_bus:.4f}")
    print(f"Recall:    {recall_bus:.4f}")
    print(f"F1 Score:  {f1_bus:.4f}")
    print("\nConfusion Matrix:")
    print(f"                Predicted No Churn  Predicted Churn")
    print(f"Actual No Churn      {conf_matrix_bus[0][0]:>6}            {conf_matrix_bus[0][1]:>6}")
    print(f"Actual Churn         {conf_matrix_bus[1][0]:>6}            {conf_matrix_bus[1][1]:>6}")
    
    # Business impact calculation
    churners_caught_opt = conf_matrix_opt[1][1]
    churners_missed_opt = conf_matrix_opt[1][0]
    false_alarms_opt = conf_matrix_opt[0][1]
    
    churners_caught_bus = conf_matrix_bus[1][1]
    churners_missed_bus = conf_matrix_bus[1][0]
    false_alarms_bus = conf_matrix_bus[0][1]
    
    print("\n" + "=" * 70)
    print("BUSINESS IMPACT ANALYSIS")
    print("=" * 70)
    print(f"\nOptimal Threshold ({optimal_threshold:.2f}):")
    print(f"  Churners Caught: {churners_caught_opt} ({churners_caught_opt/374*100:.1f}%)")
    print(f"  Churners Missed: {churners_missed_opt} ({churners_missed_opt/374*100:.1f}%)")
    print(f"  False Alarms: {false_alarms_opt}")
    
    print(f"\nBusiness Threshold ({business_threshold:.2f}):")
    print(f"  Churners Caught: {churners_caught_bus} ({churners_caught_bus/374*100:.1f}%)")
    print(f"  Churners Missed: {churners_missed_bus} ({churners_missed_bus/374*100:.1f}%)")
    print(f"  False Alarms: {false_alarms_bus}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "=" * 70)
    print("TOP 10 IMPORTANT FEATURES")
    print("=" * 70)
    print(feature_importance.head(10).to_string(index=False))
    print("=" * 70)
    
    print("\nRECOMMENDATION:")
    if recall_bus > recall_opt:
        print(f"Use BUSINESS threshold ({business_threshold}) for maximum customer retention!")
    else:
        print(f"Use OPTIMAL threshold ({optimal_threshold:.2f}) for best balanced performance!")

# COMMAND ----------

# MAGIC %md
# MAGIC # PHASE: DRIFT DETECTION & MONITORING

# COMMAND ----------

from scipy import stats
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType, IntegerType
import datetime
import json
import pandas as pd
import numpy as np

print("=" * 70)
print("DRIFT DETECTION - MANUAL IMPLEMENTATION")
print("=" * 70)

def calculate_drift_score(reference_data, current_data, threshold=0.05):
    """
    Calculate drift using Kolmogorov-Smirnov test for numerical features
    and Chi-square test for categorical features.
    
    Returns:
    - drift_score: % of features that have drifted
    - drift_details: list of drifted features with p-values
    """
    drift_details = []
    drifted_count = 0
    total_features = len(reference_data.columns)
    
    for column in reference_data.columns:
        ref_col = reference_data[column].dropna()
        curr_col = current_data[column].dropna()
        
        # Check if numeric or categorical
        if pd.api.types.is_numeric_dtype(reference_data[column]):
            # Use Kolmogorov-Smirnov test for numerical features
            statistic, p_value = stats.ks_2samp(ref_col, curr_col)
        else:
            # Use Chi-square test for categorical features
            ref_counts = ref_col.value_counts()
            curr_counts = curr_col.value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]
            
            try:
                statistic, p_value = stats.chisquare(curr_freq, ref_freq)
            except:
                p_value = 1.0  # No drift if test fails
                statistic = 0.0
        
        # Check if drifted (p-value < threshold indicates drift)
        is_drifted = p_value < threshold
        
        if is_drifted:
            drifted_count += 1
            drift_details.append({
                'feature': column,
                'p_value': float(p_value),
                'statistic': float(statistic),
                'drift_detected': True
            })
        
    drift_score = drifted_count / total_features if total_features > 0 else 0
    
    return drift_score, drift_details, drifted_count

# 1. Calculate Drift
print("\nCalculating drift between training and test data...")
drift_score, drift_details, n_drifted = calculate_drift_score(X_train, X_test, threshold=0.05)

print("\n" + "=" * 70)
print("DRIFT DETECTION RESULTS")
print("=" * 70)
print(f"Total Features: {len(X_train.columns)}")
print(f"Drifted Features: {n_drifted}")
print(f"Drift Score: {drift_score:.2%}")
print(f"Dataset Drift: {'YES' if drift_score > 0.3 else 'NO'}")

# 2. Show drifted features
if drift_details:
    print("\n" + "=" * 70)
    print("DRIFTED FEATURES (p-value < 0.05)")
    print("=" * 70)
    drift_df_display = pd.DataFrame(drift_details)
    drift_df_display = drift_df_display.sort_values('p_value')
    print(drift_df_display.to_string(index=False))
else:
    print("\n✅ No significant drift detected in any features!")

# 3. Calculate statistical summaries for key features
print("\n" + "=" * 70)
print("FEATURE STATISTICS COMPARISON")
print("=" * 70)

key_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract']
stats_comparison = []

for feature in key_features:
    if feature in X_train.columns:
        stats_comparison.append({
            'Feature': feature,
            'Train_Mean': X_train[feature].mean(),
            'Test_Mean': X_test[feature].mean(),
            'Train_Std': X_train[feature].std(),
            'Test_Std': X_test[feature].std(),
            'Mean_Diff_%': ((X_test[feature].mean() - X_train[feature].mean()) / X_train[feature].mean() * 100)
        })

stats_df = pd.DataFrame(stats_comparison)
print(stats_df.to_string(index=False))

# 4. Save to Delta Table
current_time = datetime.datetime.now()

# Determine threshold strategy
rec_strategy = "Optimal" if recall_opt >= recall_bus else "Business"
threshold_value = optimal_threshold if rec_strategy == "Optimal" else business_threshold

# Define schema
schema = StructType([
    StructField("check_time", TimestampType(), True),
    StructField("drift_score", FloatType(), True),
    StructField("n_drifted_columns", IntegerType(), True),
    StructField("dataset_drift", StringType(), True),
    StructField("model_version", StringType(), True),
    StructField("threshold_used", FloatType(), True),
    StructField("threshold_strategy", StringType(), True),
    StructField("model_accuracy", FloatType(), True),
    StructField("model_recall", FloatType(), True),
    StructField("model_roc_auc", FloatType(), True),
    StructField("drifted_features", StringType(), True)
])

# Create log data
log_data = [(
    current_time,
    float(drift_score),
    int(n_drifted),
    "YES" if drift_score > 0.3 else "NO",
    "v1_random_forest_200",
    float(threshold_value),
    rec_strategy,
    float(acc_opt if rec_strategy == "Optimal" else acc_bus),
    float(recall_opt if rec_strategy == "Optimal" else recall_bus),
    float(roc_auc),
    json.dumps(drift_details)
)]

# Create and save DataFrame
drift_log_df = spark.createDataFrame(log_data, schema)
table_name = "workspace.default.model_monitoring_logs"
drift_log_df.write.mode("append").saveAsTable(table_name)

print("\n" + "=" * 70)
print("✅ SUCCESS - MONITORING DATA SAVED")
print("=" * 70)
print(f"Table: {table_name}")
print(f"Drift Score: {drift_score:.2%}")
print(f"Model Version: v1_random_forest_200")
print(f"Timestamp: {current_time}")

# 5. Display monitoring history
print("\n" + "=" * 70)
print("MONITORING HISTORY (Last 10 Records)")
print("=" * 70)
monitoring_df = spark.table(table_name).orderBy("check_time", ascending=False).limit(10)
display(monitoring_df)

# 6. Drift Alert Logic
print("\n" + "=" * 70)
print("DRIFT ALERT STATUS")
print("=" * 70)
if drift_score > 0.3:
    print("⚠️  HIGH DRIFT DETECTED!")
    print("   Action Required: Model should be retrained with recent data")
    print(f"   {n_drifted} features have significantly drifted")
elif drift_score > 0.15:
    print("⚠️  MODERATE DRIFT DETECTED")
    print("   Action: Monitor closely, consider retraining soon")
else:
    print("✅ LOW DRIFT - Model is performing within expected parameters")
    print("   No immediate action required")

print("=" * 70)

# COMMAND ----------

