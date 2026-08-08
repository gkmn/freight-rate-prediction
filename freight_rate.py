import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.stats import randint, uniform


# PREPROCESSING FUNCTION
# Ensures Train, Validation, and December data go through the EXACT same steps

train_df = pd.read_csv("data/train-test.csv") 

# Calculating median values
cols_to_median = ['delivery_lat', 'pickup_lat', 'delivery_lon', 'pickup_lon', 'market_index', 'quote_signal', 'distance', 'weight']
train_medians = {col: train_df[col].median() for col in cols_to_median if col in train_df.columns}

def prepare_features(df_input, reference_columns=None, medians=None):
    df = df_input.copy()
    
    required_cols = [
        "pickup_lat", "pickup_lon",
        "delivery_lat", "delivery_lon",
        "market_index", "quote_signal"
    ]

    if medians is None:
        medians = {}
            
    for col in required_cols:
        if col not in df.columns or df[col].isna().all():
            default_val = medians.get(col, 0.0)
            df[col] = default_val
            df[col + "_missing"] = 1
        else:
            missing_mask = df[col].isna()
            df[col] = df[col].fillna(medians.get(col, df[col].median()))
            df[col + "_missing"] = missing_mask.astype(int)
    
    df["weight"] = df["weight"].abs() #absolute of weight
    
    #New features
    
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    
    df["delta_lat"] = df["delivery_lat"] - df["pickup_lat"]
    df["delta_lon"] = df["delivery_lon"] - df["pickup_lon"]
    
    df["ton_miles"] = df["weight"] * df["distance"]
    df["market_pressure"] = df["market_index"] * df["quote_signal"]
    df["is_weekend"] = df["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
    
    df["equipment"] = df["equipment"].astype(str).str.strip()
    df = pd.get_dummies(df, columns=['equipment'], drop_first=True, dtype=int)
    
    X = df.drop(columns=["load_id", "pickup", "delivery", "date", "posted_rate"], errors='ignore')
    
    # To prevent schema mismatches (Fills missing dummy/new variables with 0)
    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0)
        
    return X


# TRAINING DATA AND PIPELINE SETUP
# NOTE: Update the file paths according to your directory structure (e.g., "data/train-test.csv")

y = train_df["posted_rate"]
X = prepare_features(train_df, reference_columns=None, medians=train_medians)

num_features = [
    'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 
    'distance', 'weight', 'market_index', 'quote_signal', 
    'delta_lat', 'delta_lon', 'ton_miles', 'market_pressure', 'is_weekend'
]

num_pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("imputer", KNNImputer(n_neighbors=5))
])

preprocessor = ColumnTransformer(
    transformers=[('num', num_pipeline, num_features)],
    remainder="passthrough"
)


# DATA ANALYSIS & OPTIMIZATION
# Reviewers can remove the '"""' quotes to see how the code was structured.
# These sections are commented out for fast execution 
# I used this code section to determine what model should i chose.


# CORRELATION MATRIX (EDA)
plt.figure(figsize=(20, 16))
numeric_df = train_df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(method='spearman'), annot=True, fmt=".2f", linewidths=0.5, cmap="coolwarm")
plt.title("Spearman Correlation Matrix")
plt.show()

"""
# MODEL COMPARISON (CROSS-VALIDATION)
tscv = TimeSeriesSplit(n_splits=5)
models = {
    "Ridge": Ridge(),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "CatBoost": CatBoostRegressor(random_state=42, verbose=0, thread_count=-1)
}
results, names = [], []
for name, model in models.items():
    full_pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    cv_scores = cross_val_score(full_pipeline, X[X["month"] <= 8], y[X["month"] <= 8], cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    results.append(np.abs(cv_scores))
    names.append(name)
plt.boxplot(results, labels=names, patch_artist=True)
plt.title("Time-Based CV Model Comparison")
plt.ylabel("Mean Absolute Error (MAE)")
plt.show()


# HYPERPARAMETER OPTIMIZATION (RANDOMIZED SEARCH)
cat_param_dist = {'model__iterations': randint(100, 600), 'model__learning_rate': uniform(0.01, 0.25), 'model__depth': randint(3, 10)}
cat_pipeline = Pipeline([('preprocessor', preprocessor), ('model', CatBoostRegressor(random_state=42, verbose=0, thread_count=-1))])
cat_search = RandomizedSearchCV(estimator=cat_pipeline, param_distributions=cat_param_dist, n_iter=30, cv=tscv, scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
cat_search.fit(X[X["month"] <= 8], y[X["month"] <= 8])
print("Best Params:", cat_search.best_params_)
"""


# FINAL MODEL & TRAINING
# I used the parameters found during hyperparameter optimization thanks to randomized search.

final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', CatBoostRegressor(
        depth=6,
        iterations=363,
        learning_rate=0.0185971302788046,
        random_state=42,
        verbose=0,
        thread_count=-1
    ))
])

print("Training the final model on the entire dataset")
final_model.fit(X, y)
print("Training completed")


# FEATURE IMPORTANCE
# Remove the '"""' quotes around the block below to view the importance chart.

"""
cat_model = final_model.named_steps['model']
passthrough_cols = [col for col in X.columns if col not in num_features]
all_feature_names = num_features + passthrough_cols

importances = cat_model.get_feature_importance()
indices = np.argsort(importances)[::-1]
sorted_features = [all_feature_names[i] for i in indices]
sorted_importances = importances[indices]

plt.figure(figsize=(12, 8))
plt.barh(range(len(indices)), sorted_importances[::-1], align='center', color='skyblue', edgecolor='black')
plt.yticks(range(len(indices)), sorted_features[::-1])
plt.xlabel('Relative Importance (%)')
plt.title('CatBoost Feature Importance (What drives the price?)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
"""

# GENERATING & SAVING PREDICTIONS
# Validation Predictions (12,000 Loads)
val_df = pd.read_csv("data/validation.csv") 
val_X = prepare_features(val_df, reference_columns=X.columns, medians=train_medians)

print("Generating validation predictions")
val_predictions = pd.DataFrame({
    "load_id": val_df["load_id"], 
    "predicted_rate": final_model.predict(val_X)
})
val_predictions.to_csv("validation_predictions.csv", index=False)
print("-> 'validation_predictions.csv' successfully saved.")

# December Predictions for score.py
dec_df = pd.read_csv("data/december-chart-inputs.csv") 
dec_X = prepare_features(dec_df, reference_columns=X.columns, medians=train_medians)

print("Generating December chart predictions")
dec_output = dec_df.copy()
dec_output["predicted_rate"] = final_model.predict(dec_X)

# Exact order expected by the score.py file
dec_output = dec_output[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]]
dec_output.to_csv("december_predictions.csv", index=False)
print("-> 'december_predictions.csv' successfully saved.")

print("\nALL PROCESSES COMPLETED")

