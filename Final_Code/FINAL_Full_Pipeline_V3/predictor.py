# predictor.py
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report
)



student_trips_filtered = pd.read_csv("training_data.csv")

# List of CENSUS_D columns
census_d_cols = {
    'New England':1,
    'Middle Atlantic':2,
    'East North Central':3,
    'West North Central':4,
    'South Atlantic':5,
    'East South Central':6,
    'West South Central':7,
    'Mountain':8,
    'Pacific':9
}

# Dictionary to store mini dataframes
mini_dfs = {}

# Iterate over each Census D column
for col in census_d_cols:
    # Select only rows where this category is 1
    subset = student_trips_filtered[student_trips_filtered["CENSUS_D"] == census_d_cols[col]].copy()

    # Drop all Census D columns
    subset = subset.drop(columns="CENSUS_D")

    # Store in dictionary
    mini_dfs[col] = subset
    print(subset.shape)

# Now mini_dfs contains a separate DataFrame for each Census D category

# Assume mini_dfs is the dictionary from your previous step
target_col = 'NEEDS_SCHOOL_BUS'

# Dictionary to store trained Random Forests
rf_models = {}

# Dictionary to store metrics
rf_metrics = {}

for census_cat, df in mini_dfs.items():
    print(f"\n=== Processing {census_cat} ===")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-test split 80-20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=64, stratify=y
    )

    percent_ones = (y_train == 1).mean() * 100
    print(f"{percent_ones:.2f}% of training rows have col_name = 1")

    percent_ones = (y_test == 1).mean() * 100
    print(f"{percent_ones:.2f}% of testing rows have col_name = 1")



    # Train RF
    rf = RandomForestClassifier(random_state=95)
    rf.fit(X_train, y_train)

    # Store model
    rf_models[census_cat] = rf

    # Predictions
    y_prob = rf.predict_proba(X_test)[:, 1]  # for ROC-AUC
    y_pred = (y_prob >= 0.40).astype(int)   # apply threshold of 0.40

    train_prob = rf.predict_proba(X_train)[:, 1]
    print("Train ROC-AUC:", roc_auc_score(y_train, train_prob))
    print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))


    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_0': precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        'precision_1': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'recall_0': recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        'recall_1': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        'f1_0': f1_score(y_test, y_pred, pos_label=0, zero_division=0),
        'f1_1': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        "y_pred_array": y_pred,
        "y_test_array": y_test.values,
        "y_prob_array": y_prob,
        "y_train_array": y_train.values,
        "y_train_prob_array": rf.predict_proba(X_train)[:, 1]
    }

    rf_metrics[census_cat] = metrics

    # Display metrics
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Precision: 0={metrics['precision_0']:.3f}, 1={metrics['precision_1']:.3f}")
    print(f"Recall: 0={metrics['recall_0']:.3f}, 1={metrics['recall_1']:.3f}")
    print(f"F1-score: 0={metrics['f1_0']:.3f}, 1={metrics['f1_1']:.3f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])



print(f"Finished training {len(rf_models)} Random Forest models.")

#individual
def predict_bus_need(vector):
  df_row = pd.DataFrame([vector])

  census_d_value = df_row['CENSUS_D'].iloc[0]

  if census_d_value not in rf_models:
      raise ValueError(f"No RF model found for census_d = {census_d_value}")

  model = rf_models[census_d_value]

  trained_columns = model.feature_names_in_

  for col in trained_columns:
      if col not in df_row.columns:
          df_row[col] = 0  # fill missing columns with 0
          print("filled in column", col)
  df_row = df_row[trained_columns]
  df_row.columns

  pred_class = model.predict(df_row)[0]
  pred_prob = model.predict_proba(df_row)[0, 1]  # probability of class 1

  return pred_class, pred_prob

#map
def predict_bus_need_matrix(matrix, rf_models, division_col="CENSUS_D"):
    """
    matrix: full DataFrame (contains additional non-feature columns)
    rf_models: dict of trained RF models keyed by census division ("New England", "Pacific", etc.)
    feature_cols: list of columns used by the RF model
    division_col: column name in matrix indicating census division (default "division")

    Returns:
        matrix with two added columns:
            - 'rf_prob'
            - 'rf_pred'
    """

    # --- 1. Validate division column ---
    if division_col not in matrix.columns:
        raise ValueError(f"Matrix must contain a '{division_col}' column.")

    # ALL rows in matrix belong to a single census division (your design)
    unique_divisions = matrix[division_col].dropna().unique()

    if len(unique_divisions) != 1:
        raise ValueError(f"Expected exactly 1 census division in matrix, found: {unique_divisions}")

    division = unique_divisions[0]

    # --- 2. Pick the correct pre-trained model ---
    if division not in rf_models:
        raise ValueError(f"No trained RF model found for division '{division}'")

    rf = rf_models[division]
    feature_cols = mini_dfs[division].drop(columns=["NEEDS_SCHOOL_BUS"]).columns
    

    # --- 3. Filter matrix to only the model’s feature columns ---
    missing = [c for c in feature_cols if c not in matrix.columns]
    if missing:
        raise ValueError(f"Matrix is missing required RF feature columns: {missing}")

    X = matrix[feature_cols].copy()

    # --- 4. Run the model ---
    probs = rf.predict_proba(X)[:, 1]
    preds = (probs >= 0.40).astype(int)

    # --- 5. Attach predictions back to the ORIGINAL matrix ---
    matrix = matrix.copy()
    matrix["rf_prob"] = probs
    matrix["rf_pred"] = preds

    return matrix



