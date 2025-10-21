import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set tracking URI with error handling
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print("Connected to MLflow tracking server")
except Exception as e:
    print(f"Warning: Could not connect to MLflow server: {e}")


# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Convert to DataFrame for better visualization
feature_names = wine.feature_names
target_names = wine.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)

# Enhanced parameters
params = {
    'max_depth': 10,
    'n_estimators': 5,
    'random_state': 42,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

# Set experiment
mlflow.set_experiment('MLFLOW-Exp1')

with mlflow.start_run(run_name='RF_Wine_Classification'):
    # Log all parameters
    for param, value in params.items():
        mlflow.log_param(param, value)
    
    # Train model
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    
    # Calculate multiple metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log multiple metrics
    mlflow.log_metrics({
        'accuracy': accuracy,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    })

    # Create comprehensive visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names,
                ax=axes[0])
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    axes[0].set_title('Confusion Matrix')
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    feature_importance.plot(kind='barh', x='feature', y='importance', 
                           ax=axes[1], legend=False)
    axes[1].set_title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig("model_analysis.png", dpi=300, bbox_inches='tight')
    
    # Log the comprehensive plot
    mlflow.log_artifact("model_analysis.png")

    # Log classification report as text file
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("classification_report.csv")
    mlflow.log_artifact("classification_report.csv")
    
    # Log feature importance as CSV
    feature_importance.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")
    
    # Enhanced tags
    tags = {
        "Author": "Ayush",
        "Project": "Wine Classification", 
        "Dataset": "Wine",
        "Model_Type": "Random Forest",
        "Version": "1.0"
    }
    mlflow.set_tags(tags)
    
    # Model logging with signature
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, rf.predict(X_train))
    
    mlflow.sklearn.log_model(
        rf, 
        "random-forest-model",
        signature=signature,
        input_example=X_train[:5],
        registered_model_name="Wine_Classifier_RF"
    )
    
    print(f"Model training completed with accuracy: {accuracy:.4f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# Function to run multiple experiments with different parameters
def run_experiment_with_params(params, run_name):
    with mlflow.start_run(run_name=run_name):
        # Train and log model with given parameters
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters and metrics
        for param, value in params.items():
            mlflow.log_param(param, value)
        mlflow.log_metric('accuracy', accuracy)
        
        print(f"{run_name} - Accuracy: {accuracy:.4f}")

# Test different hyperparameters
param_sets = [
    ({'n_estimators': 5, 'max_depth': 5}, "RF_5_5"),
    ({'n_estimators': 10, 'max_depth': 10}, "RF_10_10"),
    ({'n_estimators': 20, 'max_depth': 15}, "RF_20_15"),
]

for params, name in param_sets:
    run_experiment_with_params(params, name)

# Function to run multiple experiments with different parameters
def run_experiment_with_params(params, run_name):
    with mlflow.start_run(run_name=run_name):
        # Train and log model with given parameters
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters and metrics
        for param, value in params.items():
            mlflow.log_param(param, value)
        mlflow.log_metric('accuracy', accuracy)
        
        print(f"{run_name} - Accuracy: {accuracy:.4f}")

# Test different hyperparameters
param_sets = [
    ({'n_estimators': 5, 'max_depth': 5}, "RF_5_5"),
    ({'n_estimators': 10, 'max_depth': 10}, "RF_10_10"),
    ({'n_estimators': 20, 'max_depth': 15}, "RF_20_15"),
]

for params, name in param_sets:
    run_experiment_with_params(params, name)
