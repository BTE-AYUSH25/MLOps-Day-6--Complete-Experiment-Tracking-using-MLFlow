# Import necessary libraries
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier  # Our machine learning model
from sklearn.model_selection import train_test_split  # To split data into train/test
from sklearn.datasets import load_breast_cancer  # The dataset we'll use
import pandas as pd  # For data manipulation
import mlflow  # MLOps platform for tracking experiments

# Set up MLflow tracking - this connects to the MLflow UI
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLflow server address

# Load the Breast Cancer dataset - a classic binary classification dataset
data = load_breast_cancer()
# Convert features to DataFrame for better handling and visualization
X = pd.DataFrame(data.data, columns=data.feature_names)
# Convert target to Series (0 = malignant, 1 = benign)
y = pd.Series(data.target, name='target')

# Print dataset info for understanding
print(f"Dataset shape: {X.shape}")
print(f"Feature names: {data.feature_names}")
print(f"Target distribution:\n{y.value_counts()}")

# Splitting into training and testing sets
# test_size=0.2 means 20% for testing, 80% for training
# random_state=42 ensures reproducible splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Creating the RandomForestClassifier model
# random_state=42 ensures reproducible results
rf = RandomForestClassifier(random_state=42)

# Defining the parameter grid for GridSearchCV
# We'll test different combinations of these parameters
param_grid = {
    'n_estimators': [10, 50, 100],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30]  # Maximum depth of each tree
}
# Total combinations: 3 n_estimators × 4 max_depth = 12 combinations

# Applying GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=rf,        # The model to tune
    param_grid=param_grid,  # Parameters to try
    cv=5,                # 5-fold cross-validation
    n_jobs=-1,           # Use all available CPU cores
    verbose=2,           # Print progress messages
    return_train_score=True  # Also track training scores
)

# Create a dedicated experiment in MLflow
mlflow.set_experiment('breast-cancer-rf-hyperparameter-tuning')

# Start the main (parent) run to track the entire GridSearch process
with mlflow.start_run(run_name="GridSearch_Parent_Run") as parent_run:
    
    # Fit GridSearchCV - this will train 12 models × 5 folds = 60 models total
    print("Starting GridSearchCV... This may take a moment.")
    grid_search.fit(X_train, y_train)
    print("GridSearchCV completed!")
    
    # Log all the individual hyperparameter combinations as nested (child) runs
    # This creates a hierarchy: Parent run → Multiple child runs
    print("Logging individual parameter combinations as child runs...")
    for i in range(len(grid_search.cv_results_['params'])):
        # Start a nested run for each parameter combination
        with mlflow.start_run(run_name=f"HP_Combination_{i+1}", nested=True) as child:
            # Log the parameters for this combination
            mlflow.log_params(grid_search.cv_results_["params"][i])
            # Log the cross-validation accuracy for this combination
            mlflow.log_metric("mean_cv_accuracy", grid_search.cv_results_["mean_test_score"][i])
            # Also log the standard deviation to understand score stability
            mlflow.log_metric("std_cv_accuracy", grid_search.cv_results_["std_test_score"][i])
    
    # Get the best performing parameters and score from GridSearch
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Log the best parameters and score in the parent run
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    # Log best parameters to parent run
    mlflow.log_params(best_params)
    
    # Log best score to parent run
    mlflow.log_metric("best_cv_accuracy", best_score)
    
    # Log dataset information for reproducibility
    # Training data
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_dataset = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_dataset, "training_dataset")
    
    # Test data
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_dataset = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_dataset, "testing_dataset")
    
    # Log the source code for complete reproducibility
    mlflow.log_artifact(__file__)
    
    # Log the best model (the one with optimal hyperparameters)
    mlflow.sklearn.log_model(
        grid_search.best_estimator_,  # The best performing model
        "best_random_forest_model",   # Name for the model in MLflow
        registered_model_name="Breast_Cancer_RF_Classifier"  # Register in model registry
    )
    
    # Set tags for better organization and filtering in MLflow UI
    mlflow.set_tags({
        "author": "Vikash Das",
        "project": "Breast Cancer Classification",
        "model_type": "Random Forest",
        "task": "Binary Classification",
        "dataset": "Breast Cancer Wisconsin",
        "tuning_method": "GridSearchCV"
    })
    
    # Additional logging: Feature importance from the best model
    feature_importance = pd.DataFrame({
        'feature': data.feature_names,
        'importance': grid_search.best_estimator_.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Log top 10 most important features
    top_features = feature_importance.head(10)
    print("\nTop 10 Most Important Features:")
    print(top_features.to_string(index=False))
    
    # Log feature importance as a metric (top feature importance)
    mlflow.log_metric("top_feature_importance", top_features['importance'].iloc[0])
    
    # Final results summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation Accuracy: {best_score:.4f}")
    print(f"Total Parameter Combinations Tested: {len(grid_search.cv_results_['params'])}")
    print(f"Parent Run ID: {parent_run.info.run_id}")
    print("Check MLflow UI at http://127.0.0.1:5000 to visualize results!")
    print("="*50)

# Print instructions for viewing results
print("\n" + "🔍 HOW TO VIEW YOUR RESULTS:")
print("1. Start MLflow server: mlflow ui --port 5000")
print("2. Open browser: http://localhost:5000")
print("3. Look for experiment: 'breast-cancer-rf-hyperparameter-tuning'")
print("4. Click on the parent run to see all child runs!")