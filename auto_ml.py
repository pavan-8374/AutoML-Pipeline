# AutoML Pipeline for Machine Learning training and evaluation on cleaned data

import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Main function to run the AutoML pipeline on the cleaned DataFrame.
def run_automl_model(df: pd.DataFrame, target_column: str, task_type: str, selected_models: list):
    """
    Train and evaluate AutoML models on the provided dataset.
    
    Args:
        df: Input DataFrame with features and target
        target_column: Name of the target column
        task_type: Type of task ('classification' or 'regression')
        selected_models: List of model names to train
    
    Returns:
        Dictionary containing trained models and evaluation results
    """
    
    # Step 1: Split data into train and test sets
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize results dictionary
    results = {
        "models": selected_models,
        "task_type": task_type,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }
    
    return json.loads(json.dumps(results))
