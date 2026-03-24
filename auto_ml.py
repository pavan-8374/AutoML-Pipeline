# AutoML Pipeline for Machine Learning training and evaluation on cleaned data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, r2_score

# Machine Learning Models (Classification)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Machine Learning Models (Regression)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

# Main function to run the AutoML pipeline on the cleaned DataFrame.
def run_automl_model(df: pd.DataFrame, target_column: str, task_type: str, selected_models: list):
    """
    Main engine to dynamically preprocess data, train selected models, 
    and return a leaderboard of the results.
    """
    # 1. Separate Features (X) and Target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 2. Automatically detect column types for preprocessing
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Only grab text columns if they have fewer than 50 unique values
    # This prevents from exploding the memory
    categorical_features = []
    columns_to_drop = []
    
    raw_categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
    
    for col in raw_categorical_cols:
        if X[col].nunique() < 50:  # Safety limit: 50 unique categories maximum
            categorical_features.append(col)
        else:
            # If it has too many unique values. 
            # We must drop it so it doesn't crash the ML model.
            columns_to_drop.append(col)
            
    # Drop the dangerous columns from our Feature set
    if columns_to_drop:
        X = X.drop(columns=columns_to_drop)

    # 3. Create the Preprocessing Engines
    numeric_transformer = StandardScaler()
    # handle_unknown='ignore' prevents crashes if the test data has a category the model hasn't seen before
    categorical_transformer = OneHotEncoder(handle_unknown='ignore') 

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. Split the data into Training and Testing sets (80% train, 20% test)
    # We hide 20% of the data from the model so we can give it a fair test at the end
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Model Dictionary Mapping
    # Maps the text from Streamlit dropdown to the actual Scikit-Learn code objects
    model_dictionary = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(random_state=42),
        "SVC": SVC(),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Ridge Regression": Ridge()
    }
    
    # 6. Train and Evaluate Loop
    results = []
    best_model_object = None
    best_score = -float('inf')

    for model_name in selected_models:
        if model_name in model_dictionary:
            model = model_dictionary[model_name]
            
            # Bundle preprocessing and modeling in a Pipeline
            # This ensures no data leakage and makes deployment infinitely easier
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('model', model)])
            
            # Train the model!
            pipeline.fit(X_train, y_train)
            
            # Make predictions on the 20% hidden test set
            predictions = pipeline.predict(X_test)
            
            # Evaluate how well it did
            if task_type == "Classification":
                score = accuracy_score(y_test, predictions)
            else:
                score = r2_score(y_test, predictions)
                
            results.append({"Model Name": model_name, "Score": score})
            
            # Keep track of the best model to return
            if score > best_score:
                best_score = score
                best_model_object = pipeline

    # 7. Format the Leaderboard
    leaderboard_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    
    # Returning an empty dict for plot_data for now until we add charts later
    return leaderboard_df, best_model_object, {}