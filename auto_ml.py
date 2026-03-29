import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, r2_score, f1_score, confusion_matrix

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC

def run_automl_model(df: pd.DataFrame, target_column: str, task_type: str, selected_models: list):
    """
    Main engine with F1-Score and Interactive Confusion Matrix generation.
    """
    # 1. Separate Features and Target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert target to string if it's classification (helps with matrix labels)
    if task_type == "Classification":
        y = y.astype(str)

    # 2. This helps to automatically identify numeric and categorical features, and drop high-cardinality ones
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = []
    columns_to_drop = []
    
    raw_categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
    for col in raw_categorical_cols:
        if X[col].nunique() < 10:  
            categorical_features.append(col)
        else:
            columns_to_drop.append(col)
            
    if columns_to_drop:
        X = X.drop(columns=columns_to_drop)

    # 3. Preprocessor for numeric and categorical features
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore') 
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. Split Data into Train and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Model Dictionary with 'balanced' class weights for classification models to handle imbalanced datasets (like Promotions, Cancer, Fraud)
    model_dictionary = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'), #   This 'balanced' helps with imbalanced data
        "Random Forest Classifier": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "SVC": SVC(class_weight='balanced'),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Ridge Regression": Ridge()
    }
    
    # 6. Train and Evaluate Loop with F1-Score and Confusion Matrix for Classification, R2 Score for Regression
    results = []
    best_model_object = None
    best_score = -float('inf')
    best_predictions = None

    for model_name in selected_models:
        if model_name in model_dictionary:
            model = model_dictionary[model_name]
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            
            # New evaluation logic based on task type
            if task_type == "Classification":
                # F1 Score is better for imbalanced data (Promotions, Cancer, Fraud)
                score = f1_score(y_test, predictions, average='weighted')
                accuracy = accuracy_score(y_test, predictions)
                results.append({"Model Name": model_name, "F1 Score": round(score, 4), "Accuracy": round(accuracy, 4)})
            else:
                score = r2_score(y_test, predictions)
                results.append({"Model Name": model_name, "R2 Score": round(score, 4)})
            
            # Keep track of the best model
            if score > best_score:
                best_score = score
                best_model_object = pipeline
                best_predictions = predictions # Save predictions to build the chart

    # 7. Generate Plot Data for the Best Model
    plot_data = {}
    if task_type == "Classification" and best_predictions is not None:
        # Build the Confusion Matrix
        labels = sorted(list(y_test.unique()))
        cm = confusion_matrix(y_test, best_predictions, labels=labels)
        
        # Create a beautiful interactive Plotly Heatmap
        fig = px.imshow(cm, 
                        text_auto=True, 
                        labels=dict(x="Predicted by Model", y="Actual Truth", color="Count"),
                        x=labels, 
                        y=labels,
                        color_continuous_scale="Blues",
                        title="Confusion Matrix for Best Model")
        plot_data["confusion_matrix_fig"] = fig

    leaderboard_df = pd.DataFrame(results).sort_values(by=list(results[0].keys())[1], ascending=False)
    
    return leaderboard_df, best_model_object, plot_data