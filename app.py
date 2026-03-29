import json
import joblib
import os
import re
import pandas as pd
import streamlit as st

from cleaning_pipeline import run_cleaning_pipeline
from auto_ml import run_automl_model

#============================================================= Streamlit page config
st.set_page_config(page_title="A Generic AutoML Pipeline", layout="wide")

#======================================================= Helper functions for safe Streamlit display
def make_streamlit_safe_value(x):
    """
    Convert complex Python objects into strings so Streamlit
    can display them safely.
    """
    if pd.isna(x):
        return x

    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)

    if isinstance(x, list):
        return json.dumps(x, ensure_ascii=False, default=str)

    if isinstance(x, tuple):
        return json.dumps(list(x), ensure_ascii=False, default=str)

    if isinstance(x, set):
        return json.dumps(sorted(list(x)), ensure_ascii=False, default=str)

    return x


def make_streamlit_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns that contain lists/dicts/tuples/sets
    into string-safe values for Streamlit display.
    """
    safe_df = df.copy()

    for col in safe_df.columns:
        if safe_df[col].dtype == "object":
            has_complex_objects = safe_df[col].apply(
                lambda v: isinstance(v, (list, dict, tuple, set))
            ).any()

            if has_complex_objects:
                safe_df[col] = safe_df[col].apply(make_streamlit_safe_value)

    return safe_df

# ========================================================= App navigation and state management
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose your mode", ["ML Model Training", "Trained Model"])
st.sidebar.divider()

# ========================================================= Mode 1: ML Model Training
if app_mode == "ML Model Training":

 # App title and description
    st.title("A Generic AutoML Pipeline for Cleaning CSV Data")
    st.write(
    "Upload an uncleaned CSV file, Click 'Run Data Cleaning', preview the cleaned output, "
    "and download the result."
 )

 #============================================================= Sidebar options
    st.sidebar.header("Cleaning Options")
    normalize_headers = st.sidebar.checkbox("Normalize headers", value=True)
    trim_spaces = st.sidebar.checkbox("Clean text spaces", value=True)
    collapse_spaces = st.sidebar.checkbox("Collapse repeated spaces", value=True)
    infer_types = st.sidebar.checkbox("Infer column data types", value=True)
    remove_duplicates = st.sidebar.checkbox("Remove duplicate rows", value=True)
    parse_addresses = st.sidebar.checkbox("Clean address columns", value=True)
    parse_reviews = st.sidebar.checkbox("Clean review columns", value=True)
    st.sidebar.divider()
    st.sidebar.subheader("Missing Value Strategy")
    missing_strategy_choice = st.sidebar.radio(
        "options to handle missing data",
        options=[
        "Fill blanks", 
        "Drop blank rows", 
        "Ignore blanks" 
        ],
        index=0 # Defaults to the first option
    )

    # Mapping the UI text to our backend keywords
    strategy_map = {
        "Fill blanks": "fill",
        "Drop blank rows": "drop",
        "Ignore blanks": "ignore"
    }

    options = {
        "normalize_headers": normalize_headers,
        "trim_spaces": trim_spaces,
        "collapse_spaces": collapse_spaces,
        "infer_types": infer_types,
        "remove_duplicates": remove_duplicates,
        "parse_addresses": parse_addresses,
        "parse_reviews": parse_reviews,
        "missing_value_strategy": strategy_map[missing_strategy_choice],
    }

    #=============================================================== File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    #============================================================== Main app logic
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")

        # 1. Run the pipeline and save to session state
        if st.button("Run Data Cleaning"):
            with st.spinner("Cleaning your data..."):
                raw_df, cleaned_df, report, meta = run_cleaning_pipeline(uploaded_file, options)
            
                # Saving results to session state so they don't disappear
                st.session_state['cleaned'] = True
                st.session_state['raw_df'] = raw_df
                st.session_state['cleaned_df'] = cleaned_df
                st.session_state['report'] = report
                st.session_state['meta'] = meta

        # 2. Display everything if it's cleaned session state
        if st.session_state.get('cleaned', False):        
            # Retrieve the saved variables
            raw_df = st.session_state['raw_df']
            cleaned_df = st.session_state['cleaned_df']
            report = st.session_state['report']
            meta = st.session_state['meta']

            #================================================= File details
            st.subheader("File Details")
            meta_col1, meta_col2 = st.columns(2)
            meta_col1.metric("Encoding Used", meta.get("encoding_used", "Unknown"))
            meta_col2.metric("Delimiter Used", meta.get("delimiter_used", "Unknown"))

            #================================================= Summary report
            st.subheader("Cleaning Summary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Rows Before", report.get("rows_before", 0))
            c2.metric("Rows After", report.get("rows_after", 0))
            c3.metric("Duplicates Removed", report.get("duplicates_removed", 0))

            c4, c5, c6 = st.columns(3)
            c4.metric("Columns Before", report.get("columns_before", 0))
            c5.metric("Columns After", report.get("columns_after", 0))
            c6.metric("Missing Values After Cleaning", report.get("missing_after", 0))

            #============================================ Address parsing details
            if report.get("address_columns_parsed"):
                st.subheader("Address Cleaning")
                st.write("Parsed address columns:")
                st.write(report["address_columns_parsed"])

            #============================================== Review parsing details
            if report.get("review_columns_parsed"):
             st.subheader("Review Cleaning")
             st.write("Parsed review columns:")
             st.write(report["review_columns_parsed"])

            #=============================================== Preview tabs
            tab1, tab2, tab3 = st.tabs(
                ["Cleaned Data", "Original Data", "Column Data Types"]
            )

            with tab1:
                st.write("Preview of cleaned data:")
                safe_cleaned_preview = make_streamlit_safe_df(cleaned_df.head(5))
                st.dataframe(safe_cleaned_preview, use_container_width=True)

            with tab2:
                st.write("Preview of original uploaded data:")
                safe_raw_preview = make_streamlit_safe_df(raw_df.head(5))
                st.dataframe(safe_raw_preview, use_container_width=True)
            
            with tab3:
                st.write("Detected data types after cleaning:")
                dtype_rows = [
                    {"column": col, "dtype": dtype}
                    for col, dtype in report.get("type_summary", {}).items()
                ]
                dtype_df = pd.DataFrame(dtype_rows)
                st.dataframe(dtype_df, use_container_width=True)

            # ===================================== Show dataframe shape
            with st.expander("Show cleaned dataframe shape"):
                st.write(cleaned_df.shape)
             
            # Changed this to 'False' so it doesn't trigger automatically
            if st.session_state.get('cleaned', False):
                st.success("Cleaning completed successfully")

            #======================================== Safe CSV download
            download_df = make_streamlit_safe_df(cleaned_df)
            csv_data = download_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Cleaned CSV",
                data=csv_data,
                file_name="cleaned_output.csv",
                mime="text/csv"
            )

    else:
     st.info("Please upload a CSV file to begin.")

    #=============================================  User will select ML models to train
    if st.session_state.get('cleaned', False):
        st.divider()
        st.header("Machine Learning Model Training and Results Evaluation")
    
        # 1. FIXED FORM INDENTATION
        with st.form("automl_form"):
         col1, col2 = st.columns(2)
        
        with col1:
            # Target column selection
            target_column = st.selectbox("Select target column for modeling", st.session_state['cleaned_df'].columns)
            # 2. Task type selection
            task_type = st.radio("Select Machine Learning Task", ["Classification", "Regression"])
                
        with col2:
            # Model selection
            if task_type == "Classification":
                available_models = ["Logistic Regression", "Random Forest Classifier", "XGBoost Classifier", "SVC"]
            if task_type == "Regression":
                available_models = ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor", "Ridge Regression"]
            
            selected_models = st.multiselect("Select Models to Train:", available_models, default=available_models)
        
            # Submit button for the form (Must be inside the 'with st.form' block)
            start_training = st.form_submit_button("Train Models")

    # 2. EXECUTE TRAINING (Indented under the 'if cleaned' block so it doesn't crash on boot)
        if start_training:
            if len(selected_models) == 0:
                st.error("Please select at least one model to train.")
            else:
                with st.spinner(f"Training {len(selected_models)} models... This may take a minute."):
                 try:
                        # Run the AutoML function with the cleaned dataframe and user selections
                        results, best_model, plot_data = run_automl_model(
                        df=st.session_state['cleaned_df'], 
                        target_column=target_column,
                        task_type=task_type, 
                        selected_models=selected_models
                    )
                        st.success(f"Training Complete! Best Model: {results.iloc[0]['Model Name']}")
                        
                        # Display the Results DataFrame (Leaderboard)
                        st.subheader("ML Model Results")
                        st.dataframe(results, use_container_width=True)

                        # Save the best model to disk using joblib
                        file_name = uploaded_file.name.rsplit('.', 1)[0]
                        model_filename = f"{file_name}_best_model.joblib"
                        joblib.dump(best_model, model_filename)
                        st.success(f"Best model saved as: '{model_filename}'")

                        # This displays the visuals of the ML model
                        if "confusion_matrix_fig" in plot_data:
                            st.subheader("Model Diagnostics")
                            st.write("This matrix shows where the best model guessed correctly vs. where it got confused.")
                            st.plotly_chart(plot_data["confusion_matrix_fig"], use_container_width=True)
                        
                 except Exception as e:
                        st.error(f"Error during model training: {str(e)}")
                        
# ======================================================== Mode 2: Trained Model
elif app_mode == "Trained Model":
    st.title("Use a Trained Model for Predictions")
    st.write("This section will allow you to upload a trained model and use it for making predictions on CSV data.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Select the trained model")
        saved_models = [f for f in os.listdir() if f.endswith('.joblib')]
        if len(saved_models) == 0:
            st.warning("No trained models found. Please train a model in the 'ML Model Training' section first.")
            selected_model = None
        else:
            selected_model = st.selectbox("Select a trained model to use for predictions", saved_models)
    with col2:
        st.subheader("2. Upload CSV data for predictions")
        prediction_file = st.file_uploader("Upload a CSV file for making predictions", type=["csv"])
    if selected_model and prediction_file:
        st.success(f"loading model '{selected_model}' and making predictions on uploaded data...")
        if st.button("Run Predictions"):
            with st.spinner("Running predictions..."):
                try:
                    # Load the selected model using joblib
                    loadmodel = joblib.load(selected_model)

                    # Read the uploaded CSV file for predictions
                    input_df = pd.read_csv(prediction_file)

                    # Clean the headers of the input data to match the training data
                    input_df.columns = [re.sub(r"_+", "_", re.sub(r"[\s\-]+", "_", re.sub(r"[^\w\s-]", "", str(col).strip().lower()))).strip("_") for col in input_df.columns]

                    # Preprocess the input data if necessary (this depends on how your model was trained)
                    expected_columns = loadmodel.feature_names_in_
                    aligned_df = input_df[expected_columns]

                    # Make predictions using the loaded model
                    predictions = loadmodel.predict(aligned_df)

                    # Display the predictions
                    input_df['Predictions'] = predictions
                    st.write("Predictions added as a new column in the input data:")
                    st.success("Predictions Output")
                    st.dataframe(input_df.head(6), use_container_width=True)
                    final_csv = input_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Predictions CSV",
                        data=final_csv,
                        file_name="predictions_output.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.warning("Make sure the uploaded CSV for predictions has the same structure and columns as the data used to train the model.")