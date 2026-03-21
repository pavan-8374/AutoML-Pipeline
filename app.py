import json
import pandas as pd
import streamlit as st

from cleaning_pipeline import run_cleaning_pipeline


# Streamlit page config
st.set_page_config(page_title="A Generic AutoML Pipeline", layout="wide")


# Helper functions for safe Streamlit display
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

# ========================================================= App title and description
st.title("A Generic AutoML Pipeline for Cleaning Unstructured CSV Data")
st.write(
    "Upload an unstructured CSV file, run automatic cleaning, preview the cleaned output, "
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
        "Fill with estimates (Median/Mode)", 
        "Drop rows with missing data", 
        "Ignore them blank"
    ],
    index=0 # Defaults to the first option
)

# Map the human-readable UI text to our backend keywords
strategy_map = {
    "Fill with estimates (Median/Mode)": "fill",
    "Drop rows with missing data": "drop",
    "Ignore them blank": "ignore"
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
            
            # Save results to session state so they don't disappear
            st.session_state['cleaned'] = True
            st.session_state['raw_df'] = raw_df
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['report'] = report
            st.session_state['meta'] = meta

    # 2. Display everything IF it's in the session state
    if st.session_state.get('cleaned', False):
        st.success("Cleaning completed successfully")
        
        # Retrieve the saved variables
        raw_df = st.session_state['raw_df']
        cleaned_df = st.session_state['cleaned_df']
        report = st.session_state['report']
        meta = st.session_state['meta']

        #================================================= File metadata
        st.subheader("Detected File Metadata")
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
            st.subheader("Address Parsing")
            st.write("Parsed address columns:")
            st.write(report["address_columns_parsed"])

        #============================================== Review parsing details
        if report.get("review_columns_parsed"):
            st.subheader("Review Parsing")
            st.write("Parsed review columns:")
            st.write(report["review_columns_parsed"])

        #=============================================== Preview tabs
        tab1, tab2, tab3 = st.tabs(
            ["Original Data", "Cleaned Data", "Column Data Types"]
        )

        with tab1:
            st.write("Preview of original uploaded data:")
            safe_raw_preview = make_streamlit_safe_df(raw_df.head(20))
            st.dataframe(safe_raw_preview, use_container_width=True)

        with tab2:
            st.write("Preview of cleaned data:")
            safe_cleaned_preview = make_streamlit_safe_df(cleaned_df.head(20))
            st.dataframe(safe_cleaned_preview, use_container_width=True)

        with tab3:
            st.write("Detected data types after cleaning:")
            dtype_rows = [
                {"column": col, "dtype": dtype}
                for col, dtype in report.get("type_summary", {}).items()
            ]
            dtype_df = pd.DataFrame(dtype_rows)
            st.dataframe(dtype_df, use_container_width=True)

        # ===================================== Optional: show dataframe shape
        with st.expander("Show cleaned dataframe shape"):
            st.write(cleaned_df.shape)

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