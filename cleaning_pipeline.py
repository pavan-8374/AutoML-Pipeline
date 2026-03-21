import ast
import csv
import io
import json
import re
from typing import Dict, Tuple
import pandas as pd

def detect_encoding(file_bytes: bytes) -> str:
    """
    Try a few common encodings and return the first one that works.
    """
    encodings_to_try = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    for enc in encodings_to_try:
        try:
            file_bytes.decode(enc)
            return enc
        except Exception:
            continue

    return "latin1"

#---------------------------------------------------- 1) .CSV file reading
def read_csv_robust(uploaded_file) -> Tuple[pd.DataFrame, Dict]:
    """
    Read CSV using safer parsing for messy quoted fields such as reviews_list.
    """
    file_bytes = uploaded_file.getvalue()
    encoding = detect_encoding(file_bytes)

    # Try reading directly from bytes buffer first
    uploaded_file.seek(0)

    try:
        df = pd.read_csv(
            uploaded_file,
            sep=",",
            quotechar='"',
            escapechar="\\",
            doublequote=True,
            engine="python",
            encoding=encoding,
            on_bad_lines="skip"
        )
    except Exception:
        # fallback using decoded text
        decoded_text = file_bytes.decode(encoding, errors="replace")

        df = pd.read_csv(
            io.StringIO(decoded_text),
            sep=",",
            quotechar='"',
            escapechar="\\",
            doublequote=True,
            engine="python",
            on_bad_lines="skip"
        )

    meta = {
        "encoding_used": encoding,
        "delimiter_used": ","
    }
    
    return df, meta

#------------------------------------------------ 2) Column cleaning
def normalize_column_name(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^\w\s-]", "", col)   # remove special chars
    col = re.sub(r"[\s\-]+", "_", col)   # spaces/hyphens -> underscore
    col = re.sub(r"_+", "_", col)        # multiple underscores -> one
    return col.strip("_")


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dataframe headers and avoid duplicate column names.
    """
    df = df.copy()
    new_cols = []
    seen = {}

    for col in df.columns:
        clean_col = normalize_column_name(col)

        if clean_col in seen:
            seen[clean_col] += 1
            clean_col = f"{clean_col}_{seen[clean_col]}"
        else:
            seen[clean_col] = 0

        new_cols.append(clean_col)

    df.columns = new_cols
    return df
#============================================= repair_mojibake_text()
def repair_mojibake_text(text):
    """
    Try to repair common mojibake / encoding corruption.
    """
    if pd.isna(text):
        return text

    text = str(text)

    try:
        repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        if repaired and repaired.count("Ã") < text.count("Ã"):
            text = repaired
    except Exception:
        pass
    return text

#================================================================ 3) Text cleaning for all object/text columns
def clean_text_value(x):
    if pd.isna(x):
        return x 
    x = str(x)
    x = repair_mojibake_text(x)
    x = re.sub(r"\s+", " ", x).strip()
    x = x.replace("?", "")
    return x


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean all object/text columns.
    """
    df = df.copy()
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].apply(clean_text_value)
    return df


#=================================================================== 4) Missing values
def standardize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert common missing-value tokens into pd.NA.
    """
    df = df.copy()
    missing_tokens = {"", "na", "n/a", "null", "none", "-", "--", "?", "nan"}

    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].apply(
            lambda x: pd.NA if isinstance(x, str) and x.strip().lower() in missing_tokens else x
        )

    return df


#================================================================== 5) Address cleaning 
def is_address_like_column(col_name: str) -> bool:
    """
    Detect whether a column likely contains addresses.
    """
    col_name = str(col_name).strip().lower()
    address_keywords = ["address", "addr", "location_address", "business_address", "customer_address"]
    return any(keyword in col_name for keyword in address_keywords)
 

def clean_address_text(x):
    """
    Basic cleanup for address strings.
    """
    if pd.isna(x):
        return x

    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    x = re.sub(r"\s*,\s*", ", ", x)
    x = re.sub(r"(,\s*)+", ", ", x)
    return x.strip(", ").strip()


def parse_single_address(address: str) -> Dict:
    """
    Parse an address from right to left into:
    street_info, locality, area, city
    """
    result = {
        "street_info": pd.NA,
        "locality": pd.NA,
        "area": pd.NA,
        "city": pd.NA
    }

    if pd.isna(address):
        return result

    address = clean_address_text(address)
    parts = [p.strip() for p in address.split(",") if p.strip()]

    if len(parts) >= 4:
        result["city"] = parts[-1]
        result["area"] = parts[-2]
        result["locality"] = parts[-3]
        result["street_info"] = ", ".join(parts[:-3])

    elif len(parts) == 3:
        result["city"] = parts[-1]
        result["area"] = parts[-2]
        result["street_info"] = parts[0]

    elif len(parts) == 2:
        result["city"] = parts[-1]
        result["street_info"] = parts[0]

    elif len(parts) == 1:
        result["street_info"] = parts[0]

    return result


def parse_address_column(df: pd.DataFrame, col: str, drop_original: bool = True) -> pd.DataFrame:
    """
    Parse one address column into structured columns.
    """
    df = df.copy()
    parsed = df[col].apply(parse_single_address).apply(pd.Series)

    df[f"{col}_street_info"] = parsed["street_info"]
    df[f"{col}_locality"] = parsed["locality"]
    df[f"{col}_area"] = parsed["area"]
    df[f"{col}_city"] = parsed["city"]

    if drop_original:
        df = df.drop(columns=[col], errors="ignore")

    return df


def clean_address_columns(df: pd.DataFrame, drop_original: bool = True) -> Tuple[pd.DataFrame, list]:
    """
    Detect address-like columns, parse them, and optionally drop the original.
    """
    df = df.copy()
    parsed_columns = []

    for col in list(df.columns):
        if is_address_like_column(col):
            df[col] = df[col].apply(clean_address_text)
            df = parse_address_column(df, col, drop_original=drop_original)
            parsed_columns.append(col)

    return df, parsed_columns


# 6)============================================================= Review cleaning into rating columns
def is_review_like_column(col_name: str) -> bool:
    """
    Detect whether a column likely contains reviews.
    """
    col_name = str(col_name).strip().lower()
    review_keywords = ["review", "reviews", "customer_review", "rating_review", "feedback", "comments"]
    return any(keyword in col_name for keyword in review_keywords)


def clean_review_text(text):
    """
    Clean a single review text string.
    """
    if pd.isna(text):
        return pd.NA

    text = str(text)
    text = text.replace("RATED", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("?", "")

    return text if text else pd.NA


def extract_numeric_rating(rating_text):
    """
    Extract numeric rating from text like 'Rated 1.0'
    """
    if pd.isna(rating_text):
        return None

    rating_text = str(rating_text)
    match = re.search(r"(\d+(?:\.\d+)?)", rating_text)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def parse_reviews_to_rating_columns(cell) -> Dict:
    """
    Convert one reviews cell into rating-based review columns.
    """
    result = {
        "rating_1_0": [],
        "rating_2_0": [],
        "rating_3_0": [],
        "rating_4_0": [],
        "rating_5_0": [],
    }

    if pd.isna(cell):
        return {k: pd.NA for k in result}

    parsed_obj = cell

    # Try to parse stringified list/tuple
    if isinstance(cell, str):
        text = cell.strip()
        try:
            parsed_obj = ast.literal_eval(text)
        except Exception:
            # Plain text review with no rating structure -> cannot assign to rating column
            return {k: pd.NA for k in result}

    if isinstance(parsed_obj, (list, tuple)):
        for item in parsed_obj:
            # Tuple/list like  
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                rating_text = item[0]
                review_text = item[1]

                rating_val = extract_numeric_rating(rating_text)
                cleaned_review = clean_review_text(review_text)

                if rating_val is not None and pd.notna(cleaned_review):
                    if rating_val == 1.0:
                        result["rating_1_0"].append(cleaned_review)
                    elif rating_val == 2.0:
                        result["rating_2_0"].append(cleaned_review)
                    elif rating_val == 3.0:
                        result["rating_3_0"].append(cleaned_review)
                    elif rating_val == 4.0:
                        result["rating_4_0"].append(cleaned_review)
                    elif rating_val == 5.0:
                        result["rating_5_0"].append(cleaned_review)

    final_result = {}
    for key, values in result.items():
        if values:
            final_result[key] = " || ".join(values)
        else:
            final_result[key] = pd.NA

    return final_result


def parse_review_columns(df: pd.DataFrame, drop_original: bool = True) -> Tuple[pd.DataFrame, list]:
    """
    Detect review-like columns and create rating-specific review columns.
    Example:
        reviews_rating_1_0
        reviews_rating_2_0
        reviews_rating_3_0
        reviews_rating_4_0
        reviews_rating_5_0
    """
    df = df.copy()
    parsed_columns = []

    for col in list(df.columns):
        if is_review_like_column(col):
            parsed = df[col].apply(parse_reviews_to_rating_columns).apply(pd.Series)

            df[f"{col}_rating_1_0"] = parsed["rating_1_0"]
            df[f"{col}_rating_2_0"] = parsed["rating_2_0"]
            df[f"{col}_rating_3_0"] = parsed["rating_3_0"]
            df[f"{col}_rating_4_0"] = parsed["rating_4_0"]
            df[f"{col}_rating_5_0"] = parsed["rating_5_0"]

            if drop_original:
                df = df.drop(columns=[col], errors="ignore")

            parsed_columns.append(col)

    return df, parsed_columns


#===================================================================== 7) Duplicate removal
def make_hashable_for_dedup(x):
    """
    Convert unhashable objects into hashable string representations
    for safe duplicate detection.
    """
    if pd.isna(x):
        return x

    if isinstance(x, dict):
        return json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)

    if isinstance(x, list):
        return json.dumps(x, ensure_ascii=False, default=str)

    if isinstance(x, tuple):
        return json.dumps(list(x), ensure_ascii=False, default=str)

    if isinstance(x, set):
        return json.dumps(sorted(list(x)), ensure_ascii=False, default=str)

    return x


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows safely even if dataframe contains unhashable objects.
    """
    before = len(df)

    temp_df = df.copy()
    for col in temp_df.columns:
        temp_df[col] = temp_df[col].apply(make_hashable_for_dedup)

    duplicate_mask = temp_df.duplicated()
    df = df.loc[~duplicate_mask].copy()

    removed = before - len(df)
    return df, removed


#=============================================================== 8) Type inference
TRUE_VALUES = {"true", "yes", "y", "1", "t"}
FALSE_VALUES = {"false", "no", "n", "0", "f"}


def try_parse_bool(series: pd.Series) -> pd.Series:
    def convert(x):
        if pd.isna(x):
            return x
        val = str(x).strip().lower()
        if val in TRUE_VALUES:
            return True
        if val in FALSE_VALUES:
            return False
        return x

    converted = series.apply(convert)
    bool_ratio = converted.apply(lambda x: isinstance(x, bool) or pd.isna(x)).mean()

    if bool_ratio > 0.8:
        return converted
    return series


def try_parse_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[,%$₹€£]", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )
    converted = pd.to_numeric(cleaned, errors="coerce")
    ratio = converted.notna().mean()

    if ratio > 0.7:
        return converted
    return series


def try_parse_datetime(series: pd.Series) -> pd.Series:
    converted = pd.to_datetime(series, errors="coerce", dayfirst=True)
    ratio = converted.notna().mean()

    if ratio > 0.7:
        return converted
    return series


def infer_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer simple column types (bool, numeric, datetime).
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            original = df[col]

            maybe_bool = try_parse_bool(original)
            if not maybe_bool.equals(original):
                df[col] = maybe_bool
                continue

            maybe_num = try_parse_numeric(original)
            if not maybe_num.equals(original):
                df[col] = maybe_num
                continue

            maybe_dt = try_parse_datetime(original)
            if not maybe_dt.equals(original):
                df[col] = maybe_dt
                continue

    return df


#=========================================================== 9) Fill missing values
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
    - numeric -> median
    - bool -> mode
    - object/text -> 'Unknown'
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        mode_val = df[col].mode(dropna=True)
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])

    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].fillna("Unknown")

    return df

#============================================================== 10) Main cleaning pipeline
def clean_dataframe(df: pd.DataFrame, options: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Main cleaning pipeline.
    """
    report = {
        "rows_before": len(df),
        "columns_before": df.shape[1],
        "missing_before": int(df.isna().sum().sum()),
        "duplicates_removed": 0,
        "type_summary": {},
        "address_columns_parsed": [],
        "review_columns_parsed": []
    }

    # 1) Normalize headers
    if options.get("normalize_headers", True):
        df = normalize_headers(df)

    # 2) Clean text columns
    if options.get("trim_spaces", True) or options.get("collapse_spaces", True):
        df = clean_text_columns(df)

    # 3) Standardize missing tokens
    df = standardize_missing_values(df)

    # 4) Parse address columns and drop originals
    if options.get("parse_addresses", True):
        df, parsed_address_cols = clean_address_columns(df, drop_original=True)
        report["address_columns_parsed"] = parsed_address_cols

    # 5) Parse review columns into rating columns and drop originals
    if options.get("parse_reviews", True):
        df, parsed_review_cols = parse_review_columns(df, drop_original=True)
        report["review_columns_parsed"] = parsed_review_cols

    # 6) Remove duplicates
    if options.get("remove_duplicates", True):
        df, removed = remove_duplicates(df)
        report["duplicates_removed"] = removed

    # 7) Infer types
    if options.get("infer_types", True):
        df = infer_types(df)

    # 8) Fill missing values
    if options.get("fill_missing", True):
        df = fill_missing_values(df)

    report["rows_after"] = len(df)
    report["columns_after"] = df.shape[1]
    report["missing_after"] = int(df.isna().sum().sum())
    report["type_summary"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return df, report


#============================================================== 11) Wrapper function
def run_cleaning_pipeline(uploaded_file, options: Dict):
    """
    Complete wrapper:
    - read raw CSV
    - clean dataframe
    - return raw df, cleaned df, report, and metadata
    """
    raw_df, meta = read_csv_robust(uploaded_file)
    cleaned_df, report = clean_dataframe(raw_df, options)
    return raw_df, cleaned_df, report, meta