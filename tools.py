from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from langchain_core.tools import tool
from registry import DATASET_REGISTRY, MODEL_REGISTRY

@tool

def rem_null_duplicates(dataset_id: str) -> dict:
    """Remove null values and duplicate rows from dataset."""
    df = DATASET_REGISTRY[dataset_id]
    before_shape = df.shape
    df = df.dropna().drop_duplicates()
    DATASET_REGISTRY[dataset_id] = df
    return {
        "before_rows": before_shape[0],
        "after_rows": df.shape[0],
        "columns": df.columns.tolist()
    }

@tool

def data_profile_tool(dataset_id: str) -> dict:
    """Generate dataset profile summary."""
    df = DATASET_REGISTRY[dataset_id]
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "description": df.describe(include='number').round(3)
    }

@tool

def correlation_tool(dataset_id: str, threshold: float = 0.7) -> dict:
    """Find strong correlations above a threshold."""
    df = DATASET_REGISTRY[dataset_id]
    corr = df.corr(numeric_only=True)
    strong_corr = (
        corr.where(abs(corr) >= threshold)
            .stack()
            .to_dict()
    )
    return {"strong_correlations": strong_corr}


@tool

def kpi_summary_tool(dataset_id: str) -> dict:
    """Generate high-level KPIs for the dataset."""
    
    df = DATASET_REGISTRY[dataset_id]
    numeric_cols = df.select_dtypes(include='number')
    return {
        "row_count": len(df),
        "column_count": df.shape[1],
        "numeric_columns": numeric_cols.columns.tolist(),
        "mean_values": numeric_cols.mean().round(3).to_dict()
    }

@tool
def encode_categorical_tool(dataset_id: str) -> dict:
    """Label-encode all categorical columns."""
    
    df = DATASET_REGISTRY[dataset_id].copy()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    DATASET_REGISTRY[dataset_id] = df
    
    return {
        "encoded_columns": cat_cols,
        "encoding_maps": encoders
    }

@tool
def preprocess_dates_tool(dataset_id: str) -> dict:
    """Convert date columns to numeric format."""
    
    df = DATASET_REGISTRY[dataset_id].copy()
    
    date_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_datetime(df[col], dayfirst=True)  
            
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            
            df = df.drop(columns=[col])
            date_cols.append(col)
        except:
            pass
    
    DATASET_REGISTRY[dataset_id] = df
    
    return {
        "processed_date_columns": date_cols
    }

@tool
def groupby_summary_tool(dataset_id: str, group_col: str, agg: str = "mean") -> dict:
    """Perform group-by aggregation on numeric columns."""
    
    df = DATASET_REGISTRY[dataset_id]
    
    summary = df.groupby(group_col).agg(agg).round(3)
    
    return summary.to_dict()

@tool
def outlier_detection_tool(dataset_id: str) -> dict:
    """Detect outliers using IQR method for numeric columns."""
    
    df = DATASET_REGISTRY[dataset_id]
    numeric_cols = df.select_dtypes(include='number')
    
    outlier_report = {}
    
    for col in numeric_cols.columns:
        Q1 = numeric_cols[col].quantile(0.25)
        Q3 = numeric_cols[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = ((numeric_cols[col] < lower) | (numeric_cols[col] > upper)).sum()
        
        outlier_report[col] = int(outliers)
    
    return outlier_report

@tool
def plot_distribution_tool(dataset_id: str, column: str) -> dict:
    """Plot distribution of a column."""
    
    df = DATASET_REGISTRY[dataset_id]
    
    plt.figure()
    df[column].hist()
    path = f"{dataset_id}_{column}_dist.png"
    plt.savefig(path)
    plt.close()
    
    return {"plot_path": path}

@tool
def plot_correlation_heatmap_tool(dataset_id: str) -> dict:
    """Plot correlation heatmap."""
    
    df = DATASET_REGISTRY[dataset_id]
    corr = df.corr(numeric_only=True)
    
    plt.figure(figsize=(8,6))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
    path = f"{dataset_id}_corr_heatmap.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
    return {"plot_path": path}

@tool
def prediction(dataset_id: str, target: str, model_name: str) -> dict:

    """Trains the model and predicts the value"""

    df = DATASET_REGISTRY[dataset_id]

    X = df.drop(columns=[target])
    y = df[target]

    model = MODEL_REGISTRY[model_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return {
        "prediction_sample": preds[:5].tolist(),
        "actual_sample": y_test[:5].tolist()
    }