import pandas as pd

def load_and_clean_data(filepath):
    """
    Load and clean customer data
    """
    df = pd.read_csv(filepath)
    
    # Basic cleaning
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Convert categorical data if needed
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    return df

def preprocess_data(df, features):
    """
    Standardize selected features
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df