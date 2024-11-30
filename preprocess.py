import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{filepath}' is empty.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def preprocess_data(data):
    """
    Preprocess the dataset by handling missing values and encoding categorical features.
    
    Args:
        data (pd.DataFrame): The input dataset.
        
    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    # Check for missing values
    print("Checking for missing values...")
    missing_columns = data.columns[data.isnull().any()]
    if not missing_columns.empty:
        print(f"Columns with missing values: {missing_columns.tolist()}")
        # Fill missing numerical values with the mean
        for col in missing_columns:
            if data[col].dtype in ['float64', 'int64']:
                data[col].fillna(data[col].mean(), inplace=True)
            else:
                data[col].fillna('Unknown', inplace=True)
    else:
        print("No missing values found.")
    
    # Encode categorical features
    print("Encoding categorical features...")
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    return data

def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Args:
        data (pd.DataFrame): The input dataset.
        target_column (str): The name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("Splitting data into training and testing sets...")
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    FILEPATH = "data/kenya_rideshare_data.csv"  # Path to the dataset
    TARGET_COLUMN = "target"  # Replace with your actual target column name
    
    # Load the data
    dataset = load_data(FILEPATH)
    
    # Preprocess the data
    preprocessed_data = preprocess_data(dataset)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(preprocessed_data, TARGET_COLUMN)
