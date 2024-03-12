# Clean data
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

def separate_data(df, response):
    """
    Function to separate features and response variable

    Parameters
    ----------
    df: dataframe
        Dataframe containing true labels of groups (clusters)

    response: string
        String of column name of response variable
    Returns
    ----------
    X: df
        Dataframe containing features from dataset

    y: array-like, (n_samples,)
        Array containing the true labels for each data point
    """
    X = df.drop(response, axis=1)
    y = df[response]
    
    return(X, y)


def scale_data(df):
    """
    Function to scale numerical data

    Parameters
    ----------
    df: dataframe
        Dataframe containing true labels of groups (clusters)

    Returns
    ----------
    df_scaled: dataframe
        Dataframe containing scaled values of numeric variables
    """
    numeric_columns = df.select_dtypes(include=['float64', 'int']).columns
    categorical_columns = df.select_dtypes(exclude=['float64', 'int']).columns
    ct = ColumnTransformer([
        ('scale', StandardScaler(), numeric_columns)
    ], remainder='passthrough')

    # Fit and transform the data
    df_scaled_array = ct.fit_transform(df)
    
    # ColumnTransformer returns an array, convert it back to a DataFrame
    # Combine the column names for transformed and non-transformed columns
    all_columns = numeric_columns.tolist() + categorical_columns.tolist()
    df_scaled = pd.DataFrame(df_scaled_array, columns=all_columns, index=df.index)
    
    return df_scaled