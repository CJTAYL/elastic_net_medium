# Data review
import pandas as pd

def get_info(df):
    """
    Function to retrieve information about a dataframe

    Parameters
    ----------
    df: dataframe
        Dataframe of features

    Returns
    ----------
    None
        Prints summry of dataframe
    """
    print('-------------------------')
    print('Dataframe Info')
    print('-------------------------')
    print(df.info())
    print()


def get_statistics(df):
    """
    Function to calculate descriptive statistics, skewness, and kurtosis of a dataframe

    Parameters
    ----------
    df: Dataframe

    Returns
    ----------
    full_stats: dataframe
        Dataframe of descriptive statistics + skewness + kurtosis numeric variables
    """
    desc_stats = df.describe()
    skewness_df = pd.DataFrame(df.skew(numeric_only=True), columns=['skew']).T
    kurtosis_df = pd.DataFrame(df.kurtosis(numeric_only=True), columns=['kurtosis']).T
    full_stats = pd.concat([desc_stats, skewness_df, kurtosis_df])
    print('-------------------------')
    print('Descriptive Statistics')
    print('-------------------------')
    with pd.option_context('display.max_columns', None, 'display.max_rows', None, 'display.expand_frame_repr', False):
        print(full_stats)
        print()
