import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


def load_data():
    """
    Load the three required CSV files into pandas DataFrames.
    """
    df_credits = pd.read_csv('data/User Credits Student Access.csv', encoding='utf-8')
    df_atlas = pd.read_csv('data/Atlas Cechu Student Access.csv', encoding='utf-8')
    df_payments = pd.read_csv('data/Payments Student Access.csv', encoding='utf-8')
    return df_credits, df_atlas, df_payments


def clean_and_merge(df_credits, df_atlas, df_payments):
    """
    Clean raw data and merge into a single DataFrame.
    """
    df_credits_cleaned = df_credits[df_credits['credits'] > 0]
    df_payments_cleaned = df_payments[df_payments['id'].notna()]
    df = pd.merge(df_payments_cleaned, df_atlas, how='inner', left_on='user', right_on='user_id')
    df.rename(columns={'credits_x': 'credits_payments', 'credits_y': 'credits_credits'}, inplace=True)
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    df = df.astype({col: int for col in df.select_dtypes(include='bool').columns})
    df = df[df['state'] != 'CANCELLED']
    return df


def enrich_datetime(df):
    """
    Extract date parts from 'created_at' and drop the original timestamp.
    """
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['day'] = df['created_at'].dt.day
    df['month'] = df['created_at'].dt.month
    df['year'] = df['created_at'].dt.year
    df['hour'] = df['created_at'].dt.hour
    df['weekday'] = df['created_at'].dt.weekday
    return df.drop(columns=['created_at'])


def prepare_features(df):
    """
    Drop unnecessary columns and bin 'credits' into categories.
    Returns X (features) and y (target).
    """
    df_modeling = df.drop(columns=['id', 'changed_at', 'user', 'batch', 'state', 'user_id'])
    df_modeling['credits_category'] = pd.cut(
        df_modeling['credits'],
        bins=[500, 600, 1000, df_modeling['credits'].max()],
        labels=['500-600', '601-1000', '1001+']
    )
    df_modeling = df_modeling[df_modeling['credits_category'].notnull()]
    df_modeling = df_modeling.drop(columns=['credits'])
    y = df_modeling['credits_category']
    X = df_modeling.drop(columns=['credits_category'])
    return X, y

def split_data(X, y, test_size=0.2, stratify=True, random_state=42):
    """
    Split data into training and testing sets with optional stratification.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        test_size (float): Proportion of data to use as test set.
        stratify (bool): Whether to stratify split based on y.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y if stratify else None,
        random_state=random_state
    )

def run_grid_search(X, y):
    """
    Run GridSearchCV with CatBoostClassifier and return the best model.
    """
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    param_grid = {
        'depth': list(range(1, 11, 2)),
        'learning_rate': np.array(range(5, 35, 5)) / 100,
        'loss_function': ['MultiClass', 'MultiClassOneVsAll'],
    }

    model = CatBoostClassifier(
        iterations=500,
        eval_metric='Accuracy',
        verbose=100
    )

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    return grid_search

def run_basic_catboost_model(X, y):
    """
    Train and evaluate a basic CatBoostClassifier using default settings.
    """
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        verbose=100
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    return model

