import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DataPreparation:
    def __init__(self):
        pass

    def read_data(self):
        try:
            df_credits = pd.read_csv('../data/User Credits Student Access.csv', encoding='utf-8')
            df_atlas = pd.read_csv('../data/Atlas Cechu Student Access.csv', encoding='utf-8')
            df_payments = pd.read_csv('../data/Payments Student Access.csv', encoding='utf-8')
        except:
            df_credits = pd.read_csv('./data/User Credits Student Access.csv', encoding='utf-8')
            df_atlas = pd.read_csv('./data/Atlas Cechu Student Access.csv', encoding='utf-8')
            df_payments = pd.read_csv('./data/Payments Student Access.csv', encoding='utf-8')            
        return df_credits, df_atlas, df_payments

    def data_cleaning(self, df_credits, df_payments):
        df_c_negative = df_credits[df_credits['credits']<0]
        df_credits[df_credits.user.isin(df_c_negative.user) == True]

        # IDK what I'm doing with payments
        df_payments[df_payments.user.isin(df_c_negative.user)==True].sort_values(['user','created_at'])
        df_payments.state.unique()
        df_payments[df_payments.user=='STUD54678']
        df_credits[df_credits.user=='STUD54678']

        df_credits_cleaned = df_credits[df_credits['credits']>0]
        df_payments_cleaned = df_payments[df_payments['user'].notna()] # we threw out from payments 2345 observations

        return df_credits_cleaned, df_payments_cleaned

    def get_merged_table(self):
        df_credits, df_atlas, df_payments = self.read_data()
        _, df_payments_cleaned = self.data_cleaning(df_credits, df_payments)
        df_merge_full = pd.merge(df_payments_cleaned, df_atlas, how='inner', left_on='user', right_on='user_id')
        df_merge_full.rename(columns={'credits_x':'credits_payments', 'credits_y':'credits_credits'})
        df_merge_full = pd.get_dummies(df_merge_full, columns=['type'], drop_first=True)
        df_merge_full = df_merge_full.astype({col: int for col in df_merge_full.select_dtypes(include='bool').columns})

        df_merge_full['created_at'] = pd.to_datetime(df_merge_full['created_at'])

        df_merge_full['day'] = df_merge_full['created_at'].dt.day
        df_merge_full['month'] = df_merge_full['created_at'].dt.month
        df_merge_full['year'] = df_merge_full['created_at'].dt.year
        df_merge_full['hour'] = df_merge_full['created_at'].dt.hour
        df_merge_full['weekday'] = df_merge_full['created_at'].dt.weekday


        df_merge_full = df_merge_full.drop(columns=['created_at'])
        return df_merge_full

    def get_split_values(df_merge_full, value:int):
        col_name = f'credits_{value}+'
        df_modeling = df_merge_full.drop(columns=['id', 'changed_at', 'user', 'batch', 'state', 'user_id'])
        df_modeling[col_name] = (df_modeling['credits'] > value).astype(int)
        df_modeling = df_modeling.drop(columns=['credits'])

        fig = px.bar(pd.DataFrame(df_modeling[col_name].value_counts().reset_index()), x=col_name, y='count', title='Credits Category Distribution')
        fig.show()
        
        df_modeling = df_modeling[df_modeling[col_name].isnull()==False] # clean data from null values
        return df_modeling, col_name

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
            Tuple: (X_train, X_test, y_train, y_test)s
        """
        return train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y if stratify else None,
            random_state=random_state
        )

    def get_smote_train(X_train, y_train):
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        return X_train_smote, y_train_smote
    
    def full_orchestration(self, category_split:int=600):
        df_merge_full = self.get_merged_table()
        df_modeling, colname = self.get_split_values(df_merge_full, category_split)
        y = df_modeling[colname]
        X = df_modeling.drop(columns=[colname])
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y , train_size=0.8, random_state=42)
        X_train_smote, y_train_smote = self.get_smote_train(X_train, y_train)

        return [X_train, X_test, y_train, y_test], [X_train_smote, y_train_smote]