import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        
        self.df.columns = [col.lower().replace(' ', '_') for col in self.df.columns]
        
        return self.df

    def clean_data(self):
        self.df.dropna(inplace=True)

        self.df['test_group_encoded'] = self.df['test_group'].map({'ad': 1, 'psa': 0})
        self.df['most_ads_day_encoded'] = pd.Categorical(self.df['most_ads_day']).codes

        self.df['converted'] = self.df['converted'].astype(int)

        return self.df

    def feature_engineering(self):
        self.df['conversion_rate'] = self.df.groupby('test_group')['converted'].transform('mean')

        self.df['ads_per_day'] = self.df['total_ads'] / 7  # Assuming weekly data

        return self.df

    def prepare_data(self):
        self.load_data()
        self.clean_data()
        self.feature_engineering()
        return self.df