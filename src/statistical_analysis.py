import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

class StatisticalAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe

    def chi_square_test(self):
        contingency_table = pd.crosstab(self.df['test_group'], self.df['converted'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof
        }

    def confidence_interval(self, confidence=0.95):
        ad_group = self.df[self.df['test_group'] == 'ad']['converted']
        psa_group = self.df[self.df['test_group'] == 'psa']['converted']

        def calc_ci(data):
            n = len(data)
            mean = np.mean(data)
            std_error = stats.sem(data)
            h = std_error * stats.t.ppf((1 + confidence) / 2, n - 1)
            return (mean - h, mean + h)

        return {
            'ad_group_ci': calc_ci(ad_group),
            'psa_group_ci': calc_ci(psa_group)
        }

    def logistic_regression(self):
        X = self.df[['total_ads', 'most_ads_day_encoded']]
        y = self.df['converted']

        X = sm.add_constant(X)
        model = sm.Logit(y, X).fit()

        return model.summary()

    def effect_size(self):
        ad_group = self.df[self.df['test_group'] == 'ad']['converted']
        psa_group = self.df[self.df['test_group'] == 'psa']['converted']

        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1), np.var(group2)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std

        return cohens_d(ad_group, psa_group)