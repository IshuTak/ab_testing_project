
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    def __init__(self, dataframe):
       
        self.df = dataframe.copy()
        
        if self.df['converted'].dtype == bool:
            self.df['converted'] = self.df['converted'].astype(int)

    def conversion_rate_plot(self):
        
        plt.figure(figsize=(10, 6))
        
        conversion_rates = self.df.groupby('test_group')['converted'].mean()
        
        conversion_rates.plot(kind='bar', color=['blue', 'orange'])
        plt.title('Conversion Rates by Test Group', fontsize=15)
        plt.xlabel('Test Group', fontsize=12)
        plt.ylabel('Conversion Rate', fontsize=12)
        plt.xticks(rotation=0)
        
        for i, v in enumerate(conversion_rates):
            plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    def conversion_by_day_plot(self):
        
        plt.figure(figsize=(12, 6))
        
        conversion_by_day = self.df.groupby(['most_ads_day', 'test_group'])['converted'].mean().unstack()
        
        conversion_by_day.plot(kind='bar', figsize=(12, 6))
        plt.title('Conversion Rates by Day and Test Group', fontsize=15)
        plt.xlabel('Day of Week', fontsize=12)
        plt.ylabel('Conversion Rate', fontsize=12)
        plt.legend(title='Test Group')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def ads_exposure_distribution(self):
        
        plt.figure(figsize=(12, 6))
        
        sns.histplot(
            data=self.df, 
            x='total_ads', 
            hue='test_group', 
            multiple='stack',
            bins=30
        )
        
        plt.title('Ad Exposure Distribution by Test Group', fontsize=15)
        plt.xlabel('Total Ads Seen', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.show()

    def conversion_distribution_boxplot(self):
        
        plt.figure(figsize=(10, 6))
        
        sns.boxplot(
            x='test_group', 
            y='converted', 
            data=self.df,
            palette='Set2'
        )
        
        sns.stripplot(
            x='test_group', 
            y='converted', 
            data=self.df,
            color='red', 
            alpha=0.3,
            jitter=0.2
        )
        
        plt.title('Conversion Distribution by Test Group', fontsize=15)
        plt.xlabel('Test Group', fontsize=12)
        plt.ylabel('Conversion (0=No, 1=Yes)', fontsize=12)
        plt.tight_layout()
        plt.show()

    def detailed_conversion_analysis(self):
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        conversion_rates = self.df.groupby('test_group')['converted'].mean()
        conversion_rates.plot(kind='bar', color=['blue', 'orange'])
        plt.title('Conversion Rates by Test Group')
        plt.ylabel('Conversion Rate')
        
        plt.subplot(1, 2, 2)
        conversion_counts = self.df.groupby(['test_group', 'converted']).size().unstack()
        conversion_counts.plot(kind='bar', stacked=True)
        plt.title('Conversion Counts by Test Group')
        plt.xlabel('Test Group')
        plt.ylabel('Number of Users')
        plt.legend(title='Converted', labels=['No', 'Yes'])
        
        plt.tight_layout()
        plt.show()

    def conversion_probability_analysis(self):
        
        plt.figure(figsize=(12, 6))
        
        conversion_probs = self.df.groupby('test_group')['converted'].agg(['mean', 'count', 'sum'])
        conversion_probs['non_converted'] = conversion_probs['count'] - conversion_probs['sum']
        
        conversion_probs[['sum', 'non_converted']].plot(kind='bar', stacked=True)
        plt.title('Conversion Probability Analysis', fontsize=15)
        plt.xlabel('Test Group', fontsize=12)
        plt.ylabel('Number of Users', fontsize=12)
        plt.legend(['Converted', 'Non-Converted'])
        
        for i, (group, row) in enumerate(conversion_probs.iterrows()):
            conversion_rate = row['mean'] * 100
            plt.text(i, row['count'], f'{conversion_rate:.2f}%', 
                     ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    def comprehensive_visualization(self):
        
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        
        conversion_rates = self.df.groupby('test_group')['converted'].mean()
        conversion_rates.plot(kind='bar', ax=axs[0, 0], color=['blue', 'orange'])
        axs[0, 0].set_title('Conversion Rates by Test Group')
        axs[0, 0].set_ylabel('Conversion Rate')
        
        conversion_by_day = self.df.groupby(['most_ads_day', 'test_group'])['converted'].mean().unstack()
        conversion_by_day.plot(kind='bar', ax=axs[0, 1])
        axs[0, 1].set_title('Conversion by Day and Test Group')
        axs[0, 1].set_xlabel('Day of Week')
        axs[0, 1].set_ylabel('Conversion Rate')
        
        sns.histplot(
            data=self.df, 
            x='total_ads', 
            hue='test_group', 
            multiple='stack', 
            ax=axs[1, 0]
        )
        axs[1, 0].set_title('Ad Exposure Distribution')
        
        sns.boxplot(
            x='test_group', 
            y='converted', 
            data=self.df, 
            ax=axs[1, 1]
        )
        axs[1, 1].set_title('Conversion Distribution by Test Group')
        
        plt.tight_layout()
        plt.show()