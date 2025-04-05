from src.data_preprocessing import DataPreprocessor
from src.statistical_analysis import StatisticalAnalyzer
from src.visualization import DataVisualizer
from src.model_evaluation import ModelEvaluator

def main():
    try:
        preprocessor = DataPreprocessor('data/marketing_ab_test.csv')
        df = preprocessor.prepare_data()

        analyzer = StatisticalAnalyzer(df)
        chi_square_results = analyzer.chi_square_test()
        confidence_intervals = analyzer.confidence_interval()
        effect_size = analyzer.effect_size()

        visualizer = DataVisualizer(df)
        visualizer.conversion_rate_plot()
        visualizer.conversion_by_day_plot()
        visualizer.ads_exposure_distribution()
        visualizer.conversion_distribution_boxplot()
        
        visualizer.comprehensive_visualization()
        
        visualizer.detailed_conversion_analysis()
        visualizer.conversion_probability_analysis()

        results = {
            'chi_square': chi_square_results,
            'confidence_intervals': confidence_intervals,
            'effect_size': effect_size
        }
        evaluator = ModelEvaluator(results)
        report = evaluator.generate_report()

        print(report)

    except Exception as e:
        print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    main()