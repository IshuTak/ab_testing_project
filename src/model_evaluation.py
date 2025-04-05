class ModelEvaluator:
    def __init__(self, statistical_results):
        self.results = statistical_results

    def make_recommendation(self):
        p_value = self.results['chi_square']['p_value']
        effect_size = self.results['effect_size']
        
        if p_value < 0.05 and effect_size > 0.2:
            return "Strong recommendation to implement"
        elif p_value < 0.05:
            return "Cautious recommendation"
        else:
            return "No significant evidence for change"

    def generate_report(self):
        report = {
            'Chi-Square Test': self.results['chi_square'],
            'Confidence Intervals': self.results['confidence_intervals'],
            'Effect Size': self.results['effect_size'],
            'Recommendation': self.make_recommendation()
        }
        return report