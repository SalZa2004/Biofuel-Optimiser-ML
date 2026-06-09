import numpy as np
class EnsembleUncertaintyFilter:
    """Filter based on ensemble variance (ExtraTreesRegressor uncertainty)."""
    
    def __init__(self, percentile_threshold: float = 90):
        self.percentile_threshold = percentile_threshold
        self.variance_threshold = None
        self.is_calibrated = False
    
    def calibrate(self, validation_variances: np.ndarray):
        """Calibrate threshold from validation set variances."""
        self.variance_threshold = np.percentile(validation_variances, self.percentile_threshold)
        self.is_calibrated = True
        
        print(f"    Variance threshold ({self.percentile_threshold}th %ile): {self.variance_threshold:.4f}")
    
    def is_reliable(self, variances: np.ndarray) -> np.ndarray:
        """Check which predictions are reliable (low variance)."""
        if not self.is_calibrated:
            return np.ones(len(variances), dtype=bool)
        
        return variances <= self.variance_threshold