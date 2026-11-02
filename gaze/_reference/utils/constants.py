# Numerical constants for stability and consistency

# Epsilon values for numerical stability
EPSILON_SMALL = 1e-8        # For denominators in correlation and similar calculations
EPSILON_MEDIUM = 1e-6       # For variance calculations and small value thresholds
EPSILON_LARGE = 1e-4        # For regularization weights

# Specific use cases
FLUX_VARIANCE_EPS = EPSILON_MEDIUM      # For flux variance calculations: 1/(flux_std + eps)
CORRELATION_EPS = EPSILON_SMALL         # For correlation denominator: sqrt(var1 * var2 + eps)
EXPRESSION_THRESHOLD = EPSILON_MEDIUM   # For gene expression thresholding
MASS_BALANCE_BASE_EPS_FACTOR = EPSILON_MEDIUM  # Base epsilon factor for mass balance