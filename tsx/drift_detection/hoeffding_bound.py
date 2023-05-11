import numpy as np

def hoeffding_drift_detected(residuals, L_val, R=1.5, delta=0.95): 
    if len(residuals) <= 1:
        return False

    residuals = np.array(residuals)
    epsilon = np.sqrt((R**2)*np.log(1/delta) / (2*L_val))
    return np.abs(residuals[-1]) > np.abs(epsilon)
