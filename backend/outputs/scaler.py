# scalers.py
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class GlobalScalers:
    x_scaler: StandardScaler
    y_scaler: StandardScaler
