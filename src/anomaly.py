import numpy as np
from sklearn.ensemble import IsolationForest

def detect(seed=42):
    rng = np.random.default_rng(seed)
    latency_ms = rng.normal(110, 12, 200)
    error_rate = np.clip(rng.normal(0.01, 0.004, 200), 0, 1)
    latency_ms[[35, 112, 178]] = [410, 520, 360]
    error_rate[[35, 112, 178]] = [0.21, 0.32, 0.19]
    X = np.column_stack([latency_ms, error_rate])
    labels = IsolationForest(contamination=0.025, random_state=42).fit_predict(X)
    indices = np.where(labels == -1)[0].tolist()
    print('Anomalías (índices de muestra):', indices)
    print('Nota: una anomalía estadística requiere investigación; no prueba un incidente.')
    return indices

if __name__ == '__main__':
    detect()
