import numpy as np


def linear_data(start: int,
                stop:int,
                slope:float=1.0,
                intersect:float=0.0,
                N_instances: int=50,
                sigma: float=1,
                random_seed: int=42) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_seed)
    noise = np.random.normal(0, sigma, N_instances)
    X = np.linspace(start, stop, N_instances)
    y = (slope * X + intersect) + noise
    return X, y