# The function the quest's model wrote for equation E1 (a correct random implementation, taken from the quest's code).
import numpy as np


# E1
def z_statistic(mu: float, rng: np.random.Generator | None = None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    epsilon = rng.standard_normal()
    return float(mu + epsilon)
