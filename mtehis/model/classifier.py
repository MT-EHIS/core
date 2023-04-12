import numpy as np
from sklearn.neural_network import MLPClassifier


class CustomMLPClassifier(MLPClassifier):
    def __init__(self, inputs_count: int, **kwargs):
        self.inputs_count = inputs_count
        super().__init__(**kwargs)
        super()._validate_input(np.array([[0] * inputs_count] * 2, np.float64), np.array([0, 1]), False, True)
        layer_units = [inputs_count] + list(self.hidden_layer_sizes) + [1]
        super()._initialize(np.zeros((1, 1)), layer_units, np.float64)

    def _init_coef(self, fan_in, fan_out, dtype):
        coef_init = np.full((fan_in, fan_out), 2 / (fan_in * fan_out), dtype=dtype)
        intercept_init = np.zeros(fan_out, dtype=dtype)
        return coef_init, intercept_init
