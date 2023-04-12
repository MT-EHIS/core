import numpy as np
from typing import Union
from streamad.base import BaseDetector
from streamad.process import ZScoreCalibrator, TDigestCalibrator


class ConfiguredDetector:
    def __init__(self, detector: BaseDetector, calibrator: Union[ZScoreCalibrator, TDigestCalibrator] = None,
                 feature_index: int = None):
        self.detector = detector
        self.calibrator = calibrator
        self.feature_index = feature_index

    def fit_score(self, x):
        score = self.detector.fit_score(x) if self.feature_index is None or len(x) == 1 else self.detector.fit_score(
            np.array([x[self.feature_index]], np.float64))
        return self.calibrator.normalize(score) if self.calibrator else score

    def __str__(self):
        return self.detector.data_type + (
            ' (feature index: ' + str(self.feature_index) + ')' if self.feature_index is not None else '') + ' ' + type(
            self.detector).__name__ + ' with ' + type(self.calibrator).__name__
