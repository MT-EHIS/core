import inspect
import numpy as np
import streamad.model as streamad_models
from streamad.process import ZScoreCalibrator
from ..data import LearningDetectorData
from ..base import ConfiguredDetector
from .classifier import CustomMLPClassifier


class LearningDetector:
    def __init__(self, features_count):
        self.features_count = features_count
        self.detectors = []

        for name, detector in inspect.getmembers(streamad_models, inspect.isclass):
            if name == 'RandomDetector':
                continue
            current_detector = detector()
            if current_detector.data_type == 'multivariate':
                self.detectors.append(ConfiguredDetector(current_detector, ZScoreCalibrator(sigma=3)))
            else:
                for feature_index in range(features_count):
                    self.detectors.append(ConfiguredDetector(detector(), ZScoreCalibrator(sigma=3), feature_index))

        self.classifier = CustomMLPClassifier(inputs_count=len(self.detectors), solver='lbfgs', hidden_layer_sizes=(),
                                              warm_start=True)

    def get_data(self):
        coefs = dict()
        for i in range(len(self.detectors)):
            coefs[str(self.detectors[i])] = self.classifier.coefs_[0][i][0]
        return LearningDetectorData(self.features_count, coefs, self.classifier.intercepts_[0][0])

    @classmethod
    def create_from_data(cls, data: LearningDetectorData):
        learning_detector = cls(data.features_count)
        for i in range(len(learning_detector.detectors)):
            detector_str = str(learning_detector.detectors[i])
            learning_detector.classifier.coefs_[0][i][0] = data.coefs.get(detector_str, 0.0)
        learning_detector.classifier.intercepts_[0][0] = data.bias
        return learning_detector

    def fit(self, xs, ys):
        scores = np.array([[detector.fit_score(x) for detector in self.detectors] for x in xs], np.float32)
        ys = ys[~np.isnan(scores).any(axis=1)]
        scores = scores[~np.isnan(scores).any(axis=1)]
        self.classifier.fit(scores, ys)

    def predict(self, xs):
        scores = np.array([[detector.fit_score(x) for detector in self.detectors] for x in xs], np.float32)
        scores[np.isnan(scores)] = 0
        return self.classifier.predict(scores)
