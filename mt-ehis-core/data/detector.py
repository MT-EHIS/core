class LearningDetectorData:
    def __init__(self, features_count: int, coefs: dict, bias: float):
        self.features_count = features_count
        self.coefs = coefs
        self.bias = bias

    def __repr__(self):
        return "{cls_name}({features_count}, {coefs}, {bias})".format(cls_name=type(self).__name__,
                                                                      features_count=self.features_count,
                                                                      coefs=self.coefs, bias=self.bias)
