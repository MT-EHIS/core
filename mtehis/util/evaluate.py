import pandas as pd
from streamad.evaluate import PointAwareMetircs
from streamad.util import CustomDS, plot
from ..model import LearningDetector


def evaluate(model: LearningDetector, train_ds: CustomDS = None, test_ds: CustomDS = None, init_window=200):
    assert (train_ds is not None or test_ds is not None, 'At least one dataset must be provided!')
    # initializing detector
    init_data = (train_ds if train_ds is not None else test_ds).data[:init_window]
    model.predict(init_data)

    # initializing metrics
    metrics = PointAwareMetircs()
    results = []

    if train_ds is not None:
        # train data evaluation before training
        scores = model.predict(train_ds.data)
        precision, recall, F1 = metrics.evaluate(train_ds.label, scores)
        results.append(['untrained', precision, recall, F1])

        # training model
        model.fit(train_ds.data, train_ds.label)

        # train data evaluation after training
        scores = model.predict(train_ds.data)
        precision, recall, F1 = metrics.evaluate(train_ds.label, scores)
        results.append(['trained', precision, recall, F1])

    if test_ds is not None:
        # test data evaluation after training
        scores = model.predict(test_ds.data)
        precision, recall, F1 = metrics.evaluate(test_ds.label, scores)
        results.append(['test', precision, recall, F1])

    return pd.DataFrame.from_records(results, columns=['name', 'precision', 'recall', 'F1']).set_index('name')
