import json

import numpy as np
import pytest

from src import GSAreport
from src.benchmark import bbobbenchmarks as bn

report = None
problem = {}


def test_train_model():
    with open("../data/problem.json") as json_file:
        problem = json.load(json_file)
    fun, opt = bn.instantiate(5, iinstance=1)
    report = GSAreport.SAReport(
        problem,
        top=10,
        name="F5",
        output_dir="../output",
        data_dir="../data",
        model_samples=128,
    )
    X_lhs, X_morris, X_sobol = report.generateSamples(128)
    y_lhs = np.asarray(list(map(fun, X_lhs)))
    r2 = report.trainModel(X_lhs, y_lhs)
    assert report.model == True
    assert np.mean(r2) > 0
