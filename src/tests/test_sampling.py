import json

import pytest

from src import GSAreport
from src.benchmark import bbobbenchmarks as bn

report = None
problem = {}


def test_generate_samples():
    with open("../data/problem.json") as json_file:
        problem = json.load(json_file)
    fun, opt = bn.instantiate(5, iinstance=1)
    report = GSAreport.SAReport(
        problem,
        top=10,
        name="F5",
        output_dir="../output",
        data_dir="../data",
        model_samples=500,
    )
    X_lhs, X_morris, X_sobol = report.generateSamples(128)
    assert (
        len(X_lhs) == 128 * problem["num_vars"]
    ), f"Unexpected number of samples for LHS {len(X_lhs)}"
    assert len(X_morris) == 128 * (
        problem["num_vars"] + 1
    ), f"Unexpected number of samples for Morris {len(X_morris)}"
    assert len(X_sobol) == 128 * (
        problem["num_vars"] + 1
    ), f"Unexpected number of samples for Sobol {len(X_sobol)}"


def test_generate_samples_with_model():
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
    report.loadData()
    report.sampleUsingModel()
    assert (
        len(report.x_lhs) == 128 * problem["num_vars"]
    ), f"Unexpected number of samples for LHS {len(report.x_lhs)}"
    assert len(report.x_morris) == 128 * (
        problem["num_vars"] + 1
    ), f"Unexpected number of samples for Morris {len(report.x_morris)}"
    assert len(report.x_sobol) == 128 * (
        problem["num_vars"] + 1
    ), f"Unexpected number of samples for Sobol {len(report.x_sobol)} != {128*(problem['num_vars']+1)}"
