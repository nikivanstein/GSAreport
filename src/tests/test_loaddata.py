import json

import pytest

from src import GSAreport
from src.benchmark import bbobbenchmarks as bn

report = None
problem = {}


def test_load_data():
    with open("../data/problem.json") as json_file:
        problem = json.load(json_file)
    report = GSAreport.SAReport(
        problem,
        top=10,
        name="F5",
        output_dir="../output",
        data_dir="../data",
        model_samples=128,
    )
    report.loadData()
    assert (
        len(report.x_lhs) == 200 * problem["num_vars"]
    ), f"Unexpected number of samples for LHS {len(report.x_lhs)}"
    assert len(report.x_morris) == 200 * (
        problem["num_vars"] + 1
    ), f"Unexpected number of samples for Morris {len(report.x_morris)}"
    assert len(report.x_sobol) == 200 * (
        2 * problem["num_vars"] + 2
    ), f"Unexpected number of samples for Sobol {len(report.x_sobol)} != {200*(2*problem['num_vars']+2)}"
