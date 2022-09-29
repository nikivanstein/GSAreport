import pytest
from src import GSAreport
import json
from src.benchmark import bbobbenchmarks as bn

report = None
problem = {}



def test_generate_samples():
    with open("data/problem.json") as json_file:
        problem = json.load(json_file)
    fun, opt = bn.instantiate(5, iinstance=1)
    report = GSAreport.SAReport(
        problem,
        top=10,
        name="F5",
        output_dir="output",
        data_dir="../data",
        model_samples=500,
    )
    X_lhs, X_morris, X_sobol = report.generateSamples(100)
    assert len(X_lhs) == 100*problem['num_vars'], "Unexpected number of samples for LHS"
    assert len(X_morris) == 100*(problem['num_vars']+1), "Unexpected number of samples for Morris"
    assert len(X_sobol) == 100*(problem['num_vars']+2), "Unexpected number of samples for Sobol"