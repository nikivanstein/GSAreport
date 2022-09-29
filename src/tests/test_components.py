import json

import pytest

from src import GSAreport
from src.benchmark import bbobbenchmarks as bn


def test_SA_components():
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
    lhs_scripts, lhs_divs = report._lhs_methods()
    morris_scripts, morris_divs, _ = report._morris_plt()
    sobol_scripts, sobol_divs = report._sobol_plt()
    assert lhs_scripts != "", "LHS script is empty"
    assert morris_scripts != "", "Morris script is empty"
    assert sobol_scripts != "", "Sobol script is empty"

    assert lhs_divs != "", "LHS div is empty"
    assert morris_divs != "", "Morris div is empty"
    assert sobol_divs != "", "Sobol div is empty"
