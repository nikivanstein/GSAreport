<h1><img src="src/gsa-logo.png" width="128" style="float:left;">Global Sensitivity Analysis Reporting</h1>


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/445445786.svg)](https://zenodo.org/badge/latestdoi/445445786)

* [See here the full documentation and how to contribute](https://basvanstein.github.io/GSAreport/)
* [How to install](https://basvanstein.github.io/GSAreport/1.0.1/installation/)
* [Quickstart and common usecases](https://basvanstein.github.io/GSAreport/1.0.1/usecases/)

GSAreport is an application to easily generate reports that describe the global sensitivities of your input parameters as best as possible. You can use the reporting application to inspect which features are important for a given real world function / simulator or model. Using the dockerized application you can generate a report with just one line of code and no additional dependencies (except for Docker of course).

Global Sensitivity Analysis is one of the tools to better understand your machine learning models or get an understanding in real-world processes.

## What is Sensitivity Analysis?
According to Wikipedia, sensitivity analysis is "the study of how the uncertainty in the output of a mathematical model or system (numerical or otherwise) can be apportioned to different sources of uncertainty in its inputs." The sensitivity of each input is often represented by a numeric value, called the sensitivity index. Sensitivity indices come in several forms:

- *First-order* indices: measures the contribution to the output variance by a single model input alone.
- *Second-order* indices: measures the contribution to the output variance caused by the interaction of two model inputs.
- *Total-order* index: measures the contribution to the output variance caused by a model input, including both its first-order effects (the input varying alone) and all higher-order interactions.

Sensitivity Analysis is a great way of getting a better understanding of how machine learning models work (Explainable AI), what parameters are of importance in real-world applications and processes and what interactions parameters have with other parameters.  
**GSAreport** makes it easy to run a wide set of SA techniques and generates a nice and visually attractive report to inspect the results of these techniques. By using Docker no additional software needs to be installed and no coding experience is required.


<figure>
<p><img alt="Report example" src="https://basvanstein.github.io/GSAreport/1.0.1/example.png">
  </p>
<figcaption>For a full example report see<a href="https://basvanstein.github.io/GSAreport/example-report/example.html">here</a>.</figcaption>
</figure>


### Downloading and setting up the source code
You can also use the python package by installing the dependencies on your own system.

1. Install graph-tool (https://graph-tool.skewed.de/)
2. Install python 3.7+
3. Install node (v14+)
4. Clone the repository with git or download the zip
5. Install all python requirements (`pip install -r src/requirements.txt`)
6. Run `python src/GSAreport.py -h`

## How to use
<div id="Howtouse"></div>

Generate a global sensitivity analysis report for a given data set or function with the simple Docker / python or executable command options.

To start, you always need to provide the program with a `problem` definition. This definition can be supplied as json file, see also `data/problem.json` for an example. The problem definition contains the dimensionality of your problem (number of input variables) `num_vars`, the `names` of these variables (X0 to X4 in the example), and the `bounds` of each variables as a list of tuples (lower bound, upper bound).
    
```python
#Example problem definition in python (you can store this dict using json.dump to a json file)
dim = 5
problem = {
    'num_vars': dim,
    'names': ['X'+str(x) for x in range(dim)],
    'bounds': [[-5.0, 5.0]] * dim
}
```

Once you have the problem definition (specify it with `-p path/to/problem.json`) you can directly load an existing data set containing input and output files for analysis by passing the path to the directory (with `-d <path>`) in which these files are stored. The application searches for the following csv files:

- x.csv, y.csv  #optional, in case you use an existing design of experiments
- x_sobol.csv, y_sobol.csv
- x_morris.csv, y_morris.csv
- x_lhs.csv, y_lhs.csv

Sample csv files can be found in the `/data/` directory of this repository. Sample files can also be generated with the `--demo` parameter.

When you have your own design of experiments you can store these in x and y.csv (space delimited). The Sobol, Morris and LHS (Latin Hypercube Sampling) files can be used when you have samples and results from a specific sampling technique which can be used for different Sensitivity analysis algorithms. The GSA report application can generate the `x_` version of these files (the input). Using the input files you can then evaluate the data points and store the target values `y` in the csv file with the same name convention. If you only provide an x.csv and y.csv file, a machine learning algorithm will be used to interpolate the remaining samples to generate the appropriate design of experiments required for the sensitivity analysis.

A python example to read the `x_*.csv` files  and produce the correspondig `y_*.csv` files using your own objective function is provided in the next section.

## Testing the Installation

Run 

```zsh
cd src
python -m pytest
```

To execute the automated tests to verify the installation.

## Building binaries (for developers)
If you want to build the executables yourself you can use the following commands. We use pyinstaller to package the executables.
Make sure you have pyinstaller installed using `pip install pyinstaller`.

On your operating system, build the exe once you have the python source code up and running:

```zsh
pyinstaller --distpath dist/darwin/ GSAreport.spec
```

We provide binaries for Linux and Mac-OS in the releases section.

To generate a new version of the documentation run `mike deploy --push --update-aliases <version> latest`

## References
This tool uses Savvy [1] and SALib [2].

[1] Hough, B., Fu, C. and Paliwal, S. (2016). savvy: visualize high dimensionality sensitivity analysis data. Updated with full sensitivity analysis from ligpy model. (v2.0). Zenodo. https://doi.org/10.5281/zenodo.53099  
[2] Herman, J. and Usher, W. (2017) SALib: An open-source Python library for sensitivity analysis. Journal of Open Source Software, 2(9). doi:10.21105/joss.00097

## Cite our paper

Use the following bibtex to cite our paper when you use GSAreport.

```
@ARTICLE{9903639,  
    author={Stein, Bas Van and Raponi, Elena and Sadeghi, Zahra and Bouman, Niek and Van Ham, Roeland C. H. J. and Bäck, Thomas},  
    journal={IEEE Access},   
    title={A Comparison of Global Sensitivity Analysis Methods for Explainable AI With an Application in Genomic Prediction},   
    year={2022},  
    volume={10},  
    number={},  
    pages={103364-103381},  
    doi={10.1109/ACCESS.2022.3210175}
}
```
