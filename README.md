<h1><img src="src/gsa-logo.png" width="128" style="float:left;">Global Sensitivity Analysis Reporting</h1>


GSAreport is an application to easily generate reports that describe the global sensitivities of your input parameters as best as possible. You can use the reporting application to inspect which features are important for a given real world function / simulator or model. Using the dockerized application you can generate a report with just one line of code and no additional dependencies (except for Docker of course).

Global Sensitivity Analysis is one of the tools to better understand your machine learning models or get an understanding in real-world processes.

## What is Sensitivity Analysis?
According to Wikipedia, sensitivity analysis is "the study of how the uncertainty in the output of a mathematical model or system (numerical or otherwise) can be apportioned to different sources of uncertainty in its inputs." The sensitivity of each input is often represented by a numeric value, called the sensitivity index. Sensitivity indices come in several forms:

- *First-order* indices: measures the contribution to the output variance by a single model input alone.
- *Second-order* indices: measures the contribution to the output variance caused by the interaction of two model inputs.
- *Total-order* index: measures the contribution to the output variance caused by a model input, including both its first-order effects (the input varying alone) and all higher-order interactions.

Sensitivity Analysis is a great way of getting a better understanding of how machine learning models work (Explainable AI), what parameters are of importance in real-world applications and processes and what interactions parameters have with other parameters.  
**GSAreport** makes it easy to run a wide set of SA techniques and generates a nice and visually attractive report to inspect the results of these techniques. By using Docker no additional software needs to be installed and no coding experience is required.

## Installation

### Using Docker (Recommended)
The easiest way to use the GSAreport application is directly using docker. This way you do not need to install any third party software.

1. Install docker (https://docs.docker.com/get-docker/)
2. Run the image `emeraldit/gsareport` as container with a volume for your data and for the output generated.

Example to show help text:  

    docker run -v `pwd`/output:/output -v `pwd`/data:/data emeraldit/gsareport -h


### Using executables
If you cannot or do not want to install Docker, you can also use the pre-compiled executables from the Releases section.
The executables do not contain graph-tool support and will not generate a sobol network plot, all other functionality is included. 

You can use the executables from the command line with the same console parameters as explained below in the section <a href="#Howtouse">How to use</a>.

### Using python source
You can also use the package by installing the dependencies to your own system.

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
    
    
    #Example problem definition in python (you can store this dict using json.dump to a json file)
    dim = 5
    problem = {
        'num_vars': dim,
        'names': ['X'+str(x) for x in range(dim)],
        'bounds': [[-5.0, 5.0]] * dim
    }

Once you have the problem definition (specify it with `-p path/to/problem.json`) you can directly load an existing data set containing input and output files for analysis by passing the path to the directory (with `-d <path>`) in which these files are stored. The application searches for the following csv files:

- x.csv, y.csv  #optional, in case you use an existing design of experiments
- x_sobol.csv, y_sobol.csv
- x_morris.csv, y_morris.csv
- x_lhs.csv, y_lhs.csv

When you have your own design of experiments you can store these in x and y.csv (space delimited). The Sobol, Morris and LHS (Latin Hypercube Sampling) files can be used when you have samples and results from a specific sampling technique which can be used for different Sensitivity analysis algorithms. The GSA report application can generate the `x_` version of these files (the input). Using the input files you can then evaluate the data points and store the target values `y` in the csv file with the same name convention. If you only provide an x.csv and y.csv file, a machine learning algorithm will be used to interpolate the remaining samples to generate the appropriate design of experiments required for the sensitivity analysis.

A python example to read the `x_*.csv` files  and produce the correspondig `y_*.csv` files using your own objective function is provided in the `examples` directory.

### Common use cases using Docker
There are three main steps in using the GSA report application, first to generate designs of experiments (the input files), second to evaluate these design of experiments and store them as `y_*.csv` files (using your own logic / simulators / real world experiments), and last but not least to load the data and perform the sensitivity analysis.

To generate the samples for evaluation by your own code / simulator you can run the following docker command:

```zsh
docker run --rm \
    -v `pwd`/data:/data \
    emeraldit/gsareport -p /data/problem.json -d /data --sample --samplesize 1000
```

Here we run a docker image called `emeraldit/gsareport`, which is the GSAreport program packaged with all the required dependencies. The following line that start with `-v` creates a volume, sharing the folder `data` in our current working directory with the docker image (in location `/data` on the image). That way the program can access the `data` directory to store the design of experiment files (`x_*.csv`).
We then give the following parameters to the program, `-p` to specify where to find the `problem.json` file (in the shared volume), `-d` to specify where to find the data, `--sample` to tell the program to generate the samples and `--samplesize 1000` to specify that we want designs of experiments with 1000 samples.

Analyse the samples with their output stored in the data folder  

    docker run --rm -v `pwd`/output:/output -v `pwd`/data:/data emeraldit/gsareport -p /data/problem.json -d /data -o /output

Analyse a real-world data set and use a Random Forest model to interpolate (data folder should contain x.csv and y.csv) 
 
    docker run --rm -v `pwd`/output:/output -v `pwd`/data:/data emeraldit/gsareport -p /data/problem.json -d /data -o /output --samplesize 10000

### Common use cases using Python
Generate samples for evaluation by a real world function / simulator  

    python GSAreport.py -p problem.json -d data_dir --sample --samplesize 1000

Analyse the samples with their output stored in the data folder  

    python GSAreport.py -p problem.json -d data_dir -o output_dir

Analyse a real-world data set and use a Random Forest model to interpolate (data_dir contains x.csv and y.csv) 

    python GSAreport.py -p problem.json -d data_dir -o output_dir --samplesize 10000


## Building binaries (for developers)
If you want to build the executables yourself you can use the following commands. We use pyinstaller to package the executables.
Make sure you have pyinstaller installed using `pip install pyinstaller`.

On your operating system, build the exe once you have the python source code up and running:

    pyinstaller --distpath dist/darwin/ GSAreport.spec

We provide binaries for Linux and Mac-OS in the releases section.

## References
This tool uses Savvy [1] and SALib [2].

[1] Blake Hough, ., Chris Fu, ., & Swapil Paliwal, . (2016). savvy: visualize high dimensionality sensitivity analysis data. Updated with full sensitivity analysis from ligpy model. (v2.0). Zenodo. https://doi.org/10.5281/zenodo.53099  
[2] Herman, J. and Usher, W. (2017) SALib: An open-source Python library for sensitivity analysis. Journal of Open Source Software, 2(9). doi:10.21105/joss.00097

