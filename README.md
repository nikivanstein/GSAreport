# Global Sensitivity Analysis Reporting
--------------------------------

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

1. Install docker
2. (optional) open a terminal and build the docker file with `docker build -t emeraldit/gsareport .`
3. Run the docker file with a volume for your data and for the output

Example to show help text:  

    > docker run -v `pwd`/output:/output -v `pwd`/data:/data emeraldit/gsareport -h

### Using python source
You can also use the package by installing the dependencies to your own system.

1. Install graph-tool (https://graph-tool.skewed.de/)
2. Install python 3.7+
3. Install node (v14+)
4. Clone the repository
5. Install all python requirements (pip install -r src/requirements.txt)
6. Run `python src/GSAreport.py -h`

## How to use
Generate a global sensitivity analysis report for a given data set or function with the simple Docker or python command options.

You always need to provide the program with a `problem` definition. This definition can be supplied as json file.
    
    #Example problem definition in python (you can store this dict using json.dump to a json file)
    problem = {
        'num_vars': dim,
        'names': ['X'+str(x) for x in range(dim)],
        'bounds': [[-5.0, 5.0]] * dim
    }

Once you have the problem definition you can directly load a pair of input and output files for analysis by passing the path to the directory in which these files are stored. The application searches for the following csv files:

- x.csv, y.csv
- x_sobol.csv, y_sobol.csv
- x_morris.csv, y_morris.csv
- x_lhs.csv, y_lhs.csv

When you have your own design of experiments you can store these in x and y.csv (space delimited). The Sobol, Morris and LHS (Latin Hypercube Sampling) files can be used when you have samples and results from a specific sampling technique which can be used for different Sensitivity analysis algorithms. If you only provide an x.csv and y.csv file, a machine learning algorithm will be used to interpolate the remaining samples.

You can also use the tool to first generate the `x_*.csv` files, then use these samples to collect and store the `y_*.csv` files using your own code or program, and finally run the tool to analyse these results.

### Common uses cases using Docker
Generate samples for evaluation by a real world function / simulator  
    
    > docker run -v `pwd`/output:/output -v `pwd`/data:/data emeraldit/gsareport -p /data/problem.json -d /data --sample --samplesize 1000

Analyse the samples with their output stored in the data folder  

    > docker run -v `pwd`/output:/output -v `pwd`/data:/data emeraldit/gsareport -p /data/problem.json -d /data -o /output

Analyse a real-world data set and use a Random Forest model to interpolate (data folder should contain x.csv and y.csv) 
 
    > docker run -v `pwd`/output:/output -v `pwd`/data:/data emeraldit/gsareport -p /data/problem.json -d /data -o /output --samplesize 10000

### Common uses cases using Python
Generate samples for evaluation by a real world function / simulator  

    > python GSAreport.py -p problem.json -d data_dir --sample --samplesize 1000

Analyse the samples with their output stored in the data folder  

    > python GSAreport.py -p problem.json -d data_dir -o output_dir

Analyse a real-world data set and use a Random Forest model to interpolate (data_dir contains x.csv and y.csv) 

    > python GSAreport.py -p problem.json -d data_dir -o output_dir --samplesize 10000

## References
This tool uses Savvy [1] and SALib [2].

[1] Blake Hough, ., Chris Fu, ., & Swapil Paliwal, . (2016). savvy: visualize high dimensionality sensitivity analysis data. Updated with full sensitivity analysis from ligpy model. (v2.0). Zenodo. https://doi.org/10.5281/zenodo.53099  
[2] Herman, J. and Usher, W. (2017) SALib: An open-source Python library for sensitivity analysis. Journal of Open Source Software, 2(9). doi:10.21105/joss.00097

