
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

When you have your own design of experiments you can store these in x and y.csv (space delimited). The Sobol, Morris and LHS (Latin Hypercube Sampling) files can be used when you have samples and results from a specific sampling technique which can be used for different Sensitivity analysis algorithms. The GSA report application can generate the `x_` version of these files (the input). Using the input files you can then evaluate the data points and store the target values `y` in the csv file with the same name convention. If you only provide an x.csv and y.csv file, a machine learning algorithm will be used to interpolate the remaining samples to generate the appropriate design of experiments required for the sensitivity analysis.

A python example to read the `x_*.csv` files  and produce the correspondig `y_*.csv` files using your own objective function is provided in the next section.

### Common use cases
There are three main steps in using the GSA report application, first to generate designs of experiments (the input files), second to evaluate these design of experiments and store them as `y_*.csv` files (using your own logic / simulators / real world experiments), and last but not least to load the data and perform the sensitivity analysis.

To generate the samples for evaluation by your own code / simulator you can run the following docker command:

=== "Docker"
    ```zsh
    docker run --rm -v "$(pwd)"/data:/data \ 
    ghcr.io/nikivanstein/gsareport:main -p /data/problem.json -d /data --sample --samplesize 1000
    ```
    Here we run a docker image called `ghcr.io/nikivanstein/gsareport:main`, which is the latest GSAreport program packaged with all the required dependencies. The following line that start with `-v` creates a volume, sharing the folder `data` in our current working directory (`"$(pwd)"` on linux and `$pwd` on windows PowerShell) with the docker image (in location `/data` on the image). That way the program can access the `data` directory to store the design of experiment files (`x_*.csv`).
=== "Python"
    ```zsh
    python GSAreport.py -p problem.json -d "$(pwd)"/data --sample --samplesize 1000
    ```
=== "Executable"
    ```zsh
    ./GSAreport -p problem.json -d "$(pwd)"/data --sample --samplesize 1000
    ```

We give the following parameters to the program, `-p` to specify where to find the `problem.json` file (in the shared volume), `-d` to specify where to find the data, `--sample` to tell the program to generate the samples and `--samplesize 1000` to specify that we want designs of experiments with 1000 samples.  
After running this step we will have 3 .csv files in our `"$(pwd)"/data` folder (X_sobol.csv, X_lhs.csv and X_morris.csv). Using these 3 files
we can generate the corresponding output files (outside of this software) using any tool. In this example we run a small python script that uses a test function from the SALib package as the problem to analyse.

```python
import numpy as np
from SALib.test_functions import Ishigami

#First we load the data
data_dir = "data"
X_sobol = np.loadtxt(f"{data_dir}/x_sobol.csv")
X_morris = np.loadtxt(f"{data_dir}/x_morris.csv")
X_lhs = np.loadtxt(f"{data_dir}/x_lhs.csv")
#generate the y values
y_sobol = Ishigami.evaluate(X_sobol)
y_morris = Ishigami.evaluate(X_morris)
y_lhs = Ishigami.evaluate(X_lhs)
#store the results in files
np.savetxt(f"{data_dir}/y_lhs.csv", y_lhs)
np.savetxt(f"{data_dir}/y_morris.csv", y_morris)
np.savetxt(f"{data_dir}/y_sobol.csv", y_sobol)
```

The next and final step is to analyse the just evaluated design of experiments using the SA methods and generate the report.

=== "Docker"
    ```zsh
    docker run --rm -v "$(pwd)"/output:/output -v "$(pwd)"/data:/data \
        ghcr.io/nikivanstein/gsareport:main -p /data/problem.json -d /data -o /output
    ```
=== "Python"
    ```zsh
    python GSAreport.py -p problem.json -d data_dir -o output_dir
    ```
=== "Executable"
    ```zsh
    ./GSAreport -p problem.json -d data_dir -o output_dir
    ```
Here we give an additional volume to our docker image such that we can access the generated output report in the output directory.
We ommit the `--sample` instruction here such that it will load the data and start the analysis.
