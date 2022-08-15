
### Using Docker (Recommended)
The easiest way to use the GSAreport application is directly using docker. This way you do not need to install any third party software.

1. Install docker (https://docs.docker.com/get-docker/)
2. Run the image `ghcr.io/basvanstein/gsareport:main` as container with a volume for your data and for the output generated.

Example to show help text:  

```zsh
docker run -v `pwd`/output:/output -v `pwd`/data:/data ghcr.io/basvanstein/gsareport:main -h
```

### Using executables
If you cannot or do not want to install Docker, you can also use the pre-compiled executables from the Releases section.
The executables do not contain graph-tool support and will not generate a sobol network plot, all other functionality is included. 

You can use the executables from the command line with the same console parameters as explained in the Quick start section.

!!! important "Graph tool support"
    Note that when using the precompiled executables the Sobol network plots are not included in the reports. 
    The executables do not have support for the graph-tool package. 

### Using python source
You can also use the package by installing the dependencies to your own system.

1. Install graph-tool (https://graph-tool.skewed.de/)
2. Install python 3.7+
3. Install node (v14+)
4. Clone the repository with git or download the zip
5. Install all python requirements (`pip install -r src/requirements.txt`)
6. Run `python src/GSAreport.py -h`