---
title: 'GSAreport: Easy to Use Global Sensitivity Reporting'
tags:
  - Python
  - global sensitivity analysis
  - explainable ai
authors:
  - name: van Stein, Bas^[Corresponding author]
    orcid: 0000-0002-0013-7969
    affiliation: 1
  - name: Raponi, Elena
    affiliation: 2
affiliations:
 - name: LIACS, Leiden University, The Netherlands
   index: 1
 - name: Technical University of Munich, Germany
   index: 2
date: 24 June 2022
bibliography: paper.bib

---

# Summary

`GSAreport` is an application to easily generate reports that describe the global sensitivities of a Machine Learning model, simulator or real-world process input parameters as best as possible. 
You can use the reporting application to inspect which features are important for a given real world function, simulator or model by just providing an existing data set or generating a design of experiments to be evaluated. Using the dockerized application you can generate a report with just one line of code and no additional dependencies. The report contains a wide variety of variance-, density- and model-based global sensitivity methods that are visualized in an intuitive and interactive manner.
Global Sensitivity Analysis is a crucial technique in order to better understand your machine learning models or get an understanding of complex high dimensional real-world processes.

# Statement of need

`GSAreport` is a Dockerized and packaged Python application to easily generate sensitivity analysis (SA) reports.
The API for `GSAreport` was designed to provide a class-based and user-friendly interface to easily utilize many different
and somewhat complex to implement SA methods and sampling techniques.

`GSAreport` was designed to be used by both Machine Learning researchers as well as by domain experts in industry to validate
and inspect different machine learning models and simulators and to get better insights in their complex industrial processes.

# References

The following open source software libraries are used (and extended) within `GSAreport`.

- SALib [SALibHerman2017], a sensitivity analysis library for Python containing different SA methods and sampling schemes.  
- SHAP [shapNIPS2017_7062], a local SA method for explainable AI.

# Acknowledgements

This work is partly funded by the Dutch Research Council (NWO) as part of the XAIPre project (with project number 19455).

