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
    orcid: 0000-0001-6841-7409
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

`GSAreport` is an application to easily generate reports that describe the global sensitivities of a machine learning (ML) model, simulator or real-world process input parameters as well as possible. 
With the reporting application, you can inspect which features are important for a particular target function or model by simply providing an existing data set or generating a design of experiments to be evaluated. With the dockerized application, you can create a report with just one line of code and no additional dependencies. The report includes a wide variety of variance-, density- and model-based global sensitivity methods, which are visualized in an intuitive and interactive way.
Global sensitivity analysis (GSA) quantifies the importance of model inputs and their interactions with respect to model output. It is crucial to better understand your machine learning models or get an understanding of complex high dimensional real-world processes.

# Statement of need

`GSAreport` is a Dockerized and packaged Python application to easily generate GSA reports.
The API for `GSAreport` was designed to provide a class-based and user-friendly interface to easily utilize many different
and somewhat complex to implement GSA methods and sampling techniques.

`GSAreport` was designed to be used by both ML researchers as well as by domain experts in industry to validate and inspect different machine learning models and simulators and gain better insights into their complex industrial processes.

`GSAreport` unifies methods of the SHAP and SALib packages in one easy to use application with an interactive graphical report. Each method in the original SALib package requires a different sampling scheme and different sample size, in GSAreport we streamline this to allow for any arbitrary sample scheme and size. In addition we use empirically found rules of thumb to automatically select GSA methods that work well for a given sample size and dimensionality.

# References

The following open source software libraries are used (and extended) within `GSAreport`.

- SALib [SALibHerman2017], a sensitivity analysis library for Python containing different SA methods and sampling schemes.  
- SHAP [shapNIPS2017_7062], a local SA method for explainable AI.

# Acknowledgements

This work is partly funded by the Dutch Research Council (NWO) as part of the XAIPre project (with project number 19455).

