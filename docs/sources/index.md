<h1>Global Sensitivity Analysis Reporting</h1>


GSAreport is an application to easily generate reports that describe the global sensitivities of your input parameters as best as possible. You can use the reporting application to inspect which features are important for a given real world function / simulator or model. Using the dockerized application you can generate a report with just one line of code and no additional dependencies (except for Docker of course).

Global Sensitivity Analysis is one of the tools to better understand your machine learning models or get an understanding in real-world processes.

## What is Sensitivity Analysis?
According to Wikipedia, sensitivity analysis is "the study of how the uncertainty in the output of a mathematical model or system (numerical or otherwise) can be apportioned to different sources of uncertainty in its inputs." The sensitivity of each input is often represented by a numeric value, called the sensitivity index. Sensitivity indices come in several forms:

- *First-order* indices: measures the contribution to the output variance by a single model input alone.
- *Second-order* indices: measures the contribution to the output variance caused by the interaction of two model inputs.
- *Total-order* index: measures the contribution to the output variance caused by a model input, including both its first-order effects (the input varying alone) and all higher-order interactions.

Sensitivity Analysis is a great way of getting a better understanding of how machine learning models work (Explainable AI), what parameters are of importance in real-world applications and processes and what interactions parameters have with other parameters.  
**GSAreport** makes it easy to run a wide set of SA techniques and generates a nice and visually attractive report to inspect the results of these techniques. By using Docker no additional software needs to be installed and no coding experience is required.
