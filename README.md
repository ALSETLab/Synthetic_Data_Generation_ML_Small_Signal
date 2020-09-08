Synthetic Data Generation for ML-based Small-Signal Stability Assessment
================
**Authors:** Sergio A. Dorado-Rojas, Marcelo de Castro Fernandes, Luigi Vanfretti

## Cite this Work

This work has been accepted and will be presented at the [2020 IEEE International Conference on Communications, Control, and Computing Technologies for Smart Grids](https://sgc2020.ieee-smartgridcomm.org/).

## Installation

Clone the repository. Then, navigate with a terminal to the folder and install the requirements. We recommend to create a dedicated `conda` environment (Python==3.7).

```
git clone https://github.com/ALSETLab/Static_Small-Signal_Classification_DL-ML
python pip install -r requirements.txt
```

## Contact

For pulling, contact Sergio A. Dorado-Rojas (sergio.dorado.rojas@gmail.com)

## Abstract

This work presents a simulation-based massive data generation procedure with applications in training machine learning (ML) solutions to automatically assess the small-signal stability condition of a power system subjected to contingencies. This method of scenario generation for employs a Monte Carlo two-stage sampling procedure to set up a contingency condition while considering the likelihood of a given combination of line outages. The generated data is pre-processed and then used to train several ML models (logistic and softmax regression, support vector machines, $$k$$-nearest Neighbors, Naïve Bayes and decision trees), and a deep learning neural network. The performance of the ML algorithms shows the potential to be deployed in efficient real-time solutions to assist power system operators.

## Scalability

Below, we provide a scalability analysis regarding how many scenarios can be generated for different system size. Since the main type of contingency studied is related to line openings, systems with more than 30 branches (such as the Nordic 44) were constrained to have a maximum of 5 simultaneous trippings (i.e., $$N-5$$). Below is a comparative chart illustrating how the number of scenarios increases for different systems.

| System  | Buses  | Number of States | Number of Variables | Number of Branches | Total Number of Scenarios | Generation Time |
|---|---|---|---|---| --- | --- |
| IEEE 9  | 9 | 24  | 203 | 9 | 510 | 0.0348 s |
| SevenBus | 7  | 132 | 678 | 18 | 262,142 | 0.0781 s |
| IEEE 14  | 14  | 49 | 426 | 20 | 1,048,574 | 0.3038 s |
| Nordic 44 | 44 | 1294 | 6315 | 79 | 24,122,225 | 5.5966 s |

This scalability benchmark was performed on a Ubuntu 18.04.5 LTS machine with a AMD Epyc 7601 32-core processor and 512 GB of RAM.
