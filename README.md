Synthetic Data Generation for ML-based Small-Signal Stability Assessment
================

**Authors:** Sergio A. Dorado-Rojas, Marcelo de Castro Fernandes, Luigi Vanfretti

## Cite this Work

This work has been submitted to the 2020 IEEE SmartGridComm.

> <insert paper citation>

## Installation

Clone the repository. Then, navigate with a terminal to the folder and install the requirements. We recommend to create a dedicated `conda` environment (Python==3.7).

```
git clone https://github.com/ALSETLab/Static_Small-Signal_Classification_DL-ML.git
python pip install -r requirements.txt
```

## Contact

For pulling, contact Sergio A. Dorado-Rojas (sergio.dorado.rojas@gmail.com)

## Abstract

This article presents a simulation-based massive data generation procedure with applications in training machine learning (ML) solutions to automatically assess the small-signal stability condition of a power system subjected to a contingency. This method of scenario generation for employs a Monte Carlo two-stage sampling procedure to set up a contingency condition while considering the likelihood of a given combination of line outages. The generated data is pre-processed and then used to train several ML models (logistic and softmax regression, support vector machines, $k$-nearest Neighbors, Naïve Bayes and decision trees), and a deep learning neural network. The performance of the ML algorithms shows the potential to be deployed in efficient real-time solutions to assist power system operators.
