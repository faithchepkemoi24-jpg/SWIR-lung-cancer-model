# SWIR Lung Cancer Model

This repository contains the Python implementation of the SWIR mathematical model developed to study the progression of lung cancer associated with waterpipe (shisha) smoking in Nairobi County, Kenya.

The model was developed as part of a postgraduate research thesis titled:

**"Modified Mathematical Modelling and Analysis of Lung Cancer Progression Dynamics and Control Strategy in Kenya."**

## Model Description

The SWIR model divides the population into four compartments:

- **S(t)** – Susceptible individuals (non-smokers, cancer-free)
- **W(t)** – Waterpipe (shisha) smokers
- **I(t)** – Individuals diagnosed with lung cancer
- **R(t)** – Recovered or clinically stable individuals

The model is represented by a system of ordinary differential equations describing transitions between these compartments.

## Numerical Simulation

The model equations are solved numerically using:

- **Python 3.14**
- **SciPy `solve_ivp` ODE solver**
- **Runge–Kutta method (RK45)**

Simulations are conducted over a **50-year period** to analyze the long-term dynamics of lung cancer progression.

## Intervention Scenarios

Three intervention scenarios are considered:

1. **No intervention** (δ = 0)
2. **Weak intervention** (δ = 0.2)
3. **Strong intervention** (δ = 0.8)

These scenarios represent different levels of effectiveness of public health interventions such as smoking cessation programs and awareness campaigns.

## Files in this Repository

- `comparison.py` – Python script used to simulate the SWIR model and generate the simulation graphs.
- `README.md` – Description of the project.

## Requirements

To run the simulation, install the following Python libraries:
numpy
matplotlib
scipy


You can install them using:

## Running the Simulation

Run the Python script using:

The script will generate graphs showing the dynamics of the infected population under different intervention scenarios.

## Reproducibility

The code provided in this repository reproduces the numerical simulations presented in **Chapter 4 of the thesis**. It is made publicly available to enhance transparency, reproducibility, and further research.

## Author

Faith Chepkemoi  
Department of Mathematics  
Karatina University,Kenya
