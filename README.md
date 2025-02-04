**Scripts, code, and data for the paper: "Exploring the Inverse Line-Source Scattering Problem in Dielectric Cylinders with Deep Neural Networks"** 
(published in Physica Scripta-IOP, available at: https://iopscience.iop.org/article/10.1088/1402-4896/ad852c)

This repository provides the implementation and data used in the study, which explores a novel approach utilizing **deep neural networks to address the inverse line-source scattering problem in dielectric cylinders.** By employing **Multi-layer Perceptron (MLP)** models, we aim to identify the number, positions, and strengths of hidden internal sources. This is achieved using single-frequency phased data from limited measurements of real electric and magnetic surface fields. Training data for the neural networks were generated by solving the corresponding direct scattering problems using exact analytical representations. Through extensive numerical experiments, the efficiency of our approach is demonstrated, even in scenarios involving noise, reduced sample sizes, and fewer measurements. The study also examines the empirical scaling laws that govern model performance and provides a local analysis to explore how the neural networks handle the inherent ill-posedness of the considered inverse problems.

**Problem Breakdown:**

**Classification Problem:**
We classify the number of unknown sources, N=1 or 2, using the surface electric and magnetic fields. The corresponding Python code can be found in the file "classification_2sources_final.py".

**Inverse Problem with One Source:**
We predict the position and strength of a single line source inside the dielectric cylinder. This regression problem is solved in the file "regression_1source_final.py".

**Inverse Problem with Two Sources and Fixed Strength:**
We predict the positions of two line sources, assuming their strengths are fixed. The Python code for this is in "regression_2sourcesFix_final.py".

**Inverse Problem with Two Sources and Variable Strength:**
In this case, we predict both the positions and strengths of two line sources. The implementation can be found in "regression_2sourcesVary_final.py".

**Data Generation for the Direct Problem:**
The direct scattering problems were solved analytically. Specifically, the data were generated numerically from the surface fields (solutions of the direct scattering problem), evaluated at the surface ρ=α. For our experiments, we used an electric radius 
kα=2 (representative of an intermediate frequency regime) and a refractive index η=1.75. The series used in the solution were truncated at n=25 to ensure maximum accuracy. Surface measurements were taken at 10 evenly spaced observation angles, 0≤ϕ<2π, with a step of 
π/5.

**Datasets:**
For testing purposes, small subsets of the training datasets are included in the repository as CSV files. The full training datasets used in the study are available upon reasonable request.
