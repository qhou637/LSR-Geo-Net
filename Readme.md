# LSR-Net for General Manifolds

This repository contains the implementation of **LSR-Net: Inferring Long-Term Dynamics of Point Clouds on General Manifolds**. The code provides several examples across different scenarios, showcasing the use of **LSR-Net** for modeling dynamics on both Euclidean and non-Euclidean manifolds.

## Examples

### 1. **Euclidean 2D Plane**

#### 1.1 Cahn-Hilliard Equation

For the **Cahn-Hilliard** problem on a 2D Euclidean plane, you can run the following script:

```bash
python Gauss_GRF_Normal_CH_Wise_Pre_fourier_main.py
```
Make sure to execute this script from the following folder: **LSR-Geo-Net/Regular_Grids_2D/Cahn_Hilliard_code**


### 2. **Point Clouds on a Sphere**

**Note**: You need to configure your environment to be compatible with the **Spherical Fourier Neural Operators (SFNO)** framework, as outlined in the paper:

**[Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere](https://arxiv.org/abs/2301.08776)**  
Bonev B., Kurth T., Hundt C., Pathak J., Baust M., Kashinath K., Anandkumar A.; ICML 2023.

Specifically, you must install the `torch-harmonics` package, available at [https://github.com/NVIDIA/torch-harmonics](https://github.com/NVIDIA/torch-harmonics).

#### 2.1 Allen-Cahn Equation

For the **Allen-Cahn** problem on point clouds on the sphere, run the following script:

```bash
python AC_Auto_LSR_main.py
```
Make sure you execute the script from the directory:**LSR-Geo-Net/Sphere_Point_Clouds/Code/Allen_Cahn
**
 
#### 2.2 Turing System

For the **Turing System** on point clouds on a sphere, run the following script:

```bash
python Turing_Auto_LSR_main.py
```
Ensure that you execute the script from the following directory:***LSR-Geo-Net/Sphere_Point_Clouds/Code/Turing**


#### 2.3 Schnakenberg System

For the **Schnakenberg System** on point clouds on a sphere, run the following script:

```bash
python Schnaken_Auto_LSR_main.py
```
Make sure to execute the script from the following directory: **LSR-Geo-Net/Sphere_Point_Clouds/Code/Schnakenberg
**



### 3. **Point Clouds on General Manifolds**
**Note**: You must set up your environment to be compatible with the **DeltaConv** framework, as described in the SIGGRAPH 2022 paper:

**[DeltaConv: Anisotropic Operators for Geometric Deep Learning on Point Clouds](https://dl.acm.org/doi/10.1145/3528233)**  
Ruben Wiersma, Ahmad Nasikun, Elmar Eisemann, Klaus Hildebrandt.


#### 3.1 Allen-Cahn Equation on General Manifolds

To run the **Allen-Cahn** problem on point clouds on general manifolds, use the following script:

```bash
python Blob_AC_GINO_Delta_LSR_main.py
```
Ensure that you run the script from the following directory: **LSR-Geo-Net/General_Manifolds/Allen_Cahn_code
**

**Due to the large size of the datasets used in these experiments, they are not included in this repository. If you need access to the datasets, please contact us via email..**


