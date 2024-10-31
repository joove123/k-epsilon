# Lam-Bremhorst $k$ - $\varepsilon$ turbulence model implemented in the FEniCS computational platform

This directory contains code used in my master's thesis titled: *Turbulence modeling in computational fluid dynamics* as part of my master's degree in Applied Mathematics at the *University of Southern Denmark (SDU)*. It is also part of a submission for the *proceedings of the FEniCS 2024 conference (FEniCS 2024)*.

Although only one turbulence model was implemented, namely the *Lam-Bremhorst* $k$ - $\varepsilon$ *model*, the directory is constructed in such a way that it should be easy to add other turbulence models in the ```TurbulenceModel.py``` file.

Two test cases have been implemented to validate the model: fully developed channel flow (`ChannelSimulation.py`) and flow over a backward-facing step (`BackStepSimulation.py`). However, meshes for flow around a cylinder and flow in a diffuser geometry are also provided.

![Turbulent flow over a backward-facing step (image created in ParaView)](Static/plot.png)

## Installation ##
To install, simply copy this directory to your machine. Besides FEniCS, make sure you also have the following packages installed: `numpy`, `matplotlib`.

## Governing equations ##
The $k$ - $\varepsilon$ turbulence model consists of two transport equations, one for turbulent kinetic energy ($k$) and the other for dissipation of turbulent kinetic energy ($\varepsilon$). Together with the Reynolds-Averaged Navier-Stokes (RANS) equations, which are a "version" of the Navier-Stokes (N-S) equations that govern the mean/average flow, they form a closed set of PDEs capable of predicting the mean quantities of a turbulent flow. 

One of the key features of turbulent flow is the high diffusivity of heat/momentum caused by the large swirling bodies forming in the flow, called turbulent eddies. The turbulence is therefore modeled using the so-called *turbulent viscosity* ($\nu_t$). The entire set of equations is as follows:

$\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla{\mathbf{u}} = - \nabla{\left( \frac{1}{\rho}p + \frac{2}{3}k \right)} + \nabla \cdot \left[ (\nu + \nu_t) \nabla \mathbf{u} \right] + \mathbf{f}$

$\nabla \cdot \mathbf{u} = 0$

$\frac{\partial k}{\partial t} + \mathbf{u} \cdot \nabla{k} = \nabla \cdot \left[\left(\nu + \frac{\nu_t}{\sigma_k}\right) \nabla{k} \right] + P_k - \gamma k$

$\frac{\partial \varepsilon}{\partial t} + \mathbf{u} \cdot \nabla{\varepsilon}= \nabla \cdot \left[\left(\nu + \frac{\nu_t}{\sigma_\varepsilon}\right) \nabla{\varepsilon}\right]+ C_1 f_1 P_k \gamma- C_2 f_2 \gamma \varepsilon$

$\sigma = -\frac{1}{\rho}p \mathbf{I} + \nu \left(\nabla{\mathbf{u}} + (\nabla{\mathbf{u}})^T\right), \quad \mathbf{R} = - \frac{2}{3}k \mathbf{I} + \nu_t \left(\nabla{\mathbf{u}} + (\nabla{\mathbf{u}})^T\right)$

$\nu_t = C_\nu f_\nu \frac{k^2}{\varepsilon},\quad P_k = \mathbf{R} : \frac{1}{2} \left(\nabla{\mathbf{u}} + (\nabla{\mathbf{u}})^T\right),\quad \gamma = \frac{\varepsilon}{k}$

$f_\nu = (1 - \exp{ \left( -0.0165 Re_k \right) } )^2 \left( 1 + \frac{20.5}{ Re_\ell} \right)$

$f_1 = 1 + \left( \frac{0.05}{f_\nu}\right)^3, \quad f_2 = 1 - \exp{\left(-Re_\ell^2\right)}$

$Re_k = \frac{\sqrt{k} y}{\nu}, \quad Re_\ell = \frac{k^2}{\nu \varepsilon}$

$C_\nu = 0.09, \quad C_1 = 1.44, \quad C_2 = 1.92, \quad \sigma_k = 1.0,\quad \sigma_\varepsilon = 1.3,$

where $\mathbf{u}$ and $p$ are the mean velocity and pressure, $k$ and $\varepsilon$ are the turbulent kinetic energy and its dissipation, $\rho$ and $\nu$ are fluid's density and viscosity and $\mathbf{f}$ represents external forces.

Additionally, $\nu_t$ is the turbulent viscosity, $P_k$ and $\gamma$ are the production and reaction term for $k$, and $\sigma$ is the Cauchy stress tensor. The Reynolds stress tensor ($\mathbf{R}$) was modeled using the Boussinesq hypothesis.

On top of that $f_\nu$, $f_1$ and $f_2$ are the damping functions, which solve the models' inability to predict flow near the wall. Terms $Re_k$ and $Re_\ell$ are both referred to as the turbulent Reynolds numbers, and $y$ denotes the distance to the nearest solid wall. Lastly, $C_\nu$, $C_1$, $C_2$, $\sigma_k$, and $\sigma_\varepsilon$ are the experimentally determined model constants, and $\mathbf{I}$ is the identity tensor.
