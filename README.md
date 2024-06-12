# k-epsilon turbulence model in FEniCS computing platform

This repository contains code used in my master's thesis titled: *Turbulence modeling in computational fluid dynamics* as a part of my master's degree in Applied Mathematics at the *University of Southern Denmark (SDU)*.

The $k$-$\varepsilon$ turbulence model consists of two transport equations, one for turbulent kinetic energy ($k$) and the other for dissipation of turbulent kinetic energy ($\varepsilon$). Together with a Reynolds-Averaged Navier-Stokes (RANS) equations, which are a "version" of the Navier-Stokes (N-S) equations that govern the mean/avergae flow, they form a closed set of PDEs that are capable of predicting mean quantitities of a turbulent flow. 

One of the key features of turbulent flow is the high diffusivity of heat/momentum caused by the large swirling bodies forming in the flow, called turbulent eddies. The turbulence is therefore modeled using the so-called *turbulent viscosity* ($\nu_t$). The entire set of equations is as follows:

$\frac{\partial u}{\partial t} + (u \cdot \nabla) u = - \nabla p + \nabla \cdot [(\nu + \nu_t) \nabla u] + f$

$\nabla \cdot u = 0$

$\frac{\partial k}{\partial t} + (u \cdot \nabla) k = \nabla \cdot \left[(\nu + \frac{\nu_t}{\sigma_k}) \nabla k \right] + P_k - \varepsilon$

$\frac{\partial \epsilon}{\partial t} + (u \cdot \nabla) \epsilon = \nabla \cdot \left[(\nu + \frac{\nu_t} {\sigma_{\epsilon}}) \nabla \epsilon \right] + C_1 f_1 P_k \frac{\varepsilon}{k} - C_2 f_2 \frac{\varepsilon^2}{k}$

$R = \nu_t + \left( \nabla u + (\nabla u)^T \right)$

$P_k = R : \frac{1}{2} \left( \nabla u + (\nabla u)^T \right)$

$\nu_t = C_\nu \frac{k^2}{\varepsilon}$

$f_{\nu} = \left( 1 - \exp{(-0.0165 Re_{k})} \right)^2 \left( 1 + \frac{20.5}{Re_{\ell}} \right)$

$f_1 = 1 + \left( \frac{0.05}{f_{\nu}} \right)^3 $

$f_2 = 1 - \exp{\left( -Re_{\ell}^2 \right)}$

$Re_{\ell} = \frac{k^2}{\nu \varepsilon}, \quad Re_{k} = \frac{\sqrt{k} y_d}{\nu}  $

$C_{\nu} = 0.09,\quad C_1 = 1.44, \quad C_1 = 1.92, \quad \sigma_k = 1.0, \quad \sigma_\varepsilon = 1.3$
