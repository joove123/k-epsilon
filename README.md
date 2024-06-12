# k-epsilon turbulence model in FEniCS computing platform

\begin{equations}

\frac{\partial u}{\partial t} + (u \cdot \nabla) u = - \nabla p + \nabla \cdot [(\nu + \nu_t) \nabla u]

\nabla \cdot u = 0

\frac{\partial k}{\partial t} + (u \cdot \nabla) k = \nabla \cdot \left[(\nu + \frac{\nu_t}{\sigma_k}) \nabla k \right] + P_k - k R_k
\frac{\partial \epsilon}{\partial t} + (u \cdot \nabla) \epsilon = \nabla \cdot \left[(\nu + \frac{\nu_t} {\sigma_{\epsilon}}) \nabla \epsilon \right] + P_{\epsilon} - \epsilon R_{\varepsilon}

\end{equations}

coming: 4/06/2024 
