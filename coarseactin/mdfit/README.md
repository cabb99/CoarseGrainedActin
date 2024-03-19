# Molecular Density Fitting Using 3D Gaussian Functions

## Description
This project implements a molecular density fitting approach that tries to optimize the correlation between a experimental density map and a synthetic density map originating from molecular dynamics using 3D Gaussian functions. The algorithm generates a force in the direction that increases the correlation between the simulated and effective maps. It also allows adjusting anisotropically the spread parameter (sigma) on each direction. 

## Features
- Efficient simulation of density maps using 3D Gaussian functions.
- Optimization of molecular coordinates and sigma parameters to improve fit.
- Calculation of the correlation coefficient to measure fit quality.
- Use of numerical and analytical methods to compute derivatives.
- Implementation optimized with Numba for high performance.

## Installation

### Prerequisites
- Python 3.6+
- Numpy
- SciPy
- Numba

## Usage

### Quick Start
Here's a quick example to get you started with the `MDFit` class:

```python
import numpy as np
from coarseactin import MDFit

# Define your parameters: coordinates, sigma, experimental_map, and voxel_size
coordinates = np.array([...])  # Particle coordinates(n,3)
sigma = np.array([...])        # Standard deviation (n,3) 
density_map = np.array([...])  # Density map
voxel_size = [1, 1, 1]         # Voxel size

# Initialize the MDFit object
md_fit = MDFit(density_map, voxel_size)

#Calculate the derivative of the correlation over the coordinates and sigma
diff = md_fit.dcorr_coef(coordinates, sigma)

# Perform the fitting process just with the coordinates
md_fit.fit(coordinates, sigma)

# Access the optimized coordinates and sigma
optimized_coordinates = md_fit.coordinates
optimized_sigma = md_fit.sigma
```

## Derivation

### Energy

This class implements V_fit, a forcefield potential that can be included in a openmm molecular dynamics simulation. 

$$ V = V_{ff} +V_{Fit}$$

The potential is defined in terms of the correlation coefficient (c.c.), which is a function of the experimental and simulated densities at each voxel $(i, j, k)$.:
The effective potential $V_{Fit}$ is given by:

$$ V_{Fit} = k (1 - \text{c.c.}) $$

And the correlation coefficient (c.c.) is defined as:

$$ \text{c.c.} = \frac{\sum_{ijk} \rho_{\text{exp}}(i,j,k) \rho_{\text{sim}}(i,j,k)}{\sqrt{\sum_{ijk} \rho_{\text{exp}}(i,j,k)^2} \sqrt{\sum_{ijk} \rho_{\text{sim}}(i,j,k)^2}} $$

where $\rho_{\text{exp}}(i,j,k)$ and $\rho_{\text{sim}}(i,j,k)$ represent the experimental and synthetically simulated density of each voxel $(i,j,k)$, respectively.
The synthetically simulated density $\rho_{\text{sim}}(i,j,k)$ is obtained by integrating the three-dimensional Gaussian function over each voxel:

$$ \rho_{\text{sim}}(i,j,k) = \sum_{n=1}^N \int_{V_{ijk}} g(x,y,z;x_n,y_n,z_n) \, dx \, dy \, dz $$

with the Gaussian function for the particle $n$, $g(x,y,z;x_n,y_n,z_n)$, is defined as:

$$
g(x,y,z;x_n,y_n,z_n,\sigma_{x,n},\sigma_{y,n},\sigma_{z,n},\epsilon_n)  = \frac{\epsilon_n}{(2\pi)^{\frac{3}{2}}\sigma_{x,n}\sigma_{y,n}\sigma_{z,n}} \exp\left( -\frac{1}{2} \left[ \frac{(x-x_n)^2}{\sigma_{x,n}^2} + \frac{(y-y_n)^2}{\sigma_{y,n}^2} + \frac{(z-z_n)^2}{\sigma_{z,n}^2} \right] \right)
$$

Then the integral for each box can be written as 

$$ \rho_{\text{sim}}(i,j,k) = \sum_{n=1}^N \left( \epsilon_n \int_{x_i^{min}}^{x_i^{max}} \int_{y_j^{min}}^{y_j^{max}} \int_{z_j^{min}}^{z_j^{max}} g(x,y,z;x_n,y_n,z_n,\sigma_{x,n},\sigma_{y,n},\sigma_{z,n}) \, dz \, dy \, dx \right )$$

Where $x_i^{min}$, $x_i^{max}$, $y_j^{min}$, $y_j^{max}$, $z_j^{min}$, and $z_j^{max}$ are the boundaries in x, y and z for the voxel $(i,j,k)$. The solution to this integral, involves the error function ( $\text{erf}$ ). For a Gaussian distribution with mean $\mu$ and standard deviation $\sigma$, the integral over a finite range can be expressed using the error function as follows:

$$ \Phi(x; \mu, \sigma) = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x-\mu}{\sigma\sqrt{2}} \right) \right] $$

Where $\Phi(x; \mu, \sigma)$ represents the cumulative distribution function (CDF) for a normal distribution at point $x$. The error function ($\text{erf}(x)$) is defined as:

$$ \text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt $$

To compute the integral of the 3D Gaussian over the finite box, we apply the CDF for each dimension:

$$ \rho_{\text{sim}}(i,j,k) = \sum_{n=1}^N \epsilon_n \times \left( \Phi(x_i^{max}; x_n, \sigma_{x,n}) - \Phi(x_i^{min}; x_n, \sigma_{x,n}) \right) \times \left( \Phi(y_j^{max}; y_n, \sigma_{y,n}) - \Phi(y_j^{min}; y_n, \sigma_{y,n}) \right) \times \left( \Phi(z_k^{max}; z_n, \sigma_{z,n}) - \Phi(z_k^{min}; z_n, \sigma_{z,n}) \right) $$

### Derivatives

To compute the derivative of $V_{Fit}$ with respect to the coordinates of the nth atom $(x_n, y_n, z_n)$, we need to apply the chain rule to the derivative of $V_{Fit}$ in terms of c.c. and then the derivative of c.c. in terms of $(x_n, y_n, z_n)$. Like $V_{Fit}$ with respect to a variable $v$ is:

$$ \frac{\partial V_{Fit}}{\partial v} = \frac{\partial V_{Fit}}{\partial \text{c.c}} \left( \frac{\partial \text{c.c}}{\partial \rho_{sim}} \left( \frac{\partial \rho_{sim}}{\partial x_n} \right) \right) $$

Where the first derivative is a constant:

$$
\frac{d V_{Fit}}{d v} = k \frac{d\text{c.c.}}{dv}
$$

The second derivative is a function of $\rho_{sim}$:

$$
\frac{d\text{c.c.}}{dv} = \frac{\sum_{ijk} \frac{\partial \rho_{\text{sim}}(i,j,k)}{\partial v} \cdot \rho_{\text{exp}}(i,j,k)}{\sqrt{\sum_{ijk} \rho_{\text{sim}}(i,j,k)^2} \cdot \sqrt{\sum_{ijk} \rho_{\text{exp}}(i,j,k)^2}} - \frac{\sum_{ijk} 2 \cdot \rho_{\text{sim}}(i,j,k) \cdot \frac{\partial \rho_{\text{sim}}(i,j,k)}{\partial v} \cdot \sum_{lmn} \rho_{\text{sim}}(l,m,n) \cdot \rho_{\text{exp}}(l,m,n)}{2 \cdot \left( \sum_{ijk} \rho_{\text{sim}}(i,j,k)^2 \right)^{\frac{3}{2}} \cdot \sqrt{\sum_{ijk} \rho_{\text{exp}}(i,j,k)^2}}
$$

Where $\rho_{\text{exp}}(i,j,k)$ represents the experimental density at the voxel located at indices (i, j, k), $\rho_{\text{sim}}(i,j,k)$ is the simulated density at the voxel $(i, j, k)$, which is a function of the molecular coordinates and parameters like sigma, and $\frac{\partial \rho_{\text{sim}}(i,j,k)}{\partial v}$ denotes the partial derivative of the simulated density at voxel $(i, j, k)$ with respect to a variable $v$ which can be the coordinates or sigma.

The derivatives of the function $\Phi(x; \mu, \sigma) = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x-\mu}{\sigma\sqrt{2}} \right) \right]$ are as follows:

- In terms of $\mu$:

$$ \frac{\partial \Phi}{\partial \mu} = -\frac{e^{-\frac{(x -\mu)^2}{2\sigma^2}}}{\sqrt{2\pi}\sigma} $$

- In terms of $\sigma$:

$$ \frac{\partial \Phi}{\partial \sigma} = -\frac{(x - \mu)e^{-\frac{(x - \mu)^2}{2\sigma^2}}}{\sqrt{2\pi}\sigma^2} $$

Then the derivative of $\frac{\partial \rho_{\text{sim}}(i,j,k)}{\partial x_n}$ would be for example: 

$$
\frac{\partial \rho_{\text{sim}}(i,j,k)}{\partial x_n} = \epsilon_n \left( \frac{e^{\frac{-(x_i^{max} - x_n)^2}{2\sigma_{x,n}^2}} - e^{\frac{-(x_i^{min} - x_n)^2}{2\sigma_{x,n}^2}}}{\sqrt{2\pi}\sigma_{x,n}} \right) \times \left( \Phi(y_j^{max}; y_n, \sigma_{y,n}) - \Phi(y_j^{min}; y_n, \sigma_{y,n}) \right) \times \left( \Phi(z_k^{max}; z_n, \sigma_{z,n}) - \Phi(z_k^{min}; z_n, \sigma_{z,n}) \right)
$$

