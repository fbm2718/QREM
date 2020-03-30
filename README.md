# QREM
**QREM** - **Q**uantum **R**eadout **E**rrors **M**itigation, an open-source package for implementing techniques of measurement error mitigation on quantum devices.

  
## Status of development
This package is under development and new functionalities and notebooks are expected to be added in the future. Currently, the methods for quantum detector tomography and measurement error mitigation based on that tomography are working properly. Soon tutorial for using mitigation funcionalities will be added.
  
## Dependencies
To work properly,  the following libraries should be installed:
* numpy
* scipy
* cmath
* math
* copy
* itertools
* enum
* typing
* qiskit

## Installing 
The best way to install and use this package is to simply clone the repository:
```
git clone https://github.com/fbm2718/QREM
```

## Jupyter tutorials
Most of the functionalities are described in detail in the comments inside the code. However, before starting to use the package, we recomend to take a look at tutorials in jupyter notebooks:
1. [Tutorial for implementing Quantum Detector Tomography](QDT_Tutorial.ipynb)



## Code Style
We use PEP8 as a code style for this project.

According to PEP8 style guide all function and variable names should be in lower case. In mathematics matrices
are usually named using capital letters, for better distinction. In order for both of these obligations to be satisfied
we will use prefix _m__, to mark all matrices.


## Authors

- Filip Maciejewski, contact: filip.b.maciejewski@gmail.com
- Tomasz Rybotycki


 ## References
**The workflow of this package is mainly based on the work**:
  
[0] Filip B. Maciejewski, Zoltán Zimborás, Michał Oszmaniec, "Mitigation of readout noise in near-term quantum devices
by classical post-processing based on detector tomography", arxiv preprint, https://arxiv.org/abs/1907.08518 (2019)
  
**Important quantum information ideas appearing in the code are, to a higher or lower degree, described in following references:**
  
[1] Z. Hradil, J. Řeháček, J. Fiurášek, and M. Ježek, “3 maximum-likelihood methods in quantum mechanics,” in Quantum
State Estimation, edited by M. Paris and J. Řeháček (Springer Berlin Heidelberg, Berlin, Heidelberg, 2004) pp. 59–112.

[2] J. Fiurášek, Physical Review A 64, 024102 (2001), arXiv:quant-ph/0101027 [quant-ph].

[3] Zbigniew Puchała, Łukasz Pawela, Aleksandra Krawiec, Ryszard Kukulski, "Strategies for optimal single-shot
discrimination of quantum measurements", Phys. Rev. A 98, 042103 (2018), https://arxiv.org/abs/1804.05856

[4] T. Weissman, E. Ordentlich, G. Seroussi, S. Verdul, and M. J. Weinberger, Technical Report HPL-2003-97R1,
Hewlett-Packard Labs (2003).

[5] John A. Smolin, Jay M. Gambetta, Graeme Smith, "Maximum Likelihood, Minimum Effort", Phys. Rev. Lett. 108, 070502
(2012), https://arxiv.org/abs/1106.5458


