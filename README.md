# QREM
**QREM** - **Q**uantum **R**eadout **E**rrors **M**itigation, an open-source package for implementing 
measurement noise characterization and mitigation on quantum devices.

  
## Status of development
This package is under development and new functionalities and notebooks are expected to be added in the future.

Currently, the methods for quantum detector tomography and measurement error mitigation based on that tomography are
working properly (see Ref. [[0]](https://quantum-journal.org/papers/q-2020-04-24-257/))

Recent update implements ideas from Ref. [[0.5]](https://quantum-journal.org/papers/q-2021-06-01-464/).
This includes
a) efficient characterization of multiqubit noise correlations via Diagonal Detector Overlapping Tomography (DDOT),
b) using characterization data to construct a noise model which can be used to mitigate errors on the level of marginal probablitiy distributions (for problems such as QAOA).

Soon detailed tutorials will be added. 
  
## Dependencies
For QREM to work properly,  the following libraries should be installed:
* numpy
* scipy
* cmath
* math
* copy
* itertools
* enum
* typing
* pandas
* networkx
* typing
* collections
* tqdm 
* pickle
* colorama
* datetime
* re
* time
* qiskit (optional)
* pyquill (optional)


## Installing 
The best way to install and use this package is to simply clone the repository:
```
git clone https://github.com/fbm2718/QREM
```
In order for it to work properly one also have to initialize its' submodules. To do that, execute following commands
the repository folder:
```
git submodule init
git submodule update
```


## Jupyter tutorials
Most of the functionalities are described in detail in the comments inside the code. However, before starting to use the
package, we recommend to take a look at tutorials in jupyter notebooks:

[(**Quantum Detector Tomography**)](Tutorials/QDT)
1. [Tutorial for implementing Quantum Detector Tomography](Tutorials/QDT/01_implementing_QDT.ipynb)
2. [Tutorial for mitigating readout noise based on QDT](Tutorials/QDT/02_error_mitigation.ipynb)

(**Diagonal Detector Overlapping Tomography**)
1. Tutorials for DDOT will be added soon! In the meantime -- see [examples](examples).

## Code Style
We use PEP8 as a code style for this project.

According to PEP8 style guide all function and variable names should be in lower case. In mathematics matrices
are usually named using capital letters, for better distinction. In order for both of these obligations to be satisfied
we will use prefix _m__, to mark all matrices.


## Authors

- [Filip Maciejewski](https://github.com/fbm2718) (contact: filip.b.maciejewski@gmail.com)
- [Tomasz Rybotycki](https://github.com/Tomev)
- [Oskar Słowik](https://github.com/Feigenbaum4669)


 ## References
**The workflow of this package is mainly based on works**:
  
[0] Filip B. Maciejewski, Zoltán Zimborás, Michał Oszmaniec, "Mitigation of readout noise in near-term quantum devices
by classical post-processing based on detector tomography", 
[Quantum 4, 257 (2020)](https://quantum-journal.org/papers/q-2020-04-24-257/)

[0.5] Filip B. Maciejewski, Flavio Baccari, Zoltán Zimborás, Michał Oszmaniec, 
"Modeling and mitigation of cross-talk effects in readout noise with applications to the Quantum Approximate Optimization Algorithm", 
[Quantum 5, 464 (2021)](https://quantum-journal.org/papers/q-2021-06-01-464/)
  
**Further references:**
  
[1] Z. Hradil, J. Řeháček, J. Fiurášek, and M. Ježek, “3 maximum-likelihood methods in quantum mechanics,” in Quantum
State Estimation, edited by M. Paris and J. Řeháček (Springer Berlin Heidelberg, Berlin, Heidelberg, 2004) pp. 59–112.

[2] J. Fiurášek, [Phys. Rev. A 64, 024102 (2001)](https://arxiv.org/abs/quant-ph/0101027v2).

[3] Zbigniew Puchała, Łukasz Pawela, Aleksandra Krawiec, Ryszard Kukulski, "Strategies for optimal single-shot
discrimination of quantum measurements", [Phys. Rev. A 98, 042103 (2018)](https://arxiv.org/abs/1804.05856).

[4] T. Weissman, E. Ordentlich, G. Seroussi, S. Verdul, and M. J. Weinberger, Technical Report HPL-2003-97R1,
Hewlett-Packard Labs (2003).

[5] John A. Smolin, Jay M. Gambetta, Graeme Smith, "Maximum Likelihood, Minimum Effort", [Phys. Rev. Lett. 108, 070502
(2012)](https://arxiv.org/abs/1106.5458).

[6] J. Cotler, F. Wilczek, "Quantum Overlapping Tomography", [Phys. Rev. Lett. 124, 100401 (2020)](https://arxiv.org/abs/1908.02754).


## Citation
The following bibtex entry can be used to cite this repository:

@misc{qrem,
   url={https://github.com/fbm2718/QREM},
   title = {Quantum Readout Errors Mitigation (QREM) -- open source GitHub repository},
   author={Maciejewski, F. B. and Rybotycki, T. and S\l{}owik, O., and Oszmaniec, M.},
   year={2020},
}




