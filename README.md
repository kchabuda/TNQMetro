# TNQMetro

TNQMetro is a numerical package written in Python 3 for calculations of fundamental quantum bounds on measurement precision. Thanks to the usage of the tensor-network formalism it can beat the curse of dimensionality and provides an efficient framework to calculate bounds for finite size system as well as determine the asymptotic scaling of precision in systems where quantum enhancement amounts to a constant factor improvement over the Standard Quantum Limit. It is written in a user-friendly way so that the basic functions do not require any knowledge of tensor networks.

Introduction to the package alongside simple examples can be found in the paper [Computer Physics Communications **274**, 108282 (2022)](https://doi.org/10.1016/j.cpc.2021.108282).  
Documentation to the package can be found on the [GitHub wiki](https://github.com/kchabuda/TNQMetro/wiki).  
In-depth explanation of the tensor-network based approach to calculations of fundamental quantum bounds on measurement precision can be found in the paper [Nature Communications **11**, 250 (2020)](https://doi.org/10.1038/s41467-019-13735-9).

## Dependencies

TNQMetro requires [NumPy](https://github.com/numpy/numpy) and [ncon](https://github.com/mhauru/ncon) package.

## Installation

`pip install tnqmetro`

## Example of usage

Example of optimization of QFI using TNQMetro for N=1000 qubits with OBC and in the asymptotic regime for the problem of phase estimation with uncorrelated dephasing noise.

```
import numpy as np
import scipy.linalg
import tnqmetro

N = 1000 # number of sites in tensor-network (in this example one site = one qubit)
d = 2 # dimension of local Hilbert space (dimension of physical index)
h = np.arange(d)
h = np.diag(h) # local generator ("Hamiltonian")
c1 = 1. # uncorrelated noise strength parameter
aux = np.kron(h, np.eye(d)) - np.kron(np.eye(d), h)
Y = scipy.linalg.expm(-c1 * aux @ aux / 2) # local superoperator for uncorrelated dephasing noise

F_f, F_m_f, L_MPO_f, psi_MPS_f = tnqmetro.fin(N, [], h, [Y]) # finite appraoch
F_i, F_m_i, L_MPO_i, psi_MPS_i = tnqmetro.inf([], h, [Y]) # infinite appraoch
```
