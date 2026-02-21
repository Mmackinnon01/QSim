# QSim

QSim is a modular quantum simulation framework for working with open and closed quantum systems evolving under both time-dependent and independent dynamics

The structure of this module focuses on:

- General quantum state representation, with interchangability between wavevector and density matrix representation
- Generalised dynamics, composed of dynamical generators and propagators for the dynamics
- Efficient batch processing of simulations, both in the Heisenberg and Schrodinger picture
- Simple extensibility to include tensor network representations/non-markovian dynamics in future

---

## Core Components

QSim is structured using the following components

### 1. Operator

The operator class has been designed as a wrapper built on numpy arrays, implementing the logically functionality associated with an operator.

Operators support:

- Tensor product construction
- Hermitian conjugation
- Partial trace compatibility
- Embedding in larger hilbert spaces
- Time dependent operators (TOperators)

The design ensures that operator algebra remains readable and close to mathematical notation.

---

### 2. State

The module utilises a general QuantumState protocol that allows interchangable use of different types of states in dynamical evolution. Correct processing is handled via a double-dispatch approach.

State representations include:

- Wavevectors 
- Density matrices 

---


### 3. Dynamics

The framework supports:

- Unitary evolution (Time dependent/independent)
- GKSL / Lindblad master equations
- Time-dependent generators
- Callback-based logging during evolution

Evolution engines are designed to be backend-agnostic and compatible with both dense and future tensor-network representations.

---

## Example (Conceptual)

Evolution of a two qubit system with flip-flop coupling and a leaky node can be simulated using a GKSL master equation and RK4 solver as follows:

```python
from qsim.operator import sigmaX, sigmaMinus
from qsim.states import DensityMatrix
from qsim.dynamics import GKSLGenerator, RK4Propagator, Dynamics

rho = DensityMatrix(np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]))

generator = GKSLGenerator(H=sigmaX.tensor(sigmaX), jumps=[sigmaMinus.changeHilbertSpace(dims=(2,2), target_sites=(0,))])
propagator = RK4Propagator()

dynamics = Dynamics(generator, propagator)

result = dynamics.evolve(
    rho,
    ts=[10]
)


---

## Versioning

This project follows **Semantic Versioning**:

```
MAJOR.MINOR.PATCH
```

Tagged releases in Git provide reproducible checkpoints:

```bash
git checkout v1.0.0
```

Each tagged release corresponds to a stable, research-reproducible state of the code.

---

## Research Use

QSim is developed as part of ongoing research in:

- Open quantum systems
- Quantum information processing
- Quantum networking and repeater architectures
- Entanglement diagnostics and estimation

The framework is intended for:

- Method development
- Simulation benchmarking
- Reproducible academic publication

---

## License

MIT

---

## Author

Matthew Mackinnon  
PhD Research – Quantum Information and Quantum Networking