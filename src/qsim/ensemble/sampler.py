import numpy as np
from scipy.stats import unitary_group

from qsim.operator.base import Operator
from qsim.state.density_matrix import DensityMatrix
from qsim.state.wave_vector import Ket


class Haar:
    @staticmethod
    def generateKetBatch(
        d: int, rng: np.random.Generator | None = None, n: int = 1
    ) -> list[Ket]:
        """
        Generate a batch of Haar-random pure states (kets).

        Parameters
        ----------
        d : int
            Hilbert space dimension.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        n : int
            Number of states to generate.

        Returns
        -------
        list[Ket]
            List of Haar-random ket states of dimension d.
        """
        return [Haar.generateKet(d, rng) for i in range(n)]

    @staticmethod
    def generateKet(d: int, rng: np.random.Generator | None = None) -> Ket:
        """
        Generate a Haar-random pure state vector of dimension d.

        Parameters
        ----------
        d : int
            Hilbert space dimension.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility).

        Returns
        -------
        np.ndarray
            Complex vector of shape (d,) with unit norm.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Complex Gaussian entries
        real = rng.normal(size=d)
        imag = rng.normal(size=d)
        vec = real + 1j * imag

        # Normalize
        vec /= np.linalg.norm(vec)

        return Ket(vec)

    @staticmethod
    def generateDMBatch(
        d: int, rng: np.random.Generator | None = None, n: int = 1
    ) -> list[DensityMatrix]:
        """
        Generate a batch of Haar-random pure density matrices.

        Each density matrix is constructed from a Haar-random ket.

        Parameters
        ----------
        d : int
            Hilbert space dimension.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        n : int
            Number of density matrices to generate.

        Returns
        -------
        list[DensityMatrix]
        """
        return [Haar.generateDM(d, rng) for i in range(n)]

    @staticmethod
    def generateDM(d: int, rng: np.random.Generator | None = None) -> DensityMatrix:
        """
        Generate a Haar-random pure density matrix of dimension d.

        Parameters
        ----------
        d : int
            Hilbert space dimension.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility).

        Returns
        -------
        DensityMatrix
        """
        psi = Haar.generateKet(d, rng)
        return psi @ psi.hConj()

    @staticmethod
    def generateUnitaryBatch(
        d: int, rng: np.random.Generator | None = None, n: int = 1
    ) -> list[Operator]:
        """
        Generate a batch of Haar-random unitary operators.

        Parameters
        ----------
        d : int
            Hilbert space dimension.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        n : int
            Number of unitaries to generate.

        Returns
        -------
        list[Operator]
        """
        return [Haar.generateUnitary(d, rng) for i in range(n)]

    @staticmethod
    def generateUnitary(d: int, rng: np.random.Generator | None = None) -> Operator:
        """
        Generate a Haar-random unitary matrix of dimension d.

        Parameters
        ----------
        d : int
            Dimension of the unitary.
        rng : np.random.Generator, optional
            Random generator for reproducibility.

        Returns
        -------
        np.ndarray
            Haar-distributed unitary matrix of shape (d, d).
        """
        U = unitary_group.rvs(d, random_state=rng)

        return Operator(U)


class HilbertSchmidt:
    @staticmethod
    def generateDMBatch(
        d: int, k: int | None = None, rng: np.random.Generator | None = None, n: int = 1
    ) -> list[DensityMatrix]:
        """
        Generate a batch of density matrices from the Hilbert–Schmidt ensemble.

        Parameters
        ----------
        d : int
            Dimension of Hilbert space.
        k : int, optional
            Ancilla dimension controlling the typical rank. Default is d.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        n : int
            Number of density matrices to generate.

        Returns
        -------
        list[DensityMatrix]
            List of density matrices drawn from the Hilbert–Schmidt ensemble.
        """
        return [HilbertSchmidt.generateDM(d, k, rng) for i in range(n)]

    @staticmethod
    def generateDM(
        d: int, k: int | None = None, rng: np.random.Generator | None = None
    ) -> DensityMatrix:
        """
        Draw a random density matrix from the Hilbert–Schmidt ensemble.

        Parameters
        ----------
        d : int
            Dimension of Hilbert space.
        k : int, optional
            Ancilla dimension (controls rank). Default k=d.
        rng : np.random.Generator, optional
            Random generator for reproducibility.

        Returns
        -------
        np.ndarray
            Density matrix of shape (d, d).
        """
        if rng is None:
            rng = np.random.default_rng()
        if k is None:
            k = d

        # Ginibre matrix
        A = rng.normal(size=(d, k)) + 1j * rng.normal(size=(d, k))

        # Wishart construction
        rho = A @ A.conj().T

        # Normalize trace
        rho /= np.trace(rho)

        return DensityMatrix(rho)
