from abc import ABC, abstractmethod
import numpy as np


class Environment(ABC):
    """
    An abstract base class for different optimization or learning environments.
    """

    @abstractmethod
    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the objective function with respect to theta.

        Parameters:
        - theta (np.ndarray): The parameter vector at which to evaluate the gradient.

        Returns:
        - np.ndarray: The gradient of the objective function.
        """
        pass

    @abstractmethod
    def hessian(self, theta: np.ndarray) -> np.ndarray:
        """
        Computes the Hessian of the objective function with respect to theta.

        Parameters:
        - theta (np.ndarray): The parameter vector at which to evaluate the Hessian.

        Returns:
        - np.ndarray: The Hessian of the objective function.
        """
        pass
