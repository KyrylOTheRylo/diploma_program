from dataclasses import dataclass, field
from abc import ABC
from typing import Callable

import numpy as np


@dataclass()
class MatrixSolver(ABC):
    """Matrix solver interface."""

    x_0: int
    """Initial condition"""
    x_B: int
    """Boundary condition"""
    x_N: int
    """Control condition"""
    A: Callable = field(repr=False)  # Internal storage after validation
    b: np.ndarray = field(repr=False)

    X_domain: tuple[float, float] = field(init=False, default=(0, 10000))
    T_domain: tuple[float, float] = field(init=False, default=(-4, 20))
    x_points: int = field(init=False, default=100)
    t_points: int = field(init=False, default=100)

    u_0: np.ndarray = field(init=False)
    u_B: np.ndarray = field(init=False)
    u_N: np.ndarray = field(init=False)

    def __post_init__(self):
        self.validate_conditions()

    def validate_conditions(self):
        """Validate x_0, x_B, and x_N to ensure they are non-negative integers."""
        if not all(
            isinstance(x, int) and x >= 0 for x in (self.x_0, self.x_B, self.x_N)
        ):
            raise ValueError("x_0, x_B, and x_N must be non-negative integers.")
        if self.x_0 + self.x_B + self.x_N != len(self.b):
            raise ValueError(
                "The sum of x_0, x_B, and x_N must match the size of vector b."
            )

    def find_P_inv(self):
        """Find the inverse of the matrix P.
        Where P is Integral (A(s)@A(s).T)ds
        Domain S is [T]X[X]"""
        step_t = (self.T_domain[1] - self.T_domain[0]) / self.t_points
        step_x = (self.X_domain[1] - self.X_domain[0]) / self.x_points

        T_range = np.arange(self.T_domain[0], self.T_domain[1], step_t)
        X_range = np.arange(self.X_domain[0], self.X_domain[1], step_x)

        sum_of_matrices = 0
        for t in T_range:
            for x in X_range:
                A_s = self.A(x, t)
                sum_of_matrices += A_s @ A_s.T * step_t * step_x

        return np.linalg.pinv(sum_of_matrices)

    def A_vector(self):
        pass

    def solve(self) -> np.ndarray:
        """Solve the matrix equation (A@x - b)**2 -> min."""
        P_inv = self.find_P_inv()
        answer = lambda x, t: self.A(x, t).T @ P_inv @ self.b


# Example usage:
# Create an instance of the solver

if __name__ == "__main__":
    A = np.array([[2, 1, 5], [5, 7, 8], [7, 8, 9]])
    b = np.array([11, 13, 10])
    solver = MatrixSolver(A=A, b=b, x_0=1, x_B=2, x_N=0)
    print(solver.find_P_inv())
    # Solve the equation
    # solution = solver.solve()
    # print(solution)
    # print(A @ solution - b)
    #
    # print(solver.u_0, solver.u_B, solver.u_N)
    # from graph_utils import plot_solution
