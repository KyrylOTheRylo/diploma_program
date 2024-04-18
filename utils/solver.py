from dataclasses import dataclass, field, InitVar
from abc import ABC

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
    A: np.ndarray = field(repr=False)  # Internal storage after validation
    b: np.ndarray = field(repr=False)

    u_0: np.ndarray = field(init=False)
    u_B: np.ndarray = field(init=False)
    u_N: np.ndarray = field(init=False)

    def __post_init__(self):

        self.validate_conditions()

    def validate_conditions(self):
        """Validate x_0, x_B, and x_N to ensure they are non-negative integers."""
        if not all(isinstance(x, int) and x >= 0 for x in (self.x_0, self.x_B, self.x_N)):
            raise ValueError("x_0, x_B, and x_N must be non-negative integers.")
        if self.x_0 + self.x_B + self.x_N != len(self.b):
            raise ValueError("The sum of x_0, x_B, and x_N must match the size of vector b.")

    def solve(self) -> np.ndarray:
        """Solve the matrix equation (A@x - b)**2 -> min."""
        # Solve the matrix equation
        v = np.random.rand(self.b.shape[0])
        print(v)
        x = np.linalg.pinv(self.A) @ (self.b - self.A@v) + v

        self.u_0 = x[:self.x_0]
        self.u_B = x[self.x_0:self.x_0 + self.x_B]
        self.u_N = x[self.x_0 + self.x_B:]
        return x


# Example usage:
# Create an instance of the solver

if __name__ == '__main__':
    A = np.array([[2, 1, 5], [5, 7, 8], [7, 8, 9]])
    b = np.array([11, 13, 10])
    solver = MatrixSolver(A=A, b=b, x_0=1, x_B=2, x_N=0)
    # Solve the equation
    solution = solver.solve()
    print(solution)
    print(A @ solution - b)

    print(solver.u_0, solver.u_B, solver.u_N)
    from graph_utils import plot_solution
