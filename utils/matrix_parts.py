# We have 3 matrix subelemts: A_1, A_2, and A_3. The first matrix is a part for initial conditions, the second matrix
# is a part for boundary conditions, and the third matrix is a part for control conditions.
import cmath
import math
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from numba import njit
import numpy as np
from scipy.integrate import dblquad

from diploma_program.models.equation import ConditionType
from typing import Callable, Optional


def random_vector():
    a = np.random.rand(1)[0]
    b = np.random.rand(1)[0]
    c = np.random.rand(1)[0]
    d = np.random.rand(1)[0]

    return (
        # lambda x, t: a * np.sin(x + d * 12)
        #              + b * np.cos(t - c * 101)
        #              + c * np.cos(x - 3.5 * a * d) * np.sin(t)
        #              + d * x * t * c * a * b
        lambda x, t: 0
    )


@dataclass()
class InterfaceAMatrix(ABC):
    """Matrix A interface."""

    green_function: Callable
    """Green function."""

    condition_type: ConditionType
    """Condition type. Initial, boundary, or control."""

    conditions_list_of_dict: list[dict] = field(
        default_factory={"condition": None, "partial_derivatives": None}
    )
    """Conditions dictionary.
    The dictionary contains the following
    keys: 'condition', 'partial_derivatives'.
    list represents that we can have multiple conditions."""

    matrix: Optional[np.array] = field(default=None)
    """Matrix A."""

    vector: Optional[np.array] = field(default_factory=lambda: np.array([]))
    """Vector b."""

    random_vector: Optional[Callable] = field(default_factory=random_vector)

    tau: float = field(init=False)

    @abstractmethod
    def __call__(self, x: float, t: float) -> np.array:
        """Abstract method to compute and return matrix A at given x and t."""
        raise NotImplementedError("Subclasses must implement this method.")

    def __post_init__(self):
        self.assemble_matrix_and_vector()

    def assemble_matrix_and_vector(self):
        A = []
        b = []
        inputs_output_list_concat = []
        for condition_dict in self.conditions_list_of_dict:
            # got the dictionary set of inputs and the result as a value
            inputs_output_list = [
                [list(dict_.keys())[0], list(dict_.values())[0]]
                for dict_ in condition_dict["condition"]
            ]
            inputs_output_list_concat.extend(inputs_output_list)
            # got the dictionary set of inputs that we will pass as x and tau to the green function and the result as
            # a value

        print(inputs_output_list_concat)
        for inputs_output in inputs_output_list_concat:
            x, tau = inputs_output[0]
            self.tau = tau
            # compute the green function add default arguments x and tau to the lambda function
            # to fix the "Late Binding" issue
            print(x, tau)
            G = lambda x_prime, tau_prime, x_=x, tau_=tau: self.green_function(
                x_, tau_, x_prime, tau_prime
            )
            A.append(G)
            # compute the vector b
            b.append(inputs_output[1])
        self.matrix = lambda x_, t_: np.array([i(x_, t_) for i in A]).T
        self.vector = np.array(b)

    def matrix_compute(self, x: float, t: float) -> np.array:
        """Compute and return the matrix A."""
        if np.abs(t - self.tau) < 0.00000001:
            return np.zeros_like(self.matrix(x, t))
        return self.matrix(x, t)

    def matrix_product_for_P(self, x: float, t: float) -> np.array:
        """Compute and return the matrix A."""
        inp = self.matrix_compute(x, t).reshape(-1, 1)
        return np.dot(inp, inp.T)

    def integrate_matrix_element(self, func, x_bounds, t_bounds):
        """
        Integrate a single element of the matrix over the specified bounds.

        Parameters:
        - func: Function to integrate.
        - x_bounds: Tuple specifying the lower and upper bounds for the x variable.
        - t_bounds: Tuple specifying the lower and upper bounds for the t variable.

        Returns:
        - result: The result of the integration.
        """
        try:
            # print(f"Integrating with x bounds: {x_bounds} and t bounds: {t_bounds}")
            result, _ = dblquad(
                func,
                t_bounds[0],
                t_bounds[1],
                lambda t: x_bounds[0],
                lambda t: x_bounds[1],
                epsabs=1e-02,
                epsrel=1e-02,
            )
            # print(f"Integration result: {result}, estimated error: {error}")
        except Exception as e:
            print(f"Error during integration: {e}")
            result = np.nan  # Use NaN to indicate a failed integration
        return result

    def find_matrix_P(self, x_bounds, t_bounds) -> np.array:
        """Find the matrix P."""
        integral_result = np.zeros((len(self.vector), len(self.vector)))
        for i in range(len(self.vector)):
            for j in range(len(self.vector)):
                func = lambda x, t: self.matrix_product_for_P(x, t)[i, j]
                # print(f"Integrating matrix element for x={0}, t={10}: func={func(apply_transition(5), 1/6.9)}")
                integral_result[i, j] = self.integrate_matrix_element(
                    func, x_bounds, t_bounds
                )
                # print(f"Integral result for i={i}, j={j}: {integral_result[i, j]}")
        return integral_result

    def find_vector_A_v(self, x_bounds, t_bounds) -> np.array:
        """Find the vector A_v."""
        integral_result = np.zeros(len(self.vector))
        for i in range(len(self.vector)):
            func = lambda x, t: np.dot(
                self.matrix_compute(x, t)[i], self.random_vector(x, t)
            )
            integral_result[i] = self.integrate_matrix_element(func, x_bounds, t_bounds)
        return integral_result


class A1Matrix(InterfaceAMatrix):
    """Matrix A1 class."""

    def __call__(self, x: float, t: float) -> np.array:
        """Compute and return the matrix A1 based on initial conditions."""
        # Implement the matrix computations specifically for initial conditions
        pass


class A2Matrix(InterfaceAMatrix):
    """Matrix A2 class."""

    def __call__(self, x: float, t: float) -> np.array:
        """Compute and return the matrix A2 based on boundary conditions."""
        # Implement the matrix computations specifically for boundary conditions
        pass


class A3Matrix(InterfaceAMatrix):
    """Matrix A3 class."""

    def __call__(self, x: float, t: float) -> np.array:
        """Compute and return the matrix A3 based on control conditions."""
        # Implement the matrix computations specifically for control conditions
        pass


if __name__ == "__main__":
    r = 0.02
    sigma = 0.2
    D = -r + 1 / 2 * sigma ** 2
    T = 10
    # greens_function_cmath = njit(
    #     lambda S, t, S_prime, t_prime: (
    #             1
    #             / (sigma * cmath.sqrt(2 * cmath.pi * (t - t_prime)))
    #             * cmath.exp(
    #         -((cmath.log(S / S_prime) - (r - 0.5 * sigma ** 2) * (t - t_prime)) ** 2)
    #         / (2 * sigma ** 2 * (t - t_prime))
    #     )
    #     ) if t - t_prime > 0 else 0
    # )

    greens_function_numpy = njit(
        lambda S, t, S_prime, t_prime:
        (np.exp(-(S - S_prime) ** 2 / (2 * sigma ** 2 * (t - t_prime))) / (
            np.sqrt(2 * np.pi * sigma ** 2 * (t - t_prime)))) if t - t_prime > 0 else 0)


    def apply_transition(x):
        return np.log(x) + D * T


    # Hence strike price is 50
    x_strike = 50
    conditions_0 = [
        {
            "condition": [
                # {(np.log(0.001) + D * T, 0): 0},
                # {(np.log(0.05) + D * T, 0): 0},
                {(np.log(5) + D * T, 0): 0},
                {(np.log(10) + D * T, 0): 0},
                {(np.log(15) + D * T, 0): 0},
                # {(np.log(20) + D * T, 0): 0},
                # {(np.log(25) + D * T, 0): 0},
                {(np.log(30) + D * T, 0): 0},
                # {(np.log(35) + D * T, 0): 0},
                # {(np.log(40) + D * T, 0): 0},
                {(np.log(45) + D * T, 0): 0},
                # {(np.log(50) + D * T, 0): 0},
            ],
            "partial_derivatives": None,
        }
    ]
    conditions_1 = [
        {
            "condition": [
                {(apply_transition(0.001), 7): 0},
                # {(apply_transition(0.001) - D * T, 1 / 6): 0},
                # {(apply_transition(0.001), 1 / 5): 0},
                {(apply_transition(0.001) - D * T, 1 / 5): 0},
                {(apply_transition(0.001) - D * T, 5): 0},
                {(apply_transition(0.001) - D * T, 3): 0},
                {(apply_transition(0.001), 1 / 3): 0},
                {(apply_transition(0.001) - D, 1 / 2): 0},
                # {(apply_transition(0.001), 1 / 1.5): 0},
                # {(apply_transition(0.001) - D * T, 1 / 1.3333333): 0},
                # {(apply_transition(0.001), 1): 0},
                # {(apply_transition(0.001) - D * T / 4, 2): 0},
                # {(apply_transition(0.001), 3): 0},
                # {(apply_transition(0.001) - D * T, 4): 0},
                {(apply_transition(0.001), 9): 0},
                # {(apply_transition(0.001) - D * T / 1.5, 9): 0},
                # {(apply_transition(0.001) - D * T / 2, 9): 0},
                # {(apply_transition(0.001) - D * T / 3, 9): 0},
                {(apply_transition(0.00001), 10): 0},
                # {(apply_transition(0.00000001), 100): 0},
            ],
            "partial_derivatives": None,
        }
    ]


    def apply_e_r(x, tau):
        return x * np.exp(-r * (T - tau))


    conditions_2 = [
        {
            "condition": [
                # {(apply_transition(51), 0.01): apply_e_r(1.1, 0.001)},
                {(apply_transition(55), 0.05): apply_e_r(4.3, 0.05)},
                {(apply_transition(60), 0.1): apply_e_r(8.1, 0.1)},
                {(apply_transition(80), 0.1): apply_e_r(27.1, 0.1)},
                {(apply_transition(190), 0.1): apply_e_r(139.1, 0.1)},
                {(apply_transition(1000), 0.1): apply_e_r(950, 0.1)},
                {(apply_transition(500), 0.1): apply_e_r(450, 0.1)},
                {(apply_transition(90), 1): apply_e_r(37.1, 1)},
                {(apply_transition(130), 5): apply_e_r(57.1, 1)},

            ],
            "partial_derivatives": None,
        }
    ]

    all_conditions = [conditions_0[0], conditions_1[0], conditions_2[0]]
    a1_matrix = A1Matrix(
        green_function=greens_function_numpy,
        condition_type=ConditionType.INITIAL,
        conditions_list_of_dict=all_conditions,
    )
    P0 = a1_matrix.find_matrix_P(x_bounds=(apply_transition(0.001), apply_transition(1000)), t_bounds=(-T, 0))
    A_v_0 = (
        a1_matrix.find_vector_A_v(x_bounds=(apply_transition(0.001), apply_transition(1000)), t_bounds=(-T, 0)))
    print(
        f"Matrix P0: {P0}, Vector A_v_0: {A_v_0}"
    )
    # define a boundary condition where the stock price is 0

    a2_matrix = A2Matrix(
        green_function=greens_function_numpy,
        condition_type=ConditionType.BOUNDARY,
        conditions_list_of_dict=all_conditions,
    )
    d = time.time()
    P_B = a2_matrix.find_matrix_P(x_bounds=(apply_transition(0.001) - D * T, apply_transition(0.001)), t_bounds=(0, T))
    e = time.time()
    A_v_1 = a2_matrix.find_vector_A_v(x_bounds=(apply_transition(0.001) - D * T, apply_transition(0.001)),
                                      t_bounds=(0, T))
    f = time.time()

    a3_matrix = A3Matrix(
        green_function=greens_function_numpy,
        condition_type=ConditionType.CONTROL,
        conditions_list_of_dict=all_conditions,
    )
    # P_M = a3_matrix.find_matrix_P(x_bounds=(apply_transition(0.0001), apply_transition(1000)), t_bounds=(0, T))
    # A_v_2 = a3_matrix.find_vector_A_v(x_bounds=(apply_transition(0.0001), apply_transition(1000)), t_bounds=(0, T))
    P = P0 + P_B  # + P_M
    A_v = A_v_0 + A_v_1  # + A_v_2
    # vector c is the vector that is comprehension of all the conditions
    c = a2_matrix.vector
    u_0 = lambda x, t: a1_matrix.matrix(x, t).T @ np.linalg.pinv(P) @ (c - A_v) + a1_matrix.random_vector(x, t)
    u_b = lambda x, t: a2_matrix.matrix(x, t).T @ np.linalg.pinv(P) @ (c - A_v) + a2_matrix.random_vector(x, t)
    # u_c = lambda x,t: a3_matrix.matrix(x, t).T @ np.linalg.pinv(P) @ (c - A_v) + a3_matrix.random_vector(x, t)
    y_0 = lambda x, t: dblquad(
        lambda S_prime, t_prime: greens_function_numpy(S=x, t=t, S_prime=S_prime, t_prime=t_prime) * u_0(S_prime,
                                                                                                         t_prime),
        -T, 0, lambda t_prime: apply_transition(0.001) - D * T, lambda t_prime: apply_transition(1000), epsabs=1e-02,
        epsrel=1e-02)[0]
    y_b = lambda x, t: dblquad(
        lambda S_prime, t_prime: greens_function_numpy(S=x, t=t, S_prime=S_prime, t_prime=t_prime) * u_b(S_prime,
                                                                                                         t_prime),
        0, T, lambda t_prime: apply_transition(0.001) - D * T, lambda t_prime: apply_transition(0.001), epsabs=1e-02,
        epsrel=1e-02)[0]
    y = lambda x, t: y_0(x, t) + y_b(x, t)
    print(f"y(0.001, 0): {y(apply_transition(0.001), 0)}")
    print(f"y(0.001, 0.1): {y(apply_transition(0.001), 0.1)}")
    print(f"y(0.001, 0.2): {y(apply_transition(0.001), 0.2)}")
    print(f"y(0.001, 0.3): {y(apply_transition(0.001), 0.3)}")
    print(f"y(0.100, 0.4): {y(apply_transition(100), 0.4)}")

    # no we will calculate the c value using green function and u_0 and u_b
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    def plot_function(f, x_range=(apply_transition(0.001), apply_transition(1000)), t_range=(0, 10), num_points=6):
        """
        Plots a function f(x, t) over specified ranges for x and t.

        Parameters:
        f: The function to plot, which takes two arguments x and t.
        x_range: A tuple specifying the range of x values (default is (-10, 10)).
        t_range: A tuple specifying the range of t values (default is (-10, 10)).
        num_points: Number of points to plot along each axis (default is 100).
        """
        print(x_range)
        x = np.linspace(x_range[0], x_range[1], num_points)
        t = np.linspace(t_range[0], t_range[1], num_points)
        X, T = np.meshgrid(x, t)
        Z = np.array([[f(x, t) for x in x] for t in t])

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, T, Z, cmap='viridis')
        ax.set_title('Plot of f(x, t)')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('f(x, t)')
        plt.show()


    plot_function(y)
