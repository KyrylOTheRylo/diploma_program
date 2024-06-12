import matplotlib.pyplot as plt


def plot_solution(x, title="Solution Vector"):
    """Plot the solution vector."""
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(x)), x, color="blue")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    plt.show()
