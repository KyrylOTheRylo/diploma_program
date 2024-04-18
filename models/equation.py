import numpy
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional

from enum import Enum


class ConditionType(Enum):
    """Condition type."""
    INITIAL = 0
    BOUNDARY = 1
    CONTROL = 2


class ConditionDictType(Enum):
    """Condition dictionary type."""
    FUNCTIONAL = 0
    DISCRETE = 1


@dataclass()
class Condition:
    """Condition class."""

    condition_type: ConditionType



@dataclass()
class EuropeanOption:
    X: float
    """Stock price"""
    X_K: float
    """Strike price"""
    r: float
    """Risk-free rate"""
    tau: float
    """Time to expiration"""
    sigma: float
    """Volatility"""
    S_0: Optional[list] = field(default=None)

    G: Optional[Callable] = field(default=None)
    """Greens function"""
    Y_0: Optional[dict[Callable, Callable | float], list[int]] = field(default=(dict(), list()))


print((EuropeanOption(100, 100, 0.05, 1, 0.2)))
