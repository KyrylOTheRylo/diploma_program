from dataclasses import dataclass, field
from typing import Callable, Optional

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
    """Condition type. Initial, boundary, or control."""

    condition_dict_type: ConditionDictType
    """Condition dictionary type. Functional or discrete."""

    condition: Optional[dict[Callable, Callable | float]] = field(default=None)

    """Condition value. Functional or discrete."""

    partial_derivatives: Optional[list] = field(default=None)

    def __post_init__(self):
        """Check if the condition is valid."""
        pass


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

    conditions: Optional[list[Condition]] = field(default=None)
    """Conditions"""
