"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiply two numbers (floats) together"""
    return x * y


def id(x: float) -> float:
    """Returns a given number back unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers (floats) to each other"""
    return x + y


def neg(x: float) -> float:
    """Returns the negation of the input number"""
    return -x


def lt(x: float, y: float) -> float:
    """Checks if one input number is less than the other"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two input numbers are equal"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the greater of the two input numbers"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close in value"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function of x"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


EPS = 1e-6


def relu(x: float) -> float:
    """Computes the ReLU of x"""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Computes the natural logarithm"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal of arg"""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second arg"""
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    return d if x > 0 else 0.0


def map(func: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element of an iterable.

    Args:
    ----
        func: A function that takes an element of type float and returns type float.

    Returns:
    -------
        An iterable of elements of type float obtained by applying `func` to each element in `iterable`.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for l in ls:
            ret.append(func(l))
        return ret

    return _map


def zipWith(
    func: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combines elements from two iterables using a given function.

    Args:
    ----
        func: A function that takes two floats (one from each iterable) and combines them into a float.
        iterable1: The first iterable of elements of type float.
        iterable2: The second iterable of elements of type float.

    Returns:
    -------
        An iterable of combined floats obtained by applying `func` to pairs from `iterable1` and `iterable2`.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(func(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduces an iterable to a single value using a given function.

    Args:
    ----
        fn: A function that combines two elements of the iterable into a float.
        start: The initial value to start the reduction.

    Returns:
    -------
        A callable function that reduces an iterable to a single value using a given function.

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negates each element in the input list."""
    return map(neg)(lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements from two lists."""
    return zipWith(add)(lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Sums all elements in the list."""
    return reduce(add, 0.0)(lst)


def prod(lst: Iterable[float]) -> float:
    """Takes the product of all elements in the list."""
    return reduce(mul, 1.0)(lst)
