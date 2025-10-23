"""Functional programming utilities for function composition and transformation."""

from functools import reduce, wraps
from typing import Callable, TypeVar, Any

# Type variables for generic function types
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


def compose(*functions: Callable) -> Callable:
    """
    Compose functions from right to left.

    compose(f, g, h)(x) == f(g(h(x)))

    Args:
        *functions: Variable number of functions to compose

    Returns:
        A new function that is the composition of all input functions

    Example:
        >>> add_one = lambda x: x + 1
        >>> double = lambda x: x * 2
        >>> add_one_then_double = compose(double, add_one)
        >>> add_one_then_double(5)  # (5 + 1) * 2 = 12
        12
    """
    def composed(arg: Any) -> Any:
        return reduce(lambda acc, f: f(acc), reversed(functions), arg)
    return composed


def pipe(*functions: Callable) -> Callable:
    """
    Pipe functions from left to right (reverse of compose).

    pipe(f, g, h)(x) == h(g(f(x)))

    Args:
        *functions: Variable number of functions to pipe

    Returns:
        A new function that pipes data through all input functions

    Example:
        >>> add_one = lambda x: x + 1
        >>> double = lambda x: x * 2
        >>> add_one_then_double = pipe(add_one, double)
        >>> add_one_then_double(5)  # (5 + 1) * 2 = 12
        12
    """
    def piped(arg: Any) -> Any:
        return reduce(lambda acc, f: f(acc), functions, arg)
    return piped


def curry(func: Callable) -> Callable:
    """
    Transform a function that takes multiple arguments into a chain of functions
    that each take a single argument.

    Args:
        func: Function to curry

    Returns:
        Curried version of the function

    Example:
        >>> def add(a, b, c):
        ...     return a + b + c
        >>> curried_add = curry(add)
        >>> curried_add(1)(2)(3)  # Returns 6
        6
    """
    @wraps(func)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return lambda *more_args, **more_kwargs: curried(
            *(args + more_args), **{**kwargs, **more_kwargs}
        )
    return curried


def partial_right(func: Callable, *preset_args: Any, **preset_kwargs: Any) -> Callable:
    """
    Partial application from the right (fix rightmost arguments).

    Args:
        func: Function to partially apply
        *preset_args: Arguments to fix from the right
        **preset_kwargs: Keyword arguments to fix

    Returns:
        Partially applied function

    Example:
        >>> def subtract(a, b):
        ...     return a - b
        >>> subtract_from_10 = partial_right(subtract, 10)
        >>> subtract_from_10(3)  # 3 - 10 = -7
        -7
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, *preset_args, **{**preset_kwargs, **kwargs})
    return wrapper
