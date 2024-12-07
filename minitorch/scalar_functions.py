from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the function to the given scalar values.

        Args:
        ----
            *vals (ScalarLike): Scalar inputs to the function.

        Returns:
        -------
            Scalar: The result of applying the function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)
                # raw_vals.append(scalars[-1].data)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of the forward pass.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients for each input.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of the forward pass.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients for each input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of the forward pass.

        """
        ctx.save_for_backward(a, b)
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients for each input.

        """
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of the forward pass.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients for each input.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of the forward pass.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients for each input.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of the forward pass.

        """
        result = operators.sigmoid(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients for each input.

        """
        sigma: float = ctx.saved_values[0]
        return sigma * (1 - sigma) * d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of the forward pass.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients for each input.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of the forward pass.

        """
        result = operators.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients for each input.

        """
        out: float = ctx.saved_values[0]
        return d_output * out


class LT(ScalarFunction):
    """Less-than function $f(x, y) = 1$ if $x < y$, else $0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of the forward pass.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients for each input.

        """
        return 0.0, 0.0  # Derivative of comparison is zero.


class EQ(ScalarFunction):
    """Equality function $f(x, y) = 1$ if $x == y$, else $0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of the forward pass.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients for each input.

        """
        return 0.0, 0.0  # Derivative of equality check is zero.
