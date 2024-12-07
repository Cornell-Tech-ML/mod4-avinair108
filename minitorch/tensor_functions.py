"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING


import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            t1 (Tensor): The first input.

        Returns:
        -------
            Tensor: The result of the forward pass.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            grad_output (Tensor): The derivative of the output.

        Returns:
        -------
            Tensor: The gradients for each input.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            t1 (Tensor): The first input.

        Returns:
        -------
            Tensor: The result of the forward pass.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            grad_output (Tensor): The derivative of the output.

        Returns:
        -------
            Tensor: The gradients for each input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            t1 (Tensor): The first input.
            t2 (Tensor): The second input.

        Returns:
        -------
            Tensor: The result of the forward pass.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            grad_output (Tensor): The derivative of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients for each input.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            t1 (Tensor): The first input.
            t2 (Tensor): The second input.

        Returns:
        -------
            Tensor: The result of the forward pass.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            grad_output (Tensor): The derivative of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients for each input.

        """
        t1, t2 = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, t2), grad_output.f.mul_zip(
            grad_output, t1
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            t1 (Tensor): The first input.

        Returns:
        -------
            Tensor: The result of the forward pass.

        """
        sig = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        """Computes the backward pass of the Sigmoid function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            grad_output (Tensor): The derivative of the output.

        Returns:
        -------
            Tuple[Tensor]: The gradients for the input.

        """
        (sig,) = ctx.saved_values
        sig_grad = grad_output.f.mul_zip(
            sig,
            grad_output.f.add_zip(tensor([1]), grad_output.f.neg_map(sig)),
        )
        return (grad_output.f.mul_zip(grad_output, sig_grad),)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Calls the forward pass of the ReLU function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            t1 (Tensor): The first input.

        Returns:
        -------
            Tensor: The result of the forward pass.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass of the ReLU function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            grad_output (Tensor): The derivative of the output.

        Returns:
        -------
            Tensor: The gradients for each input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass of the Log function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            t1 (Tensor): The first input.

        Returns:
        -------
            Tensor: The result of the forward pass.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass of the Log function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            grad_output (Tensor): The derivative of the output.

        Returns:
        -------
            Tensor: The gradients for each input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass of the Exp function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            t1 (Tensor): The first input.

        Returns:
        -------
            Tensor: The result of the forward pass.

        """
        exp = t1.f.exp_map(t1)
        ctx.save_for_backward(exp)
        return exp

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass of the Exp function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            grad_output (Tensor): The derivative of the output.

        Returns:
        -------
            Tensor: The gradients for each input.

        """
        (exp,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, exp)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Computes the forward pass of the Sum function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            a (Tensor): The first input.
            dim (Tensor): The dimension to reduce over.

        Returns:
        -------
            Tensor: The result of the forward pass.

        """
        ctx.save_for_backward(a.shape, None)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the backward pass of the Sum function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            grad_output (Tensor): The derivative of the output.

        Returns:
        -------
            Tuple[Tensor, float]: The gradients for the input.

        """
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the forward pass of the Less-than function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            t1 (Tensor): The first input.
            t2 (Tensor): The second input.

        Returns:
        -------
            Tensor: The result of the forward pass.

        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass of the Less-than function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            grad_output (Tensor): The derivative of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients for each input.

        """
        t1, t2 = ctx.saved_values
        return grad_output.zeros(t1), grad_output.zeros(t2)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the forward pass of the Equality function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            t1 (Tensor): The first input.
            t2 (Tensor): The second input.

        Returns:
        -------
            Tensor: The result of the forward pass.

        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass of the Equality function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            grad_output (Tensor): The derivative of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients for each input.

        """
        t1, t2 = ctx.saved_values
        return grad_output.zeros(t1), grad_output.zeros(t2)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the forward pass of the Is Close function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            t1 (Tensor): The first input.
            t2 (Tensor): The second input.

        Returns:
        -------
            Tensor: The result of the forward pass.

        """
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Computes the forward pass of the Permute function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            a (Tensor): The input tensor to permute.
            order (Tensor): The order tensor specifying the new dimension order.

        Returns:
        -------
            Tensor: The permuted tensor.

        """
        ctx.save_for_backward(a.shape, a._tensor.strides)
        perm_order = []
        for i in order._tensor.indices():
            perm_order.append(int(order._tensor.get(i)))
        return a._new(a._tensor.permute(*perm_order))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the backward pass of the Permute function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved information.
            grad_output (Tensor): The derivative of the output.

        Returns:
        -------
            Tuple[Tensor, float]: The gradients for each input.

        """
        (shape, strides) = ctx.saved_values
        grad_input = minitorch.Tensor.make(
            grad_output._tensor._storage, shape, strides, backend=grad_output.backend
        )
        return grad_input, 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Computes the forward pass of the View function.

        Args:
        ----
            ctx (Context): The context object to save information for backpropagation.
            a (Tensor): The input tensor to be reshaped.
            shape (Tensor): The desired shape for the output tensor.

        Returns:
        -------
            Tensor: The reshaped tensor with the specified shape.

        Raises:
        ------
            AssertionError: If the input tensor is not contiguous.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))  # type: ignore
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant
        ind : the index of the arg to compute the derivative

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
