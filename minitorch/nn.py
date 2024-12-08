from typing import Tuple

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor
from typing import Optional


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator (You can implement a new class Max that inherits from Function and call that in your max function definition. This is cleaner)
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # Reshape the tensor to split the height and width dimensions
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Permute and reshape to get the desired output format
    out = reshaped.permute(0, 1, 2, 4, 3, 5)
    out = out.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return out, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
        Tensor : batch x channel x new_height x new_width

    """
    batch, channel, height, width = input.shape
    # Get tiled input
    tiled, new_height, new_width = tile(input, kernel)
    pooled = tiled.mean(4)

    # Calculate average over the last dimension (pooling window)
    return pooled.view(batch, channel, new_height, new_width)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply argmax

    Returns:
    -------
        Tensor : 1-hot tensor with 1 in the argmax position

    """
    # out = zeros(input.shape)
    # # Get the maximum value using the existing max function
    # max_val = max(input, dim)

    # # Find positions where the input equals the max value
    # for idx in input._tensor.indices():
    #     # Create index for max_val by removing the dimension we reduced over
    #     max_idx = tuple(list(idx[:dim]) + list(idx[dim + 1 :]))
    #     if input[idx] == max_val[max_idx]:
    #         out[idx] = 1.0
    # return out
    if dim is None:
        out = input.f.max_reduce(input, 0)
    else:
        out = input.f.max_reduce(input, int(input._ensure_tensor(dim).item()))
    return out == input


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Computes max pooling given an input tensor and a kernel size"""
    batch, channel, height, width = input.shape
    new_tensor, a, b = tile(input, kernel)
    pooled = max(new_tensor, 4)
    return pooled.view(batch, channel, a, b)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout for a given probability"""
    if ignore or p == 0:
        return input
    mask = rand(input.shape, input.backend) > p
    return input * mask


def softmax(input: Tensor, dim: int) -> Tensor:
    """Applies the softmax to the input tensor along the given dimension"""
    max_vals = max(input, dim)
    shifted = input - max_vals

    # Compute exp
    exp_vals = shifted.exp()

    # Sum along dimension and reshape to match input
    sum_exp = exp_vals.sum(dim)
    out_shape = list(input.shape)
    out_shape[dim] = 1
    sum_exp = sum_exp.view(*out_shape)

    # Normalize
    return exp_vals / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Applies the log of the softmax to the input tensor along the given dimension"""
    max_input = max(input, dim)
    input_minus_max = input - max_input
    exp_input_minus_max = input_minus_max.exp()
    sum_exp = exp_input_minus_max.sum(dim)
    out_shape = list(input.shape)
    out_shape[dim] = 1
    sum_exp = sum_exp.view(*out_shape)
    return input_minus_max - sum_exp.log()


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max"""
        # max_tensor = a.f.max_reduce(a, int(dim.item()))
        # mask = a == max_tensor
        # ctx.save_for_backward(mask)
        # return max_tensor
        ctx.save_for_backward(a, dim)
        return a.f.max_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for max."""
        a, dim = ctx.saved_values
        arg_max = argmax(a, int(dim.item()))
        # Otherwise, pass the full gradient
        # Distribute gradient to max values
        return grad_output * arg_max, tensor([0.0])


def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Return a new tensor with the max of the input tensor along the given dimension."""
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), tensor([0]))
    else:
        return Max.apply(input, tensor([dim]))
