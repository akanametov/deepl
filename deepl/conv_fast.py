import math
import numpy as np

def get_padded_input(x, pad_h, pad_w):
    pad = ((0,0), (0,0), (pad_h,pad_h), (pad_w, pad_w))
    return np.pad(x, pad)

def get_output_shape(x, kernel_h, kernel_w):
    batch_dim, channel_dim = x.shape[:2]
    channel_unfolded = channel_dim*kernel_h*kernel_w
    return (batch_dim, channel_unfolded, -1)

def get_indices_along_dim(input_d, kernel_size_d, dilation_d, padding_d, stride_d):
    blocks_d = input_d + padding_d*2
    blocks_d = blocks_d - dilation_d*(kernel_size_d - 1)

    blocks_d_indices = np.arange(0, blocks_d, stride_d)

    # Apply dilation on kernel and find its indices along dim d
    kernel_grid = np.arange(0, kernel_size_d * dilation_d, dilation_d)

    # Broadcast and add kernel staring positions (indices) with
    # kernel_grid along dim d, to get block indices along dim d
    blocks_d_indices = blocks_d_indices.reshape(1, -1)  # Reshape to [1, -1]
    kernel_mask = kernel_grid.reshape(-1, 1) 
    block_mask = blocks_d_indices + kernel_mask

    return block_mask

def unfold(input, kernel_size=(3,3), dilation=(1,1), padding=(0,0), stride=(1,1)):
    input_h, input_w = input.shape[-2:]

    stride_h, stride_w = stride[0], stride[1]
    padding_h, padding_w = padding[0], padding[1]
    dilation_h, dilation_w = dilation[0], dilation[1]
    kernel_h, kernel_w = kernel_size[0], kernel_size[1]

    blocks_row_indices = get_indices_along_dim(input_h, kernel_h, dilation_h, padding_h, stride_h)
    blocks_col_indices = get_indices_along_dim(input_w, kernel_w, dilation_w, padding_w, stride_w)

    output_shape = get_output_shape(input, kernel_h, kernel_w)
    padded_input = get_padded_input(input, padding_h, padding_w)
    
    output = np.take(padded_input, blocks_row_indices, axis=2)
    output = np.take(output, blocks_col_indices, axis=4)
    output = output.transpose((0, 1, 2, 4, 3, 5))
    return output.reshape(output_shape)

def conv2d(x, weight, stride=(1,1), padding=(0,0)):
    out_chan, in_chan, f_h, f_w = weight.shape
    n, in_chan, in_h, in_w = x.shape

    out_h = (in_h + 2 * padding[0] - f_h) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - f_w) // stride[1] + 1

    x_f = unfold(x, kernel_size=(f_h, f_w), padding=padding, stride=stride)
    x_f = x_f.transpose((0, 2, 1))
    
    weight_f = weight.reshape(out_chan, -1)
    weight_f = weight_f.T
    
    out_f = x_f @ weight_f
    out_f = out_f.transpose((0, 2, 1))
    out = out_f.reshape(n, out_chan, out_h, out_w)
    
    return out
