import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers

'''
Some helper classes for writing custom TorchScript LSTMs.

Goals:
- Classes are easy to read, use, and extend
- Performance of custom LSTMs approach fused-kernel-levels of speed.

A few notes about features we could add to clean up the below code:
- Support enumerate with nn.ModuleList:
  https://github.com/pytorch/pytorch/issues/14471
- Support enumerate/zip with lists:
  https://github.com/pytorch/pytorch/issues/15952
- Support overriding of class methods:
  https://github.com/pytorch/pytorch/issues/10733
- Support passing around user-defined namedtuple types for readability
- Support slicing w/ range. It enables reversing lists easily.
  https://github.com/pytorch/pytorch/issues/10774
- Multiline type annotations. List[List[Tuple[Tensor,Tensor]]] is verbose
  https://github.com/pytorch/pytorch/pull/14922
'''


def script_lstm(input_size, hidden_size, num_layers, bias=True,
                batch_first=False, dropout=False, bidirectional=False):
    '''Returns a ScriptModule that mimics a PyTorch native LSTM.'''

    # The following are not implemented.
    assert bias
    assert not batch_first

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    elif dropout:
        stack_type = StackedLSTMWithDropout
        layer_type = LSTMLayer
        dirs = 1
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(num_layers, layer_type,
                      first_layer_args=[LSTMCell, input_size, hidden_size],
                      other_layer_args=[LSTMCell, hidden_size * dirs,
                                        hidden_size])

def script_gru(input_size, hidden_size, num_layers, bias=True,
                batch_first=False, dropout=False, bidirectional=False):
    '''Returns a ScriptModule that mimics a PyTorch native LSTM.'''

    # The following are not implemented.
    assert bias
    assert not batch_first

    if bidirectional:
        stack_type = StackedGRU2
        layer_type = BidirGRULayer
        dirs = 2
    else:
        stack_type = StackedGRU
        layer_type = GRULayer
        dirs = 1

    return stack_type(num_layers, layer_type,
                      first_layer_args=[GRUCell, input_size, hidden_size],
                      other_layer_args=[GRUCell, hidden_size * dirs,
                                        hidden_size])


def script_lnlstm(input_size, hidden_size, num_layers, bias=True,
                  batch_first=False, dropout=False, bidirectional=False,
                  decompose_layernorm=False):
    '''Returns a ScriptModule that mimics a PyTorch native LSTM.'''

    # The following are not implemented.
    assert bias
    assert not batch_first
    assert not dropout

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(num_layers, layer_type,
                      first_layer_args=[LayerNormLSTMCell, input_size, hidden_size,
                                        decompose_layernorm],
                      other_layer_args=[LayerNormLSTMCell, hidden_size * dirs,
                                        hidden_size, decompose_layernorm])

def script_lngru(input_size, hidden_size, num_layers, bias=True,
                  batch_first=False, dropout=False, bidirectional=False,
                  decompose_layernorm=False):
    '''Returns a ScriptModule that mimics a PyTorch native LSTM.'''

    # The following are not implemented.
    assert bias
    assert not batch_first
    assert not dropout

    if bidirectional:
        stack_type = StackedGRU2
        layer_type = BidirGRULayer
        dirs = 2
    else:
        stack_type = StackedGRU
        layer_type = GRULayer
        dirs = 1

    return stack_type(num_layers, layer_type,
                      first_layer_args=[LayerNormGRUCell, input_size, hidden_size,
                                        decompose_layernorm],
                      other_layer_args=[LayerNormGRUCell, hidden_size * dirs,
                                        hidden_size, decompose_layernorm])


LSTMState = namedtuple('LSTMState', ['hx', 'cx'])
class GRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = Parameter(torch.randn(3 * hidden_size))

    @jit.script_method
    def forward(self, input, hx):
        # type: (Tensor, Tensor) -> Tensor
        input_vecs = torch.mm(input, self.weight_ih.t()) + self.bias_ih
        hidden_vecs = torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        reset_i, update_i, new_i = input_vecs.chunk(3, 1)
        reset_h, update_h, new_h = hidden_vecs.chunk(3, 1)
        resetgate = torch.sigmoid(reset_i + reset_h)
        updategate = torch.sigmoid(update_i + update_h)

        newgate = torch.tanh(new_i + resetgate*new_h)

        hy = (1-updategate)*newgate + updategate * hx

        return hy

class LayerNorm(jit.ScriptModule):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # XXX: This is true for our LSTM / NLP use case and helps simplify code
        assert len(normalized_shape) == 1

        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    @jit.script_method
    def compute_layernorm_stats(self, input):
        mu = input.mean(-1, keepdim=True)
        sigma = input.std(-1, keepdim=True, unbiased=False) + 1e-5
        return mu, sigma

    @jit.script_method
    def forward(self, input):
        mu, sigma = self.compute_layernorm_stats(input)
        return (input - mu) / sigma * self.weight + self.bias

class LayerNormGRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, decompose_layernorm=False, dropout=0.0):
        super(LayerNormGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh = Parameter(torch.zeros(3 * hidden_size))
        self.weight_ih.data.uniform_(-0.02, 0.02)
        self.weight_hh.data.uniform_(-0.02, 0.02)
        self.bias_ih.data[:hidden_size].fill_(-1.0)
        self.bias_hh.data[:hidden_size].fill_(-1.0)
        # The layernorms provide learnable biases

        if decompose_layernorm:
            ln = LayerNorm
        else:
            ln = nn.LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.layernorm_i = ln(3 * hidden_size, elementwise_affine=False)
        self.layernorm_h = ln(3 * hidden_size, elementwise_affine=False)
#         self.layernorm_c = ln(hidden_size)

    @jit.script_method
    def forward(self, input, hx):
        # type: (Tensor, Tensor) -> Tensor
        input_vecs = self.layernorm_i(torch.mm(input, self.weight_ih.t())) + self.bias_ih
        hidden_vecs = self.layernorm_h(torch.mm(hx, self.weight_hh.t())) + self.bias_hh
        reset_i, update_i, new_i = input_vecs.chunk(3, 1)
        reset_h, update_h, new_h = hidden_vecs.chunk(3, 1)
        resetgate = torch.sigmoid(reset_i + reset_h)
        updategate = torch.sigmoid(update_i + update_h)

        newgate = self.dropout(torch.tanh(new_i + resetgate*new_h))

        hy = (1-updategate)*newgate + updategate * hx

        return hy

class GRULayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(GRULayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, hx):
        # type: (Tensor, Tensor) -> Tensor
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            hx = self.cell(inputs[i], hx)
            outputs.append(hx.clone())
        return torch.stack(outputs, 0)

class ReverseGRULayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(ReverseGRULayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, hx):
        # type: (Tensor, Tensor) -> Tensor
        inputs = torch.flip(input, 0).unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            hx = self.cell(inputs[i], hx)
            outputs.append(hx.clone())
        return torch.flip(torch.stack(outputs, 0), 0)

class BidirGRULayer(jit.ScriptModule):
    __constants__ = ['directions']

    def __init__(self, cell, *cell_args):
        super(BidirGRULayer, self).__init__()
        self.directions = nn.ModuleList([
            GRULayer(cell, *cell_args),
            ReverseGRULayer(cell, *cell_args),
        ])

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, Tensor) -> Tensor
        outputs = jit.annotate(List[Tensor], [])
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for direction in self.directions:
            state = direction(input, states[i])
            outputs.append(state.clone())
            i += 1
        return torch.cat(outputs, -1)

def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)

class StackedGRU(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedGRU, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, Tensor) -> Tensor
        output_states = jit.annotate(List[Tensor], [])
        output = input.clone()
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            output = rnn_layer(output, states[i])
            output_states.append(output.clone())
            i += 1
        return torch.stack(output_states, 0)

# Differs from StackedLSTM in that its forward method takes
# List[List[Tuple[Tensor,Tensor]]]. It would be nice to subclass StackedLSTM
# except we don't support overriding script methods.
# https://github.com/pytorch/pytorch/issues/10733
class StackedGRU2(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedGRU2, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, Tensor) -> Tensor
        output_states = jit.annotate(List[Tensor], [])
        output = input.clone()
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            output = rnn_layer(output, states[i])
            output_states.append(output.clone())
            i += 1
        return torch.stack(output_states, 0)

def test_script_rnn_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randn(5, batch, input_size)
    state = torch.randn(batch, hidden_size)
    rnn = GRULayer(LayerNormGRUCell, input_size, hidden_size)
    out_state = rnn(inp, state)
    print(inp.shape, state.shape, out_state.shape)
    # Control: pytorch native LSTM
    lstm = nn.GRU(input_size, hidden_size, 1)
    lstm_state = state.unsqueeze(0)
    for lstm_param, custom_param in zip(lstm.all_weights[0], rnn.parameters()):
        assert lstm_param.shape == custom_param.shape
        with torch.no_grad():
            lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)
    assert (out_state[0] - lstm_out[0]).abs().max() < 1e-5
    assert (out_state[1] - lstm_out[1]).abs().max() < 1e-5


def test_script_stacked_rnn(seq_len, batch, input_size, hidden_size,
                            num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = torch.stack([torch.randn(batch, hidden_size)
              for _ in range(num_layers)], 0)
    rnn = script_gru(input_size, hidden_size, num_layers)
    custom_state = rnn(inp, states)
    # custom_state = flatten_states(out_state)

    # Control: pytorch native LSTM
    lstm = nn.GRU(input_size, hidden_size, num_layers)
    # lstm_state = torch.stack(states, 0)

    for layer in range(num_layers):
        custom_params = list(rnn.parameters())[4 * layer: 4 * (layer + 1)]
        for lstm_param, custom_param in zip(lstm.all_weights[layer],
                                            custom_params):
            assert lstm_param.shape == custom_param.shape
            with torch.no_grad():
                lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, states)

    assert (custom_state[-1] - lstm_out).abs().max() < 1e-5
    assert (custom_state[:, -1] - lstm_out_state).abs().max() < 1e-5


def test_script_stacked_bidir_rnn(seq_len, batch, input_size, hidden_size,
                                  num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = torch.stack([torch.stack([torch.randn(batch, hidden_size)
               for _ in range(2)], 0)
              for _ in range(num_layers)], 0)
    rnn = script_lngru(input_size, hidden_size, num_layers, bidirectional=True)
    custom_state = rnn(inp, states)
    # custom_state = double_flatten_states(out_state)

    # Control: pytorch native LSTM
    lstm = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True)
    # lstm_state = double_flatten_states(states)
    # for layer in range(num_layers):
    #     for direct in range(2):
    #         index = 2 * layer + direct
    #         # custom_params = list(rnn.parameters())[4 * index: 4 * index + 4]
    #         # for lstm_param, custom_param in zip(lstm.all_weights[index],
    #         #                                     custom_params):
    #         #     assert lstm_param.shape == custom_param.shape
    #         #     with torch.no_grad():
    #         #         lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, states.view(num_layers*2, batch, hidden_size))
    custom_state = custom_state.view(num_layers, seq_len, batch, 2, hidden_size)
    forward_direction = custom_state[:, -1, :, 0, :]
    backward_direction = custom_state[:, 0, :, 1, :]
    hidden = torch.stack([forward_direction, backward_direction], 2)
    # print(custom_state.shape, lstm_out_state.shape, lstm_out.shape)
    #custom_state num_layers, seq_len, batch, num_directions*hidden_size
    #lstm_out_state num_layers*num_directions, batch, hidden_size
    # assert (custom_state[-1].view(-1) - lstm_out.view(-1)).abs().max() < 1e-5
    # assert (hidden.view(-1) - lstm_out_state.view(-1)).abs().max() < 1e-5


def test_script_stacked_lnlstm(seq_len, batch, input_size, hidden_size,
                               num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = torch.stack([torch.randn(batch, hidden_size)
              for _ in range(num_layers)], 0)
    rnn = script_lngru(input_size, hidden_size, num_layers)

    # just a smoke test
    out = rnn(inp, states)

torch.manual_seed(0)
# test_script_rnn_layer(5, 2, 3, 7)
# test_script_stacked_rnn(5, 2, 3, 7, 4)
# test_script_stacked_bidir_rnn(2, 1, 1, 2, 2)
# test_script_stacked_lnlstm(5, 2, 3, 7, 4)