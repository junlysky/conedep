import tensorflow as tf

class GRUCell:
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
               bias_initializer=None,
               name=None,
               dtype=None):
    
    self._num_units = num_units
    self._activation = activation or tf.nn.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer


  def build(self, input_depth, name):

    self._gate_kernel = tf.get_variable(
        "gates_kernel/%s" %name,
        shape=[input_depth + self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias = tf.get_variable(
        "gates_bias/%s" %name,
        shape=[2 * self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else tf.constant_initializer(1.0)))
    self._candidate_kernel = tf.get_variable(
        "candidate_kernel/%s" %name,
        shape=[input_depth + self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_bias = tf.get_variable(
        "candidate_bias/%s" %name,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else tf.constant_initializer(0.0)))

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""

    gate_inputs = tf.matmul(
        tf.concat([inputs, state], 1), self._gate_kernel)
    gate_inputs = tf.nn.bias_add(gate_inputs, self._gate_bias)

    value = tf.sigmoid(gate_inputs)
    r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    candidate = tf.matmul(
        tf.concat([inputs, r_state], 1), self._candidate_kernel)
    candidate = tf.nn.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    new_h = u * state + (1 - u) * c
    return new_h, r, u

    
    
class RNN:
    def __init__(self,
               inputs,
               num_units,
#               num_layers,
               activation=None,
               reuse=None,
               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
               bias_initializer=None,
               name=None,
               dtype=None):
    
        self._inputs = inputs
        self._num_units = num_units
#       self._layers = 
    
    def bi_rnn(self):
    
        bs, num_step, step_len = self._inputs.get_shape().as_list()
        fw_cell = GRUCell(self._num_units)
        bw_cell = GRUCell(self._num_units)
        
        output = {}
        # forward pass
        # layer 1
        fw_cell.build(step_len, 'fw1')
        init_state = tf.zeros([bs, self._num_units])
        last_state = init_state
        out_layer_state  = []
        out_layer_reset  = []
        out_layer_update = []
        for i in range(num_step):
            current_step = self._inputs[:,i,:]
            new_state, r, u = fw_cell.call(current_step, last_state)
            out_layer_state.append(new_state)
            out_layer_reset.append(r)
            out_layer_update.append(u)
            last_state = new_state
        output['state_fw']  = [tf.transpose(tf.stack(out_layer_state), [1,0,2])]
        output['reset_fw']  = [tf.transpose(tf.stack(out_layer_reset), [1,0,2])]
        output['update_fw'] = [tf.transpose(tf.stack(out_layer_update),[1,0,2])]
        # layer 2
        out_layer_state  = []
        out_layer_reset  = []
        out_layer_update = []
        fw_cell.build(self._num_units, 'fw2')
        for i in range(num_step):
            current_step = output['state_fw'][0][:,i,:]
            new_state, r, u = fw_cell.call(current_step, last_state)
            out_layer_state.append(new_state)
            out_layer_reset.append(r)
            out_layer_update.append(u)
            last_state = new_state
        out_layer_state  = tf.transpose(tf.stack(out_layer_state), [1,0,2])
        out_layer_reset  = tf.transpose(tf.stack(out_layer_reset), [1,0,2])
        out_layer_update = tf.transpose(tf.stack(out_layer_update),[1,0,2])
        output['state_fw'].append(out_layer_state)
        output['reset_fw'].append(out_layer_reset)
        output['update_fw'].append(out_layer_update)
        
        # backward pass
        # layer 1
        bw_cell.build(step_len, 'bw1')
        last_state = init_state
        out_layer_state  = []
        out_layer_reset  = []
        out_layer_update = []
        for i in range(num_step-1,-1,-1):
            current_step = self._inputs[:,i,:]
            new_state, r, u = bw_cell.call(current_step, last_state)
            out_layer_state.append(new_state)
            out_layer_reset.append(r)
            out_layer_update.append(u)
            last_state = new_state
        output['state_bw']  = [tf.transpose(tf.stack(out_layer_state), [1,0,2])]
        output['reset_bw']  = [tf.transpose(tf.stack(out_layer_reset), [1,0,2])]
        output['update_bw'] = [tf.transpose(tf.stack(out_layer_update),[1,0,2])]
        # layer 2
        bw_cell.build(self._num_units, 'bw2')
        out_layer_state  = []
        out_layer_reset  = []
        out_layer_update = []
        for i in range(num_step):
            current_step = output['state_bw'][0][:,i,:]
            new_state, r, u = bw_cell.call(current_step, last_state)
            out_layer_state.append(new_state)
            out_layer_reset.append(r)
            out_layer_update.append(u)
            last_state = new_state
        out_layer_state  = tf.transpose(tf.stack(out_layer_state), [1,0,2])
        out_layer_reset  = tf.transpose(tf.stack(out_layer_reset), [1,0,2])
        out_layer_update = tf.transpose(tf.stack(out_layer_update),[1,0,2])
        output['state_bw'].append(out_layer_state)
        output['reset_bw'].append(out_layer_reset)
        output['update_bw'].append(out_layer_update)
        
        return output
    