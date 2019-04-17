class GRUcell(object):
  def __init__(self, incoming, D_input, D_cell, 
                initializer=tf.contrib.layers.variance_scaling_initializer(), 
                init_h=None):
      # 属性
      self.incoming = incoming
      self.D_input = D_input
      self.D_cell = D_cell
      self.initializer = initializer
      self.type = 'gru'
      # 如果没有提供最初的hidden state，会初始为0
      if init_h is None:
          # If init_h is not provided, initialize it
          # the shape of init_h is [n_samples, D_cell]
          self.init_h = tf.matmul(self.incoming[0,:,:], tf.zeros([self.D_input, self.D_cell]))
          self.previous = self.init_h
      # 如果没有提供最初的hidden state，会初始为0
      self.rgate = self.Gate()
      self.ugate = self.Gate()
      self.cell = self.Gate()
      # 因为所有的gate都会乘以当前的输入和上一时刻的hidden state
      # 将矩阵concat在一起，计算后再逐一分离，加快运行速度
      # W_x的形状是[D_input, 3*D_cell]
      self.W_x = tf.concat(values=[self.rgate[0], self.ugate[0], self.cell[0]], axis=1)
      self.W_h = tf.concat(values=[self.rgate[1], self.ugate[1], self.cell[1]], axis=1)
      self.b = tf.concat(values=[self.rgate[2], self.ugate[2], self.cell[2]], axis=0)
      
      
  # 初始化gate的函数   
  def Gate(self, bias = 0.001):
      # Since we will use gate multiple times, let's code a class for reusing
      Wx = self.initializer([self.D_input, self.D_cell])
      Wh = self.initializer([self.D_cell, self.D_cell])
      b  = tf.Variable(tf.constant(bias, shape=[self.D_cell]),trainable=True) 
      return Wx, Wh, b
      
  # 大矩阵乘法运算完毕后，方便用于分离各个gate
  def Slice_W(self, x, n):
     # split W's after computing
      return x[:, n*self.D_cell:(n+1)*self.D_cell]
      
  # 每个time step需要运行的步骤
  def Step(self, prev_h, current_x):
      # 分两次，统一在concat成的大矩阵中完成gates所需要的计算
      Wx = tf.matmul(current_x, self.W_x) + self.b
      Wh = tf.matmul(prev_h, self.W_h)
      # 分离和组合reset gate
      r = tf.sigmoid(self.Slice_W(Wx, 0) + self.Slice_W(Wh, 0))
      # 分离和组合update gate
      u = tf.sigmoid(self.Slice_W(Wx, 1) + self.Slice_W(Wh, 1))
      # 分离和组合新的更新信息
      # 注意GRU中，在这一步就已经有reset gate的干涉了
      c = tf.tanh(self.Slice_W(Wx, 2) + r*self.Slice_W(Wh, 2))
      # 计算当前hidden state，GRU将LSTM中的input gate和output gate的合设成1，
      # 用update gate完成两者的工作
      current_h = (1-u)*prev_h + u*c
      return current_h, r, u