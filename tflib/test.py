import numpy as np
import GRU
cell = GRU.GRUCell(1)
a=np.array([[1.,2.,3.]], dtype=np.float32)
b=np.array([[1.]], dtype=np.float32)

cell.build([1,3], 'fw')
cell.call(a, b)

