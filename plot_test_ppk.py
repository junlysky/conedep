import glob, os
import matplotlib.pyplot as plt
import numpy as np

eval_events = 'evaluate_ppk'
f = open(eval_events); ppk_lines = f.readlines(); f.close()

def xy_reader(lines):
  x, y, dy = np.array([]), np.array([]), np.array([])
  for line in lines:
    xi, yi, dyi = line.split(',')
    x = np.append(x, float(xi))
    y = np.append(y, float(yi))
    dy= np.append(dy, float(dyi))
  return np.array([x, y, dy])

ppk_error = xy_reader(ppk_lines)
ppk_error = ppk_error[:,:]

plt.figure()
plt.plot(ppk_error[0], ppk_error[1])

plt.xlabel('epoch')
plt.ylabel('error rate')
plt.title('Test Error Rate of RNN')
plt.show()
