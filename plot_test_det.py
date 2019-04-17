import glob, os
import matplotlib.pyplot as plt
import numpy as np

eval_events = 'evaluate_Events'
eval_noise  = 'evaluate_Noise'
f = open(eval_events); events_lines = f.readlines(); f.close()
f = open(eval_noise);  noise_lines  = f.readlines(); f.close()

def xy_reader(lines):
  x, y, dy = np.array([]), np.array([]), np.array([])
  for line in lines:
    xi, yi, dyi = line.split(',')
    x = np.append(x, float(xi))
    y = np.append(y, float(yi))
    dy= np.append(dy, float(dyi))
  return np.array([x, y, dy])

events_accuracy = xy_reader(events_lines)
noise_accuracy  = xy_reader(noise_lines)

events_accuracy = events_accuracy[:,:]
noise_accuracy  = noise_accuracy[:,:]

plt.figure()
plt.plot(events_accuracy[0], events_accuracy[1], label='events')
plt.plot(noise_accuracy[0],  noise_accuracy[1],  label='noise')

plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Test Accuracy of CNN')
plt.legend()

plt.tight_layout()
plt.show()
