import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
# print(digits.DESCR)
# print(digits.data)
# print(digits.target)
# plt.gray()
# plt.matshow(digits.images[100])
# plt.show()
# print(digits.target[100])
# fig = plt.figure(figsize=(6,6))
# fig.subplots_adjust(left = 0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# for i in range(64):
#   ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
#   ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
#   ax.text(0, 7, str(digits.target[i]))

# plt.show()

n_clusters = [k for k in range(1, 15)]
inertias = []
for k in n_clusters:
  model = KMeans(n_clusters = k)
  model.fit(digits.data)
  inertias.append(model.inertia_)

plt.plot(n_clusters, inertias)
plt.show()

new_samples = np.array([
[0.00,0.00,3.05,7.40,5.42,4.27,0.23,0.00,0.00,1.84,7.63,5.88,5.50,7.63,6.25,0.00,0.00,0.69,3.59,0.23,0.31,4.81,7.63,0.46,0.00,0.00,0.00,0.00,0.00,1.07,3.59,0.15,0.00,0.00,0.00,0.00,0.00,1.29,5.11,0.31,0.00,0.00,0.00,1.15,1.15,3.89,7.63,0.38,0.00,0.00,0.00,6.94,7.63,7.40,6.41,0.99,0.00,0.00,0.00,6.26,7.63,7.40,7.63,7.63],
[0.00,0.00,1.15,3.66,2.59,1.60,0.00,0.00,0.00,0.00,6.03,7.63,7.63,7.63,4.81,0.46,0.00,1.22,7.63,3.52,1.15,4.28,7.47,6.48,0.00,2.21,7.63,1.60,0.00,0.00,3.89,7.63,0.00,1.07,7.63,3.67,0.61,4.50,7.63,6.72,0.00,0.00,6.18,7.55,7.18,7.55,4.50,0.61,0.00,2.06,7.63,6.87,6.03,1.37,0.00,0.00,0.00,0.69,3.59,0.15,0.00,0.00,0.00,0.00],
[0.00,0.00,0.46,3.51,5.95,4.26,2.89,1.15,0.00,0.00,2.44,7.63,6.17,7.32,7.63,5.11,0.00,0.00,2.60,7.63,4.50,3.96,5.95,4.35,0.00,0.00,0.38,7.02,7.63,7.62,7.63,7.62,0.00,2.21,6.41,7.63,7.63,6.79,7.48,5.95,0.00,6.63,7.63,5.34,3.97,1.53,7.64,3.68,0.00,1.30,6.69,7.63,7.63,7.63,7.63,1.75,0.00,0.00,0.69,3.74,2.75,2.90,3.58,0.15],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.90,1.30,0.00,1.38,1.98,0.00,0.00,0.31,7.10,4.43,0.00,5.12,6.11,0.00,0.00,2.37,7.63,2.37,0.00,5.87,5.34,0.00,0.00,4.89,7.40,0.31,0.38,7.48,5.04,3.36,0.00,4.20,7.63,6.11,5.27,7.63,7.63,7.63,0.00,0.00,2.82,5.95,7.02,7.48,2.75,1.53,0.00,0.00,0.00,0.00,3.66,5.26,0.00,0.00]
])

model = KMeans(n_clusters = 10)
model.fit(digits.data)

new_labels = model.predict(new_samples)

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):
  ax = fig.add_subplot(2, 5, i +1)
  ax.imshow(model.cluster_centers_[i].reshape((8,8)), cmap=plt.cm.binary)

plt.show()

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')





