import matplotlib.pyplot as plt
import numpy as np
from imageio import imread

from img_compress import compress
from img_decompress import decompress

img = 'face.png'
cluster_count = np.arange(15, 128, 1)
iter = 5

original_img = imread(img)
h, w, d = original_img.shape

errors = [0.0] * len(cluster_count)
times = [0.0] * len(cluster_count)

for it in np.arange(iter):
    print("iter:%d/%d" % (it + 1, iter))
    for i, c in enumerate(cluster_count):
        centers, labels, t = compress(img, n_clusters=c)
        decompress(centers, labels)

        reconstructed_img = imread('reconstructed.png')

        o = original_img.astype('int')
        r = reconstructed_img.astype('int')
        err = np.mean((o - r) ** 2)

        errors[i] += err / iter
        times[i] += t / iter

plt.subplot(2, 1, 1)
plt.plot(np.arange(0, len(cluster_count)), errors)
plt.title("Error")
plt.subplot(2, 1, 2)
plt.plot(np.arange(0, len(cluster_count)), times)
plt.title("Time")
plt.show()
