import numpy as np
from skimage.io import imread, imsave


def decompress(centers, labels, debug=False):
    centers = np.load(centers)
    labels = imread(labels)

    if debug:
        print('centers: ', centers)
        print('labels: ', labels)

    image = np.zeros((labels.shape[0], labels.shape[1], centers.shape[1]), dtype=np.uint8)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            image[i, j, :] = centers[labels[i, j], :]

    imsave('reconstructed.png', image)
