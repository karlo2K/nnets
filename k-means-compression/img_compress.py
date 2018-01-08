import time

import numpy as np
from skimage.io import imread, imsave
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


def compress(img_path, n_clusters, n_jobs=-1, n_random=1000, debug=False):
    img = imread(img_path)
    rows, cols, depth = img.shape
    if debug:
        print("Img info\n\tw:{} h:{} d:{}", cols, rows, depth)

    img_vect = img.reshape(rows * cols, depth)
    # take n random pixels
    img_train = shuffle(img_vect)[:n_random]

    kmeans = KMeans(n_clusters=n_clusters, n_jobs=n_jobs)

    start = time.time()

    # train k-means using only n random pixels
    kmeans.fit(img_train)
    labels = kmeans.predict(img_vect)
    labels = labels.reshape(rows, cols)

    print('Total time: %.2f' % (time.time() - start))
    if debug:
        print('centers: ', kmeans.cluster_centers_)
        print('labels: ', labels)

    np.save('cmprs_cntrs.npy', kmeans.cluster_centers_)
    imsave('cmprs_lbls.png', labels)

    return 'cmprs_cntrs.npy', 'cmprs_lbls.png'
