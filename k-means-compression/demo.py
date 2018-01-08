from img_compress import compress
from img_decompress import decompress
from imageio import imread
import matplotlib.pyplot as plt

img = 'face.png'

centers, labels = compress(img, n_clusters=64)
decompress(centers, labels)

original_img = imread(img)
reconstructed_img = imread('reconstructed.png')

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_img)
plt.title('Reconstruction')
plt.axis('off')

plt.show()
