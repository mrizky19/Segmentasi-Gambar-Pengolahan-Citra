import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def segment_image(image, k, max_iter=100, epsilon=0.85):
    # Ubah warna gambar dari RGB ke BGR untuk cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Membentuk ulang gambar menjadi susunan piksel 2D dan 3 nilai warna (RGB)
    pixel_vals = image.reshape((-1, 3))

    # Mengkonversikan ke tipe float
    pixel_vals = np.float32(pixel_vals)

    # Menentukan kriteria agar algoritme berhenti berjalan
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

    # Melakukan k-means clustering
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Mengonversi data menjadi nilai 8-bit
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # Membentuk ulang data menjadi dimensi gambar asli
    segmented_image = segmented_data.reshape((image.shape))

    # Ubah kembali warna gambar dari BGR ke RGB untuk ditampilkan
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    return segmented_image

def plot_histogram(image, title):
    color = ('r', 'g', 'b')
    plt.figure()
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
    plt.xlim([0, 256])

# Load image
image_path = 'image.jpg'  # Ganti dengan path gambar Anda
image = np.array(Image.open(image_path))

# Segmentasi gambar
k = 3  # Jumlah cluster
segmented_image = segment_image(image, k)

# Menampilkan gambar asli
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Gambar Asli')
plt.axis('off')

# Menampilkan gambar yang telah disegmentasi
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Gambar Tersegmentasi')
plt.axis('off')
plt.show()

# Plot histogram untuk gambar asli
plot_histogram(image, "Histogram Gambar Asli")
plt.show()

# Plot histogram untuk gambar yang telah disegmentasi
plot_histogram(segmented_image, "Histogram Gambar Tersegmentasi")
plt.show()
