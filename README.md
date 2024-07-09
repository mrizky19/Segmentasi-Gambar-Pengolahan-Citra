# Segmentasi Gambar Menggunakan K-Means Clustering

Repository ini berisi skrip Python untuk melakukan segmentasi gambar menggunakan K-Means clustering. Skrip ini memproses gambar input untuk mensegmentasinya ke dalam `k` cluster, dan kemudian menampilkan gambar asli, gambar yang telah disegmentasi, dan histogram warna untuk kedua gambar.

## Kebutuhan

Pastikan Anda telah menginstal paket-paket berikut:

- `numpy`
- `opencv-python`
- `matplotlib`
- `Pillow`

Anda dapat menginstal paket-paket tersebut menggunakan pip:

```bash
pip install numpy opencv-python matplotlib pillow
```

## Penggunaan

1. **Clone repository ini:**

    ```bash
    git clone https://github.com/username/image-segmentation-kmeans.git
    cd image-segmentation-kmeans
    ```

2. **Letakkan gambar input Anda di direktori repository.**

3. **Perbarui variabel `image_path` di dalam skrip dengan path ke file gambar Anda:**

    ```python
    image_path = 'path_to_your_image.jpg'  # Ganti dengan path gambar Anda
    ```

4. **Jalankan skrip:**

    ```bash
    python segment_image.py
    ```

5. **Skrip akan menampilkan gambar asli, gambar yang telah disegmentasi, dan histogram untuk kedua gambar.**

## Contoh

### Gambar Asli
<img src="https://github.com/mrizky19/Segmentasi-Gambar-Pengolahan-Citra/assets/94947436/168b5519-744d-470f-9a13-ff51b078dfce" alt="Halaman Beranda" width="300">


### Gambar Tersegmentasi
![Segmentasi](https://github.com/mrizky19/Segmentasi-Gambar-Pengolahan-Citra/assets/94947436/26fd1b1c-2eef-480d-a592-ab382f98cdc6)


### Histogram untuk Gambar Asli
![Figure_1](https://github.com/mrizky19/Segmentasi-Gambar-Pengolahan-Citra/assets/94947436/1191291d-e9be-4cc1-8d5b-b8a1419096ef)


### Histogram untuk Gambar Tersegmentasi
![Figure_2](https://github.com/mrizky19/Segmentasi-Gambar-Pengolahan-Citra/assets/94947436/7c515059-43d8-4b43-8b36-aa49370500f1)


## Kode

Berikut adalah kode utama yang digunakan untuk segmentasi gambar dan plotting:

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def segment_image(image, k, max_iter=100, epsilon=0.85):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))
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

image_path = 'path_to_your_image.jpg'
image = np.array(Image.open(image_path))
k = 3
segmented_image = segment_image(image, k)

plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Gambar Asli')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(segmented_image)
plt.title('Gambar Tersegmentasi')
plt.axis('off')

plt.subplot(2, 2, 3)
plot_histogram(image, "Histogram Gambar Asli")

plt.subplot(2, 2, 4)
plot_histogram(segmented_image, "Histogram Gambar Tersegmentasi")

plt.tight_layout()
plt.show()
```
