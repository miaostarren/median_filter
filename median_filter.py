import cv2
import matplotlib.pyplot as plt
import numpy as np


def add_salt_and_pepper_noise(img, amount=0.02):
    # Make a copy of the original image to avoid direct modification
    noisy_image = np.copy(img)
    h, w = noisy_image.shape

    # Calculate the number of noise pixels
    num_noise_pixels = int(amount * h * w)

    # Add salt-and-pepper noise at random positions
    for _ in range(num_noise_pixels):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        if np.random.rand() < 0.5:
            noisy_image[y, x] = 0  # Add black noise
        else:
            noisy_image[y, x] = 255  # Add white noise
    return noisy_image


def median_filter(image, kernel_size=3):
    h, w = image.shape
    new_image = np.copy(image)

    # Calculate padding size
    padding = kernel_size // 2

    # Apply median filter to each channel of the image
    for y in range(padding, h - padding):
        for x in range(padding, w - padding):
            region = image[y - padding : y + padding + 1, x - padding : x + padding + 1]
            new_image[y, x] = np.median(region)

    return new_image


if __name__ == '__main__':
    img = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)

    # add salt and pepper noise to source image
    noisy_img = add_salt_and_pepper_noise(img, amount=0.02)

    # median filter
    fitered_image = median_filter(noisy_img, kernel_size=3)

    # show images
    plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(img, cmap='gray')
    plt.title("Original Image")
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(noisy_img, cmap='gray')
    plt.title("Noisy image")
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(fitered_image, cmap='gray')
    plt.title("Filtered image")
    plt.show()
    plt.savefig("output.png")
