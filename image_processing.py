import numpy as np

def apply_pixelation(image, pixel_size=15):
    pixelated_image = image.copy()
    for y in range(0, image.shape[0], pixel_size):
        for x in range(0, image.shape[1], pixel_size):
            block = pixelated_image[y:y + pixel_size, x:x + pixel_size]
            avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
            pixelated_image[y:y + pixel_size, x:x + pixel_size] = avg_color
    return pixelated_image
