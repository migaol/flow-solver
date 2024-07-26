'''Compare time for different kernels on morphological operations & shape detection.'''

import cv2
import numpy as np
import time

if __name__ == '__main__':
    image = cv2.imread('../standard_rectangular/10x10mania-150.png', cv2.IMREAD_GRAYSCALE)

    kernels = {
        f'{i}x{i}': cv2.getStructuringElement(cv2.MORPH_RECT, (i,i)) for i in range(1,13,2)
    }

    for size, kernel in kernels.items():
        start_time = time.time()
        dilated_image = cv2.dilate(image, kernel, iterations=1)
        eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
        end_time = time.time()
        print(f"{size} kernel: {end_time - start_time:.5f} s")