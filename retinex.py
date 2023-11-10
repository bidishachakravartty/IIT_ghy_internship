import cv2
import numpy as np

def multiscale_retinex(image, sigma_list):
    # Initializing an empty list to store the retinex results for each scale
    retinex_results = []

    # Converting the input image to float32 for numerical calculations
    image = np.float32(image)

    # Applying retinex at different scales
    for sigma in sigma_list:
        # Applying Gaussian smoothing with the given sigma to the original image
        image_blur = cv2.GaussianBlur(image, (0, 0), sigma)

        # Calculated the log-transformed version of the image
        log_image = np.log1p(image) - np.log1p(image_blur)

        # Appended the retinex result for this scale to the list
        retinex_results.append(log_image)

    # Combined the retinex results by taking the mean
    combined_retinex = np.mean(retinex_results, axis=0)

    # Normalized the result to the range [0, 255]
    max_value = np.max(combined_retinex)
    min_value = np.min(combined_retinex)
    combined_retinex = ((combined_retinex - min_value) / (max_value - min_value)) * 255

    # Converting the result back to uint8 for display
    combined_retinex = np.uint8(combined_retinex)

    return combined_retinex

# Uploading an image
input_image = cv2.imread('download1.jpg')

# Defining a list of sigma values for different scales
sigma_list = [17, 60, 245]

# Applied the Multiscale Retinex algorithm
result_image = multiscale_retinex(input_image, sigma_list)

# Display the original and enhanced images
cv2.imshow('Original Image', input_image)
cv2.imshow('Enhanced Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
