import cv2
import numpy as np

def white_balance(image):
    # Converting the image to float32 for accurate calculations
    image_float = image.astype(np.float32)

    # Calculating the average value for each color channel
    avg_r = np.mean(image_float[:, :, 2])
    avg_g = np.mean(image_float[:, :, 1])
    avg_b = np.mean(image_float[:, :, 0])

    # Calculating the scaling factors for channel
    scale_r = avg_g / avg_r
    scale_b = avg_g / avg_b

    # Applying the scaling factors to the color channels
    balanced_image = image_float.copy()
    balanced_image[:, :, 2] *= scale_r
    balanced_image[:, :, 0] *= scale_b

    # Clipping the values to the valid 8-bit range [0, 255]
    balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)

    return balanced_image

# Uploading an image
input_image = cv2.imread('images 4.jpg')

# Applying white balance correction in the image
balanced_image = white_balance(input_image)

# Lastly displaying the original and balanced images
cv2.imshow('Original Image', input_image)
cv2.imshow('Balanced Image', balanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
 