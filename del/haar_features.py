import numpy as np

class HaarFeature:
    def __init__(self, x, y, width, height, feature_type):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.feature_type = feature_type

    def compute_value(self, integral_image):
        if self.feature_type == 'two_horizontal':
            left_sum = (integral_image[self.y + self.height, self.x + self.width // 2] -
                        integral_image[self.y + self.height, self.x] -
                        integral_image[self.y, self.x + self.width // 2] +
                        integral_image[self.y, self.x])
            right_sum = (integral_image[self.y + self.height, self.x + self.width] -
                         integral_image[self.y + self.height, self.x + self.width // 2] -
                         integral_image[self.y, self.x + self.width] +
                         integral_image[self.y, self.x + self.width // 2])
            return left_sum - right_sum
        height, width = integral_image.shape
        if (self.y + self.height) >= height or (self.x + self.width) >= width:
            return 0
        return 0  # Add more feature types as needed

def compute_integral_image(image):
    image = image.astype(np.uint8)  # Ensure uint8
    height, width = image.shape
    if np.any(np.isnan(image)) or np.any(np.isinf(image)):
        print("Warning: Input image contains NaN or infinite values.")
    integral_image = np.zeros((height + 1, width + 1), dtype=np.int64)
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            integral_image[y, x] = (image[y-1, x-1] +
                                    integral_image[y, x-1] +
                                    integral_image[y-1, x] -
                                    integral_image[y-1, x-1])
    if np.any(np.isnan(integral_image)) or np.any(np.isinf(integral_image)):
        print("Warning: Integral image contains NaN or infinite values.")
    return integral_image

def generate_haar_features(window_size=24):
    features = []
    for feature_type in ['two_horizontal']:
        for width in range(2, window_size + 1, 2):
            for height in range(1, window_size + 1):
                for x in range(window_size - width + 1):
                    for y in range(window_size - height + 1):
                        features.append(HaarFeature(x, y, width, height, feature_type))
    print(f"Generated {len(features)} Haar features.")
    return features