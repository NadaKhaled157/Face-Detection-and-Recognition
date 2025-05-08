import numpy as np
import cv2


class HaarFeature:
    def __init__(self, x, y, width, height, feature_type):
        self.x = x  # Top-left corner x
        self.y = y  # Top-left corner y
        self.width = width
        self.height = height
        self.feature_type = feature_type  # e.g., 'two_horizontal', 'two_vertical'

    def compute_value(self, integral_image):
        """Compute the Haar feature value using the integral image."""
        if self.feature_type == 'two_horizontal':
            # Left rectangle: (x, y, w/2, h)
            left_sum = integral_image[self.y + self.height, self.x + self.width // 2] - \
                       integral_image[self.y + self.height, self.x] - \
                       integral_image[self.y, self.x + self.width // 2] + \
                       integral_image[self.y, self.x]
            # Right rectangle: (x + w/2, y, w/2, h)
            right_sum = integral_image[self.y + self.height, self.x + self.width] - \
                        integral_image[self.y + self.height, self.x + self.width // 2] - \
                        integral_image[self.y, self.x + self.width] + \
                        integral_image[self.y, self.x + self.width // 2]
            return left_sum - right_sum
        # Add other feature types (e.g., vertical, three-rectangle) similarly
        return 0


def compute_integral_image(image):
        """Compute the integral image of the input grayscale image."""
        height, width = image.shape
        integral_image = np.zeros((height + 1, width + 1), dtype=np.int64)

        for y in range(1, height + 1):
            for x in range(1, width + 1):
                integral_image[y, x] = image[y - 1, x - 1] + \
                                       integral_image[y, x - 1] + \
                                       integral_image[y - 1, x] - \
                                       integral_image[y - 1, x - 1]
        return integral_image


def generate_haar_features(window_size=24):
    """Generate all possible Haar-like features for a given window size."""
    features = []
    for feature_type in ['two_horizontal', 'two_vertical']:  # Add more types
        for width in range(2, window_size + 1, 2):  # Step by 2 for simplicity
            for height in range(1, window_size + 1):
                for x in range(window_size - width + 1):
                    for y in range(window_size - height + 1):
                        features.append(HaarFeature(x, y, width, height, feature_type))
    return features


def train_weak_classifier(feature, images, labels, weights):
    """Train a weak classifier for a single Haar feature."""
    values = np.array([feature.compute_value(compute_integral_image(img)) for img in images])
    best_error = float('inf')
    best_threshold = 0
    best_polarity = 1

    for threshold in np.unique(values):
        for polarity in [1, -1]:
            error = 0
            for i, (value, label, weight) in enumerate(zip(values, labels, weights)):
                prediction = 1 if polarity * value < polarity * threshold else 0
                if prediction != label:
                    error += weight
            if error < best_error:
                best_error = error
                best_threshold = threshold
                best_polarity = polarity

    return {'feature': feature, 'threshold': best_threshold, 'polarity': best_polarity, 'error': best_error}


def adaboost_train(images, labels, features, num_iterations=10):
    """Train a strong classifier using AdaBoost."""
    n_samples = len(images)
    weights = np.ones(n_samples) / n_samples
    classifiers = []

    for _ in range(num_iterations):
        best_classifier = None
        min_error = float('inf')
        for feature in features:
            classifier = train_weak_classifier(feature, images, labels, weights)
            if classifier['error'] < min_error:
                min_error = classifier['error']
                best_classifier = classifier

        # Update weights
        alpha = 0.5 * np.log((1 - min_error) / (max(min_error, 1e-10)))
        for i in range(n_samples):
            value = best_classifier['feature'].compute_value(compute_integral_image(images[i]))
            prediction = 1 if best_classifier['polarity'] * value < best_classifier['polarity'] * best_classifier['threshold'] else 0
            if prediction != labels[i]:
                weights[i] *= np.exp(alpha)
            else:
                weights[i] *= np.exp(-alpha)
        weights /= np.sum(weights)

        classifiers.append({**best_classifier, 'alpha': alpha})

    return classifiers


def cascade_classify(window, classifiers, stage_thresholds):
    """Classify a window using the cascade of classifiers."""
    for stage, threshold in enumerate(stage_thresholds):
        stage_sum = 0
        for classifier in classifiers[stage]:
            value = classifier['feature'].compute_value(window)
            prediction = 1 if classifier['polarity'] * value < classifier['polarity'] * classifier['threshold'] else 0
            stage_sum += classifier['alpha'] * prediction
        if stage_sum < threshold:
            return False
    return True


def detect_faces(image, classifiers, stage_thresholds, window_size=24, scale_factor=1.25):
    """Detect faces in the image using a sliding window."""
    detections = []
    height, width = image.shape
    integral_image = compute_integral_image(image)

    scale = 1.0
    while window_size * scale <= min(height, width):
        scaled_window = int(window_size * scale)
        step = max(1, int(scaled_window * 0.1))  # Step size for sliding
        for y in range(0, height - scaled_window, step):
            for x in range(0, width - scaled_window, step):
                window = integral_image[y:y+scaled_window+1, x:x+scaled_window+1]
                if cascade_classify(window, classifiers, stage_thresholds):
                    detections.append((x, y, scaled_window))
        scale *= scale_factor

    return detections


def non_max_suppression(detections, threshold=0.3):
    """Merge overlapping detections."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x[2])
    result = [detections[0]]
    for det in detections[1:]:
        x, y, w = det
        overlap = False
        for res in result:
            rx, ry, rw = res
            if abs(x - rx) < threshold * w and abs(y - ry) < threshold * w:
                overlap = True
                break
        if not overlap:
            result.append(det)
    return result

## EXAMPLE ##
def load_data_subfolders(faces_folder, non_faces_folder):
    train_images = []
    train_labels = []

    # Load face images
    for img_name in os.listdir(faces_folder):
        if not img_name.endswith(('.jpg', '.png')):
            continue
        img_path = os.path.join(faces_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (24, 24))
        img = cv2.equalizeHist(img)
        train_images.append(img)
        train_labels.append(1)

    # Load non-face images
    for img_name in os.listdir(non_faces_folder):
        if not img_name.endswith(('.jpg', '.png')):
            continue
        img_path = os.path.join(non_faces_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (24, 24))
        img = cv2.equalizeHist(img)
        train_images.append(img)
        train_labels.append(0)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    print(f"Loaded {len(train_images)} images with {np.sum(train_labels)} faces and {len(train_labels) - np.sum(train_labels)} non-faces.")
    return train_images, train_labels

# Example usage
faces_folder = 'path/to/images/faces/'
non_faces_folder = 'path/to/images/non_faces/'
train_images, train_labels = load_data_subfolders(faces_folder, non_faces_folder)

# Draw rectangles
for (x, y, w) in detections:
    cv2.rectangle(image, (x, y), (x+w, y+w), (0, 255, 0), 2)

# Save and display
cv2.imwrite('output_image.jpg', image)
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()