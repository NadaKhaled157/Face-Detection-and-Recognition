import numpy as np
from haar_features import HaarFeature, compute_integral_image


def train_weak_classifier(feature, images, labels, weights):
    values = np.array([feature.compute_value(compute_integral_image(img)) for img in images])
    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        print(f"Feature {feature}: Invalid values in compute_value: {values}")
        return None

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

    if best_error == float('inf'):
        print(f"Feature {feature}: No valid threshold found.")
        return None

    return {'feature': feature, 'threshold': best_threshold, 'polarity': best_polarity, 'error': best_error}


def adaboost_train(images, labels, features, num_iterations=10):
    n_samples = len(images)
    if n_samples == 0:
        raise ValueError("No training images provided.")
    if len(labels) != n_samples:
        raise ValueError(f"Mismatch: {len(labels)} labels, {n_samples} images.")
    if not features:
        raise ValueError("No Haar features provided.")

    weights = np.ones(n_samples) / n_samples
    classifiers = []

    print(f"Training with {n_samples} images, {len(features)} features, {num_iterations} iterations.")

    for iteration in range(num_iterations):
        best_classifier = None
        min_error = float('inf')
        print(f"Iteration {iteration + 1}/{num_iterations}")
        for i, feature in enumerate(features):
            classifier = train_weak_classifier(feature, images, labels, weights)
            if classifier is None or 'error' not in classifier:
                continue
            if np.isnan(classifier['error']) or np.isinf(classifier['error']):
                print(f"Feature {i}: Invalid error value {classifier['error']}.")
                continue
            if classifier['error'] < min_error:
                min_error = classifier['error']
                best_classifier = classifier
            if i % 100 == 0:
                print(f"Processed {i}/{len(features)} features, current min_error: {min_error}")

        if best_classifier is None:
            print(f"Warning: No valid classifier found in iteration {iteration + 1}, skipping.")
            continue

        alpha = 0.5 * np.log((1 - min_error) / (max(min_error, 1e-10)))
        for i in range(n_samples):
            value = best_classifier['feature'].compute_value(compute_integral_image(images[i]))
            prediction = 1 if best_classifier['polarity'] * value < best_classifier['polarity'] * best_classifier[
                'threshold'] else 0
            if prediction != labels[i]:
                weights[i] *= np.exp(alpha)
            else:
                weights[i] *= np.exp(-alpha)
        weights /= np.sum(weights)

        classifiers.append({**best_classifier, 'alpha': alpha})

    if not classifiers:
        raise ValueError("No classifiers trained.")
    print(f"Training completed. Selected {len(classifiers)} classifiers.")
    return classifiers


def cascade_classify(window, classifiers, stage_thresholds):
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
    detections = []
    height, width = image.shape
    integral_image = compute_integral_image(image)

    scale = 1.0
    while window_size * scale <= min(height, width):
        scaled_window = int(window_size * scale)
        step = max(1, int(scaled_window * 0.1))
        for y in range(0, height - scaled_window, step):
            for x in range(0, width - scaled_window, step):
                window = integral_image[y:y + scaled_window + 1, x:x + scaled_window + 1]
                if cascade_classify(window, classifiers, stage_thresholds):
                    detections.append((x, y, scaled_window))
        scale *= scale_factor

    return detections


def non_max_suppression(detections, threshold=0.3):
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