import os
import cv2
import numpy as np


def load_positive_examples(images_root, labels_file):
    """
    Load face images from the Georgia Tech Face Database and crop face regions.
    Args:
        images_root (str): Path to root folder with subdirectories s1, s2, ..., s50.
        labels_file (str): Path to a single label file with bounding boxes.
    Returns:
        train_images (list): List of 24x24 grayscale face patches.
        train_labels (list): List of labels (1 for faces).
    """
    train_images = []
    train_labels = []

    # Read label file (assumes format: image_path x_left y_top x_right y_bottom person_id)
    with open(labels_file, 'r') as f:
        labels = [line.strip().split() for line in f if line.strip()]

    for label in labels:
        # Extract label info
        img_rel_path, x_left, y_top, x_right, y_bottom, person_id = label
        x_left, y_top, x_right, y_bottom = map(int, [x_left, y_top, x_right, y_bottom])

        # Construct full image path
        img_path = os.path.join(images_root, img_rel_path)

        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping.")
            continue

        # Read image in color
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load {img_path}, skipping.")
            continue

        # Crop face region
        face = img[y_top:y_bottom, x_left:x_right]
        if face.size == 0:
            print(f"Warning: Invalid bounding box for {img_path}, skipping.")
            continue

        # Convert to grayscale
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Resize to 24x24
        face_resized = cv2.resize(face_gray, (24, 24))

        # Optional: Histogram equalization
        face_resized = cv2.equalizeHist(face_resized)

        train_images.append(face_resized)
        train_labels.append(1)  # Label as face

    return train_images, train_labels


def generate_negative_examples(images_root, labels_file, num_negatives=750):
    """
    Generate non-face patches from image backgrounds.
    Args:
        images_root (str): Path to root folder with subdirectories s1, s2, ..., s50.
        labels_file (str): Path to label file with bounding boxes.
        num_negatives (int): Number of negative patches to generate.
    Returns:
        neg_images (list): List of 24x24 grayscale non-face patches.
        neg_labels (list): List of labels (0 for non-faces).
    """
    neg_images = []
    neg_labels = []

    # Read label file
    with open(labels_file, 'r') as f:
        labels = [line.strip().split() for line in f if line.strip()]

    # Shuffle labels to randomize image selection
    np.random.shuffle(labels)

    for label in labels:
        img_rel_path, x_left, y_top, x_right, y_bottom, person_id = label
        x_left, y_top, x_right, y_bottom = map(int, [x_left, y_top, x_right, y_bottom])

        img_path = os.path.join(images_root, img_rel_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        height, width = img.shape[:2]
        face_width = x_right - x_left
        face_height = y_bottom - y_top

        # Generate random patches outside the face region
        for _ in range(5):  # Try multiple patches per image
            if len(neg_images) >= num_negatives:
                break

            # Random patch size (similar to face size)
            patch_width = np.random.randint(face_width // 2, face_width * 2)
            patch_height = np.random.randint(face_height // 2, face_height * 2)

            # Random top-left corner
            px = np.random.randint(0, width - patch_width)
            py = np.random.randint(0, height - patch_height)

            # Check if patch overlaps with face
            if (px + patch_width < x_left or px > x_right or
                py + patch_height < y_top or py > y_bottom):
                patch = img[py:py + patch_height, px:px + patch_width]
                if patch.size == 0:
                    continue

                # Convert to grayscale
                patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

                # Resize to 24x24
                patch_resized = cv2.resize(patch_gray, (24, 24))
                patch_resized = cv2.equalizeHist(patch_resized)

                neg_images.append(patch_resized)
                neg_labels.append(0)

    return neg_images, neg_labels


def load_georgia_tech_dataset(images_root, labels_file):
    """
    Load and preprocess the Georgia Tech Face Database.
    Returns:
        train_images (np.array): Array of 24x24 grayscale patches.
        train_labels (np.array): Array of labels (1 for faces, 0 for non-faces).
    """
    # Load positive examples
    pos_images, pos_labels = load_positive_examples(images_root, labels_file)

    # Generate negative examples (same number as positive)
    neg_images, neg_labels = generate_negative_examples(images_root, labels_file, num_negatives=len(pos_images))

    # Combine
    train_images = pos_images + neg_images
    train_labels = pos_labels + neg_labels

    # Convert to NumPy arrays
    train_images = np.array(train_images)  # Shape: (N, 24, 24)
    train_labels = np.array(train_labels)  # Shape: (N,)

    print(f"Loaded {len(train_images)} images: {np.sum(train_labels)} faces, "
          f"{len(train_labels) - np.sum(train_labels)} non-faces.")
    return train_images, train_labels

# Example usage
images_root = r"C:\Users\nadak\Downloads\gt_db\gt_db"  # e.g., where s1, s2, ..., s50 are located
labels_file = r"C:\Users\nadak\Downloads\labels_gt\labels"  # Adjust to your label file
train_images, train_labels = load_georgia_tech_dataset(images_root, labels_file)
