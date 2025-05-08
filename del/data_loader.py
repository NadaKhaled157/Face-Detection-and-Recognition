import os
import cv2
import numpy as np

def get_label_filename(subject_num, img_num):
    """
    Map subject and image number to label filename (e.g., s01, 1 -> lab001).
    """
    base_idx = (subject_num - 1) * 20 + 1
    label_idx = base_idx + (img_num - 1)
    return f"lab{label_idx:03d}"

def load_positive_examples(images_root, labels_folder):
    """
    Load face images and crop face regions using individual label files.
    """
    train_images = []
    train_labels = []

    for person_dir in os.listdir(images_root):
        person_path = os.path.join(images_root, person_dir)
        if not os.path.isdir(person_path):
            continue

        try:
            subject_num = int(person_dir.replace('s', ''))
        except ValueError:
            print(f"Warning: Invalid directory name {person_dir}, skipping.")
            continue

        for img_name in os.listdir(person_path):
            if not img_name.endswith('.jpg'):
                continue

            img_path = os.path.join(person_path, img_name)
            try:
                img_num = int(img_name.replace('.jpg', ''))
            except ValueError:
                print(f"Warning: Invalid image name {img_name}, skipping.")
                continue

            label_name = get_label_filename(subject_num, img_num)
            label_path = os.path.join(labels_folder, label_name)

            if not os.path.exists(label_path):
                print(f"Warning: Label {label_name} not found for {img_path}, skipping.")
                continue

            with open(label_path, 'r') as f:
                try:
                    x_left, y_top, x_right, y_bottom, person_id = f.read().strip().split()
                    x_left, y_top, x_right, y_bottom = map(int, [x_left, y_top, x_right, y_bottom])
                    if x_right <= x_left or y_bottom <= y_top:
                        print(f"Warning: Invalid bounding box for {img_path}, skipping.")
                        continue
                except ValueError:
                    print(f"Warning: Malformed label file {label_path}, skipping.")
                    continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load {img_path}, skipping.")
                continue

            face = img[y_top:y_bottom, x_left:x_right]
            if face.size == 0:
                print(f"Warning: Invalid bounding box for {img_path}, skipping.")
                continue

            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (24, 24))
            face_resized = cv2.equalizeHist(face_resized)

            if np.any(np.isnan(face_resized)) or np.any(np.isinf(face_resized)):
                print(f"Warning: Invalid pixel values in {img_path}, skipping.")
                continue

            train_images.append(face_resized)
            train_labels.append(1)

    print(f"Loaded {len(train_images)} positive examples.")
    return train_images, train_labels

def generate_negative_examples(images_root, labels_folder, num_negatives):
    """
    Generate non-face patches from image backgrounds.
    """
    neg_images = []
    neg_labels = []

    for person_dir in os.listdir(images_root):
        person_path = os.path.join(images_root, person_dir)
        if not os.path.isdir(person_path):
            continue

        try:
            subject_num = int(person_dir.replace('s', ''))
        except ValueError:
            continue

        for img_name in os.listdir(person_path):
            if not img_name.endswith('.jpg'):
                continue

            img_path = os.path.join(person_path, img_name)
            try:
                img_num = int(img_name.replace('.jpg', ''))
            except ValueError:
                continue

            label_name = get_label_filename(subject_num, img_num)
            label_path = os.path.join(labels_folder, label_name)

            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                try:
                    x_left, y_top, x_right, y_bottom, person_id = f.read().strip().split()
                    x_left, y_top, x_right, y_bottom = map(int, [x_left, y_top, x_right, y_bottom])
                    if x_right <= x_left or y_bottom <= y_top:
                        print(f"Warning: Invalid bounding box for {img_path}, skipping.")
                        continue
                except ValueError:
                    print(f"Warning: Malformed label file {label_path}, skipping.")
                    continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            height, width = img.shape[:2]
            face_width = x_right - x_left
            face_height = y_bottom - y_top

            if face_width <= 0 or face_height <= 0:
                print(f"Warning: Invalid face dimensions for {img_path}, skipping.")
                continue

            for _ in range(5):
                if len(neg_images) >= num_negatives:
                    break

                try:
                    patch_width = np.random.randint(face_width // 2, face_width * 2)
                    patch_height = np.random.randint(face_height // 2, face_height * 2)
                except ValueError as e:
                    print(f"Warning: Invalid patch size for {img_path} (face_width={face_width}, face_height={face_height}), skipping.")
                    continue

                px = np.random.randint(0, width - patch_width)
                py = np.random.randint(0, height - patch_height)

                if (px + patch_width < x_left or px > x_right or
                    py + patch_height < y_top or py > y_bottom):
                    patch = img[py:py + patch_height, px:px + patch_width]
                    if patch.size == 0:
                        continue

                    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    patch_resized = cv2.resize(patch_gray, (24, 24))
                    patch_resized = cv2.equalizeHist(patch_resized)

                    if np.any(np.isnan(patch_resized)) or np.any(np.isinf(patch_resized)):
                        continue

                    neg_images.append(patch_resized)
                    neg_labels.append(0)

    print(f"Generated {len(neg_images)} negative examples.")
    return neg_images, neg_labels

def load_georgia_tech_dataset(images_root, labels_folder):
    """
    Load and preprocess the Georgia Tech dataset.
    """
    try:
        pos_images, pos_labels = load_positive_examples(images_root, labels_folder)
        if not pos_images:
            raise ValueError("No positive examples loaded.")

        neg_images, neg_labels = generate_negative_examples(images_root, labels_folder, num_negatives=len(pos_images))

        train_images = pos_images + neg_images
        train_labels = pos_labels + neg_labels

        train_images = np.array(train_images)
        train_labels = np.array(train_labels)

        if train_images.size == 0:
            raise ValueError("No training images generated.")

        print(f"Final dataset: {train_images.shape} images, {train_labels.shape} labels.")
        print(f"Face count: {np.sum(train_labels)}, Non-face count: {len(train_labels) - np.sum(train_labels)}")
        return train_images, train_labels
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")