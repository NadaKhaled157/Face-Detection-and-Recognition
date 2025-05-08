# FOR TESTING #

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

def load_image_and_label(images_root, labels_folder, person_dir, img_name):
    """
    Load an image and its corresponding label.
    Returns: image, (x_left, y_top, x_right, y_bottom) or None if invalid.
    """
    img_path = os.path.join(images_root, person_dir, img_name)
    try:
        subject_num = int(person_dir.replace('s', ''))
        img_num = int(img_name.replace('.jpg', ''))
    except ValueError:
        print(f"Warning: Invalid name {person_dir}/{img_name}, skipping.")
        return None, None

    label_name = get_label_filename(subject_num, img_num)
    label_path = os.path.join(labels_folder, label_name)

    if not os.path.exists(label_path):
        print(f"Warning: Label {label_name} not found for {img_path}, skipping.")
        return None, None

    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Failed to load {img_path}, skipping.")
        return None, None

    with open(label_path, 'r') as f:
        try:
            x_left, y_top, x_right, y_bottom, _ = f.read().strip().split()
            x_left, y_top, x_right, y_bottom = map(int, [x_left, y_top, x_right, y_bottom])
            if x_right <= x_left or y_bottom <= y_top:
                print(f"Warning: Invalid bounding box for {img_path}, skipping.")
                return None, None
        except ValueError:
            print(f"Warning: Malformed label file {label_path}, skipping.")
            return None, None

    return img, (x_left, y_top, x_right, y_bottom)

def detect_faces_opencv(image, cascade):
    """
    Detect faces using OpenCV Haar Cascade.
    Returns: List of (x, y, w, h).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def main():
    # Paths
    images_root = 'data/images/'
    labels_folder = 'data/labels/'
    output_dir = 'del/output/'
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    # Load Haar Cascade
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print(f"Error: Failed to load Haar Cascade from {cascade_path}")
        return

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process images
    processed = 0
    for person_dir in os.listdir(images_root):
        person_path = os.path.join(images_root, person_dir)
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            if not img_name.endswith('.jpg'):
                continue

            # Load image and label
            img, gt_bbox = load_image_and_label(images_root, labels_folder, person_dir, img_name)
            if img is None or gt_bbox is None:
                continue

            # Detect faces
            detections = detect_faces_opencv(img, cascade)

            # Draw ground truth (blue)
            x_left, y_top, x_right, y_bottom = gt_bbox
            cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), (255, 0, 0), 2)

            # Draw detections (green)
            for (x, y, w, h) in detections:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save output
            output_path = os.path.join(output_dir, f"{person_dir}_{img_name}")
            cv2.imwrite(output_path, img)
            processed += 1

    print(f"Processed {processed} images. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()


