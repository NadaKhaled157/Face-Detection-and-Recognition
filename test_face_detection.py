import numpy as np
from PIL import Image, ImageDraw
import os

# Dataset paths (modify as needed)
# base_path = r"D:\College\Year Three\Second Term\Computer Vision\Face-Detection-and-Recognition\data\train"
base_path = r"data\train"
label_folder = os.path.join(base_path, "labels")
face_folders = [os.path.join(base_path, f"images/s0{i}") for i in range(1, 5)]  # s01 to s04
# non_face_folder = r"D:\College\Year Three\Second Term\Computer Vision\Face-Detection-and-Recognition\data\train\non_faces"
non_face_folder = r"data\train\non_faces"
output_dir = "output_faces"
img_size = 64
pixel_count = img_size * img_size  # 4096 pixels


# Function to load images and labels
def load_face_images_and_labels(face_folders, label_folder, img_size=64, train=False):
    images = []
    labels = []
    bboxes = []
    image_paths = []

    for folder in face_folders:
        folder_id = os.path.basename(folder)
        folder_num = int(folder_id[1:])
        img_indices = range(11, 16)  # Test images

        for i in img_indices:
            img_name = f"{i:02d}.jpg"
            img_path = os.path.join(folder, img_name)
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            img = Image.open(img_path).convert('L')
            orig_width, orig_height = img.size
            img = img.resize((img_size, img_size))
            img_array = np.array(img).flatten()

            label_id = (folder_num - 1) * 20 + i
            label_file = os.path.join(label_folder, f"lab{label_id:03d}")
            if not os.path.exists(label_file):
                print(f"Label not found: {label_file}")
                continue

            with open(label_file, 'r') as f:
                line = f.read().strip()
                parts = line.split()
                if len(parts) != 5:
                    print(f"Invalid label format in {label_file}")
                    continue
                x_min, y_min, x_max, y_max, _ = parts
                bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]

                x_scale = img_size / orig_width
                y_scale = img_size / orig_height
                scaled_bbox = [
                    bbox[0] * x_scale,
                    bbox[1] * y_scale,
                    bbox[2] * x_scale,
                    bbox[3] * y_scale
                ]

            images.append(img_array)
            labels.append(1)
            bboxes.append(scaled_bbox)
            image_paths.append((img_path, img))

    return np.array(images), np.array(labels), bboxes, image_paths


# Function to load non-face images
def load_non_face_images(non_face_folder, img_size=64, num_images=20, train=False):
    images = []
    labels = []
    image_paths = []

    if not os.path.exists(non_face_folder):
        print(f"Non-face folder not found: {non_face_folder}. Generating synthetic non-face data.")
        images = np.random.normal(loc=50, scale=30, size=(num_images, img_size * img_size))
        labels = [0] * num_images
        for i in range(num_images):
            img_array = images[i].reshape(img_size, img_size)
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            image_paths.append((f"synthetic_nonface_{i}.png", img))
        return images, np.array(labels), [None] * num_images, image_paths

    all_files = os.listdir(non_face_folder)
    selected_files = all_files[num_images:num_images * 2]

    for filename in selected_files:
        img_path = os.path.join(non_face_folder, filename)
        img = Image.open(img_path).convert('L').resize((img_size, img_size))
        img_array = np.array(img).flatten()
        images.append(img_array)
        labels.append(0)
        image_paths.append((img_path, img))

    return np.array(images), np.array(labels), [None] * len(images), image_paths


# Load test data
test_face_images, test_face_labels, test_face_bboxes, test_face_paths = load_face_images_and_labels(face_folders,
                                                                                                    label_folder)
test_non_face_images, test_non_face_labels, test_non_face_bboxes, test_non_face_paths = load_non_face_images(
    non_face_folder)
X_test = np.vstack((test_face_images, test_non_face_images))
y_test = np.concatenate((test_face_labels, test_non_face_labels))
test_paths = test_face_paths + test_non_face_paths
test_bboxes = test_face_bboxes + test_non_face_bboxes
num_test = len(y_test)

# Load PCA model
model_data = np.load("pca_model.npz")
mean_image = model_data["mean_image"]
eigenvectors = model_data["eigenvectors"]
mean_face_pca = model_data["mean_face_pca"]
mean_non_face_pca = model_data["mean_non_face_pca"]

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Project test images and classify
X_test_centered = X_test - mean_image
X_test_pca = np.dot(X_test_centered, eigenvectors)

# Classify test images and draw boundaries
predictions = []
for i in range(num_test):
    dist_to_face = np.sqrt(np.sum((X_test_pca[i] - mean_face_pca) ** 2))
    dist_to_non_face = np.sqrt(np.sum((X_test_pca[i] - mean_non_face_pca) ** 2))
    is_face = dist_to_face < dist_to_non_face
    predictions.append(1 if is_face else 0)

    img_path, img = test_paths[i]
    img = img.copy()
    draw = ImageDraw.Draw(img)

    bbox = test_bboxes[i]
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        draw.rectangle((x_min, y_min, x_max, y_max), width=2)

    # if is_face:
    #     center_x, center_y = img_size // 2, img_size // 2
    #     half_size = 20
    #     pca_bbox = [
    #         center_x - half_size,
    #         center_y - half_size,
    #         center_x + half_size,
    #         center_y + half_size
    #     ]
    #     draw.rectangle(pca_bbox, outline=(0, 255, 0), width=2)

    output_filename = f"test_image_{i}_{os.path.basename(img_path)}"
    output_path = os.path.join(output_dir, output_filename)
    img.save(output_path)

# Evaluate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Predictions: {predictions}")
print(f"True Labels: {y_test}")
print(f"Images saved to: {output_dir}")
