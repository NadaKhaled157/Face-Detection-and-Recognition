import numpy as np
from PIL import Image
import os

# Set random seed for reproducibility
np.random.seed(42)

# Dataset paths (modify as needed)
# base_path = r"D:\College\Year Three\Second Term\Computer Vision\Face-Detection-and-Recognition\data\train"
base_path = r"\data\train"
label_folder = os.path.join(base_path, "labels")
face_folders = [os.path.join(base_path, f"images/s0{i}") for i in range(1, 5)]  # s01 to s04
# non_face_folder = r"D:\College\Year Three\Second Term\Computer Vision\Face-Detection-and-Recognition\data\train\non_faces"
non_face_folder = r"data\train\non_faces"
img_size = 64
pixel_count = img_size * img_size  # 4096 pixels


# Function to load images and labels
def load_face_images_and_labels(face_folders, label_folder, img_size=64, train=True):
    images = []
    labels = []
    bboxes = []
    image_paths = []

    for folder in face_folders:
        folder_id = os.path.basename(folder)  # e.g., 's01'
        folder_num = int(folder_id[1:])  # 1, 2, 3, 4
        img_indices = range(1, 11) if train else range(11, 16)

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
def load_non_face_images(non_face_folder, img_size=64, num_images=40, train=True):
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
    selected_files = all_files[:num_images] if train else all_files[num_images:num_images * 2]

    for filename in selected_files:
        img_path = os.path.join(non_face_folder, filename)
        img = Image.open(img_path).convert('L').resize((img_size, img_size))
        img_array = np.array(img).flatten()
        images.append(img_array)
        labels.append(0)
        image_paths.append((img_path, img))

    return np.array(images), np.array(labels), [None] * len(images), image_paths


# Load training data
face_images, face_labels, face_bboxes, face_paths = load_face_images_and_labels(face_folders, label_folder, train=True)
non_face_images, non_face_labels, non_face_bboxes, non_face_paths = load_non_face_images(non_face_folder, num_images=40)
X_train = np.vstack((face_images, non_face_images))
y_train = np.concatenate((face_labels, non_face_labels))
num_train = len(y_train)

# Step 1: Compute mean image
mean_image = np.mean(X_train, axis=0)

# Step 2: Center the data
X_centered = X_train - mean_image

# Step 3: Compute covariance matrix manually
cov_matrix = np.dot(X_centered.T, X_centered) / (num_train - 1)


# Step 4: Power iteration to find top k eigenvectors
def power_iteration(A, num_iterations=10):
    b = np.random.rand(A.shape[0])
    for _ in range(num_iterations):
        b_new = np.dot(A, b)
        b = b_new / np.sqrt(np.sum(b_new ** 2))
    eigenvalue = np.dot(b, np.dot(A, b)) / np.dot(b, b)
    return eigenvalue, b


# Compute top k eigenvectors (k=10)
k = 10
eigenvectors = []
eigenvalues = []
A = cov_matrix.copy()
for _ in range(k):
    eigenvalue, eigenvector = power_iteration(A)
    eigenvectors.append(eigenvector)
    eigenvalues.append(eigenvalue)
    A = A - eigenvalue * np.outer(eigenvector, eigenvector)

eigenvectors = np.array(eigenvectors).T

# Step 5: Project training data onto PCA space
X_pca = np.dot(X_centered, eigenvectors)

# Step 6: Compute mean face and non-face vectors in PCA space
face_indices = y_train == 1
non_face_indices = y_train == 0
mean_face_pca = np.mean(X_pca[face_indices], axis=0)
mean_non_face_pca = np.mean(X_pca[non_face_indices], axis=0)

# Save the PCA model
np.savez("pca_model.npz",
         mean_image=mean_image,
         eigenvectors=eigenvectors,
         mean_face_pca=mean_face_pca,
         mean_non_face_pca=mean_non_face_pca)

print("PCA model saved to pca_model.npz")