import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        X = np.array(X)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Use np.linalg.svd instead of eigh to avoid complex numbers
        U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)

        # Eigenvectors are already sorted by eigenvalue in SVD
        self.components = Vh[:self.n_components]

    def transform(self, X):
        X = np.array(X)
        X_centered = X - self.mean
        return np.real(np.dot(X_centered, self.components.T))

class EigenFaceRecognizer:
    def __init__(self, n_components=50, model_dir='models'):
        self.n_components = n_components
        self.model_dir = model_dir
        self.pca = PCA(n_components=n_components)
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.mean = None
        os.makedirs(model_dir, exist_ok=True)
        self.train_images = []
        self.train_labels = []

    def load_images(self, folder):
        images, labels = [], []
        for subfolder in os.listdir(folder):
            label = int(subfolder[1:])
            subfolder_path = os.path.join(folder, subfolder)
            for filename in os.listdir(subfolder_path):
                path = os.path.join(subfolder_path, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (100, 100))
                images.append(img.flatten())
                labels.append(label)
        self.train_images = images
        self.train_labels = labels
        return np.array(images), np.array(labels)

    def save_models(self):
        joblib.dump(self.pca, os.path.join(self.model_dir, 'pca.pkl'))
        joblib.dump(self.knn, os.path.join(self.model_dir, 'knn.pkl'))
        joblib.dump(self.train_images, os.path.join(self.model_dir, 'train_images.pkl'))
        joblib.dump(self.train_labels, os.path.join(self.model_dir, 'train_labels.pkl'))

    def load_models(self):
        self.pca = joblib.load(os.path.join(self.model_dir, 'pca.pkl'))
        self.knn = joblib.load(os.path.join(self.model_dir, 'knn.pkl'))
        self.train_images = joblib.load(os.path.join(self.model_dir, 'train_images.pkl'))
        self.train_labels = joblib.load(os.path.join(self.model_dir, 'train_labels.pkl'))

    def train(self, train_folder, force_retrain=False):
        if not force_retrain and all(os.path.exists(os.path.join(self.model_dir, f)) for f in ['pca.pkl', 'knn.pkl', 'train_images.pkl', 'train_labels.pkl']):
            print("[INFO] Loading models...")
            self.load_models()
        else:
            print("[INFO] Training models...")
            images, labels = self.load_images(train_folder)
            self.pca.fit(images)
            projected = self.pca.transform(images)
            projected = np.real(projected)
            self.knn.fit(projected, labels)
            self.save_models()

    def predict(self, test_image_path):
        img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (100, 100))
        flat = img_resized.flatten().reshape(1, -1)
        img_pca = self.pca.transform(flat)
        img_pca = np.real(img_pca)
        label = self.knn.predict(img_pca)
        neighbors = self.knn.kneighbors(img_pca, return_distance=False)[0]

        # plt.figure(figsize=(10, 4))
        # plt.subplot(1, len(neighbors) + 1, 1)
        # plt.imshow(img_resized, cmap='gray')
        # plt.title("Test Image")
        # plt.axis('off')

        # for i, idx in enumerate(neighbors):
        #     neighbor_img = np.array(self.train_images[idx]).reshape(100, 100)
        #     # plt.subplot(1, len(neighbors) + 1, i + 2)
        #     # plt.imshow(neighbor_img, cmap='gray')
        #     # plt.title(f"Neighbor {i+1}")
        #     # plt.axis('off')

        # plt.tight_layout()
        # plt.show()

        return label[0], neighbors

    def get_neighbors_images(self, test_image_path):
        """
        Given a test image path, return a single stacked image vertically
        containing the 3 nearest neighbors with white space in between.
        """
        img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[ERROR] Could not read image at {test_image_path}")
            return None

        img_resized = cv2.resize(img, (100, 100))
        flat = img_resized.flatten().reshape(1, -1)
        img_pca = self.pca.transform(flat)
        img_pca = np.real(img_pca)
        neighbors_idx = self.knn.kneighbors(img_pca, return_distance=False)[0]

        neighbor_images = []
        spacer = 255 * np.ones((10, 100), dtype=np.uint8)  # 10px white spacer

        for i, idx in enumerate(neighbors_idx):
            img_flat = np.array(self.train_images[idx])
            if img_flat.shape[0] != 10000:
                print(f"[ERROR] Image at index {idx} has shape {img_flat.shape}")
                continue
            neighbor_img = img_flat.reshape(100, 100)
            neighbor_images.append(neighbor_img)
            if i < len(neighbor_images) - 1:
                neighbor_images.append(spacer)

        if neighbor_images:
            combined_image = np.vstack(neighbor_images)
            return combined_image
        else:
            return None
            

    def save_roc_and_confusion_matrix(self, y_true, y_pred, model_name, save_dir):
        from sklearn.metrics import roc_curve, auc, confusion_matrix
        from sklearn.preprocessing import label_binarize
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Get unique classes
        classes = np.unique(np.concatenate((y_true, y_pred)))
        n_classes = len(classes)
        
        # For confusion matrix (this remains mostly the same)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()
        
        # For ROC curve - handle multi-class
        if n_classes > 2:
            # One-hot encode the labels for multi-class ROC
            y_bin = label_binarize(y_true, classes=classes)
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            # Convert predictions to probability-like scores using one-hot encoding
            y_score = label_binarize(y_pred, classes=classes)
            
            plt.figure(figsize=(10, 8))
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=2, 
                        label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            plt.plot(fpr["micro"], tpr["micro"], 
                    label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
                    color='deeppink', linestyle=':', linewidth=4)
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Multi-class ROC for {model_name}')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_dir, f'{model_name}_roc_multiclass.png'))
            plt.close()
            
        else:
            # For binary classification (original method)
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_dir, f'{model_name}_roc.png'))
            plt.close()
        
        # Also save accuracy, precision, recall, and F1 score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        
        accuracy = accuracy_score(y_true, y_pred)
        # Use 'macro' average for multi-class
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Save metrics to a text file
        with open(os.path.join(save_dir, f'{model_name}_metrics.txt'), 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Detailed Classification Report:\n")
            f.write(classification_report(y_true, y_pred))

            
# if __name__ == "__main__":

    

#     recognizer = EigenFaceRecognizer()
#     recognizer.load_models()  # load trained models

#     test_image_path = r"data\data\images_cropped_test\s02\07.jpg"
#     stacked_img = recognizer.get_neighbors_images( test_image_path)

#     if stacked_img is not None:
#         plt.imshow(stacked_img, cmap='gray')
#         plt.axis('off')
#         plt.title("3 Nearest Neighbors")
#         plt.show()
#     else:
#         print("No neighbors found or error occurred.")



if __name__ == "__main__":
    recognizer = EigenFaceRecognizer()
    recognizer.load_models()

    # تحميل صور الاختبار
    test_folder = r"data\data\images_cropped_test"
    y_true = []
    y_pred = []

    for subfolder in os.listdir(test_folder):
        true_label = int(subfolder[1:])
        subfolder_path = os.path.join(test_folder, subfolder)
        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)
            pred_label, _ = recognizer.predict(img_path)
            y_true.append(true_label)
            y_pred.append(pred_label)

    # حفظ النتائج
    recognizer.save_roc_and_confusion_matrix(y_true, y_pred, model_name="FaceRecognitionModel", save_dir="results")
