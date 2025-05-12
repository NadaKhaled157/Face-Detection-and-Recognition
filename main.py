import sys
import os
import numpy as np
from PIL import Image
from PyQt5 import QtWidgets, uic
import matplotlib.pyplot as plt

class FaceDetectionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(FaceDetectionApp, self).__init__()
        uic.loadUi('MainWindow.ui', self)

        # Model parameters
        self.img_size = 64
        self.pixel_count = self.img_size * self.img_size

        # Load PCA model
        self.load_model()

        # Connect UI elements
        self.setup_connections()

        # Initialize variables
        self.current_image_path = None
        self.current_image = None
        self.current_image_array = None
        self.is_face = False
        self.face_coords = None

        self.setWindowTitle("Face Detection System")

    def setup_connections(self):
        """Set up UI element connections"""
        try:
            self.Widget_Org_Image.mouseDoubleClickEvent = self.on_image_widget_double_click
        except AttributeError as e:
            print(f"Error connecting Widget_Org_Image: {e}")

        try:
            self.RadioButton_detect.toggled.connect(self.on_detect_radio_toggled)
        except AttributeError as e:
            print(f"Error connecting RadioButton_detect: {e}")

    def on_image_widget_double_click(self, event):
        """Handle double-click to load an image"""
        self.load_image()

    def on_detect_radio_toggled(self, checked):
        """Handle radio button toggle for face detection"""
        if checked and self.current_image_array is not None:
            self.detect_and_locate_face()

    def load_model(self):
        """Load the PCA model"""
        try:
            model_data = np.load("pca_model.npz")
            self.mean_image = model_data["mean_image"]
            self.eigenvectors = model_data["eigenvectors"]
            self.mean_face_pca = model_data["mean_face_pca"]
            self.mean_non_face_pca = model_data["mean_non_face_pca"]
            print("Model loaded successfully")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            print(f"Error loading model: {e}")

    def load_image(self):
        """Load an image from the file system"""
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options
        )

        if file_name:
            try:
                self.current_image_path = file_name
                self.current_image = Image.open(file_name).convert('L')
                resized_img = self.current_image.resize((self.img_size, self.img_size))
                self.current_image_array = np.array(resized_img).flatten()
                self.statusBar().showMessage(f"Loaded image: {os.path.basename(file_name)}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def detect_and_locate_face(self):
        """Detect face using PCA and approximate its location"""
        if self.current_image_array is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        try:
            # PCA-based classification
            centered_image = self.current_image_array - self.mean_image
            pca_features = np.dot(centered_image, self.eigenvectors)
            dist_to_face = np.sqrt(np.sum((pca_features - self.mean_face_pca) ** 2))
            dist_to_non_face = np.sqrt(np.sum((pca_features - self.mean_non_face_pca) ** 2))
            self.is_face = dist_to_face < dist_to_non_face

            result_text = "Face Detected!" if self.is_face else "No Face Detected"
            print(f"Detection result: {result_text}")
            print(f"Distance to face class: {dist_to_face:.2f}")
            print(f"Distance to non-face class: {dist_to_non_face:.2f}")
            self.statusBar().showMessage(f"Detection completed: {result_text}")

            # If face detected, approximate its location and crop
            if self.is_face:
                self.approximate_face_location()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")
            print(f"Error in detection: {e}")

    def approximate_face_location(self):
        """Approximate face location and display cropped region"""
        try:
            # Load original image (not resized) for cropping
            original_image = Image.open(self.current_image_path).convert('L')
            orig_width, orig_height = original_image.size

            # Assume face is centered and occupies ~50% of the image area
            face_size_ratio = 0.5
            face_width = int(orig_width * face_size_ratio)
            face_height = int(orig_height * face_size_ratio)
            x = (orig_width - face_width) // 2
            y = (orig_height - face_height) // 2

            self.face_coords = (x, y, face_width, face_height)
            print(f"Approximated face coordinates: x={x}, y={y}, w={face_width}, h={face_height}")

            # Crop the face region
            cropped_face = original_image.crop((x, y, x + face_width, y + face_height))

            # Display using Matplotlib
            plt.figure(figsize=(4, 4))
            plt.imshow(cropped_face, cmap='gray')
            plt.title("Detected Face")
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"Error approximating or displaying face: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to process face: {str(e)}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec_())