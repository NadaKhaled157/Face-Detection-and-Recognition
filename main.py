import sys
import os
import numpy as np
from PIL import Image, ImageDraw
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2
import matplotlib.pyplot as plt

class FaceDetectionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(FaceDetectionApp, self).__init__()
        # Load the UI file
        uic.loadUi('MainWindow.ui', self)

        # Set up model parameters
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

        # Set window title
        self.setWindowTitle("Face Detection System")

    def setup_connections(self):
        """Set up UI element connections"""
        # Connect the double-click event on Widget_Org_Image to load image function
        try:
            self.Widget_Org_Image.mouseDoubleClickEvent = self.on_image_widget_double_click
        except AttributeError as e:
            print(f"Error connecting Widget_Org_Image: {e}")

        # Connect the radio button for detection
        try:
            self.RadioButton_detect.toggled.connect(self.on_detect_radio_toggled)
        except AttributeError as e:
            print(f"Error connecting RadioButton_detect: {e}")
            print("Please check that your UI file has elements with these names: Widget_Org_Image, RadioButton_detect")

    def on_image_widget_double_click(self, event):
        """Handle double-click on the image widget to load an image"""
        self.load_image()

    def on_detect_radio_toggled(self, checked):
        """Handle the radio button being toggled for face detection"""
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
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            print(f"Error loading model: {e}")

    def load_image(self):
        """Load an image from the file system"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options
        )

        if file_name:
            try:
                # Load the image
                self.current_image_path = file_name
                self.current_image = Image.open(file_name).convert('L')

                # Load image with OpenCV for display
                cv_image = cv2.imread(file_name)
                if cv_image is None:
                    raise ValueError("Could not load image with OpenCV")

                # Display the image in Widget_Org_Image
                self.display_image_on_widget(cv_image)

                # Prepare image for processing
                resized_img = self.current_image.resize((self.img_size, self.img_size))
                self.current_image_array = np.array(resized_img).flatten()

                # Status update
                self.statusBar().showMessage(f"Loaded image: {os.path.basename(file_name)}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def detect_and_locate_face(self):
        """Detect face using PCA and display cropped face if detected"""
        if self.current_image_array is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        try:
            # Apply PCA and classify
            centered_image = self.current_image_array - self.mean_image
            pca_features = np.dot(centered_image, self.eigenvectors)

            # Calculate distances to face/non-face classes
            dist_to_face = np.sqrt(np.sum((pca_features - self.mean_face_pca) ** 2))
            dist_to_non_face = np.sqrt(np.sum((pca_features - self.mean_non_face_pca) ** 2))

            self.is_face = dist_to_face < dist_to_non_face

            # Print the result to console
            result_text = "Face Detected!" if self.is_face else "No Face Detected"
            print(f"Detection result: {result_text}")
            print(f"Distance to face class: {dist_to_face:.2f}")
            print(f"Distance to non-face class: {dist_to_non_face:.2f}")

            # Update status bar
            self.statusBar().showMessage(f"Detection completed: {result_text}")

            # If face detected, approximate its location and display cropped face
            if self.is_face:
                self.approximate_face_location()
            else:
                # Clear the widget if no face is detected
                if hasattr(self.Widget_Org_Image, 'image_label'):
                    self.Widget_Org_Image.image_label.setPixmap(QPixmap())

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")
            print(f"Error in detection: {e}")

    def approximate_face_location(self):
        """Approximate face location and display cropped region on Widget_Org_Image"""
        try:
            # Load original image with OpenCV for cropping
            original_image = cv2.imread(self.current_image_path)
            if original_image is None:
                raise ValueError("Could not load original image with OpenCV")

            height, width = original_image.shape[:2]

            # Assume face is centered and occupies ~50% of the image area
            face_size_ratio = 0.5
            face_width = int(width * face_size_ratio)
            face_height = int(height * face_size_ratio)
            x = (width - face_width) // 2
            y = (height - face_height) // 2

            self.face_coords = (x, y, face_width, face_height)
            print(f"Approximated face coordinates: x={x}, y={y}, w={face_width}, h={face_height}")

            # Crop the face region
            cropped_face = original_image[y:y+face_height, x:x+face_width]

            # Display cropped face on Widget_Org_Image
            self.display_image_on_widget(cropped_face)

        except Exception as e:
            print(f"Error approximating or displaying face: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to process face: {str(e)}")

    def display_image_on_widget(self, image):
        """Display a numpy array image on self.Widget_Org_Image"""
        if image is None or image.size == 0:
            print("Empty or invalid image")
            return

        try:
            # Determine if image is grayscale or color
            if len(image.shape) == 2:  # Grayscale
                height, width = image.shape
                bytes_per_line = width
                q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR
                height, width = image.shape[:2]
                bytes_per_line = 3 * width
                # Convert BGR to RGB for QImage
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                q_img = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                raise ValueError("Unsupported image format!")

            # Convert to QPixmap and scale
            pixmap = QPixmap.fromImage(q_img)
            widget_size = self.Widget_Org_Image.size()
            scaled_pixmap = pixmap.scaled(
                widget_size,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )

            # Ensure widget has a label to display the image
            if not hasattr(self.Widget_Org_Image, 'image_label'):
                self.Widget_Org_Image.image_label = QtWidgets.QLabel(self.Widget_Org_Image)
                self.Widget_Org_Image.image_label.setAlignment(QtCore.Qt.AlignCenter)
                layout = QtWidgets.QVBoxLayout(self.Widget_Org_Image)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self.Widget_Org_Image.image_label)

            # Set the scaled pixmap to the label
            self.Widget_Org_Image.image_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"Error displaying image: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to display image: {str(e)}")

    def clear_display(self):
        """Clear all displayed images and results"""
        try:
            if self.Widget_Org_Image.layout():
                while self.Widget_Org_Image.layout().count():
                    item = self.Widget_Org_Image.layout().takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
        except AttributeError:
            pass

        self.current_image_path = None
        self.current_image = None
        self.current_image_array = None
        self.is_face = False

        self.statusBar().showMessage("Display cleared")



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec_())