import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageQt
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QMessageBox


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
            self.detect_face()

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

                # Display the image in Widget_Org_Image
                self.display_image(self.current_image, self.Widget_Org_Image)

                # Prepare image for processing
                resized_img = self.current_image.resize((self.img_size, self.img_size))
                self.current_image_array = np.array(resized_img).flatten()

                # Status update
                self.statusBar().showMessage(f"Loaded image: {os.path.basename(file_name)}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def detect_face(self):
        """Detect if the image contains a face"""
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

            # Print the result to console for now (no UI changes as requested)
            result_text = "Face Detected!" if self.is_face else "No Face Detected"
            print(f"Detection result: {result_text}")
            print(f"Distance to face class: {dist_to_face:.2f}")
            print(f"Distance to non-face class: {dist_to_non_face:.2f}")

            # Update status bar
            self.statusBar().showMessage(f"Detection completed: {result_text}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")
            print(f"Error in detection: {e}")

    def display_image(self, pil_image, widget):
        """Display a PIL image in a widget"""
        if pil_image and widget:
            # Convert PIL image to QPixmap
            q_image = ImageQt.toqimage(pil_image)
            pixmap = QPixmap.fromImage(q_image)

            # Create a label to hold the image
            label = QtWidgets.QLabel(widget)
            label.setScaledContents(True)

            # Set the pixmap and make the label fill the widget
            label.setPixmap(pixmap)

            # Create a layout if it doesn't exist
            if not widget.layout():
                layout = QtWidgets.QVBoxLayout(widget)
                widget.setLayout(layout)
            else:
                # Clear existing layout
                while widget.layout().count():
                    item = widget.layout().takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()

            # Add label to layout
            widget.layout().addWidget(label)

            # Make sure the image fits within the widget
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

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


# Add missing Qt namespace
from PyQt5.QtCore import Qt

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec_())