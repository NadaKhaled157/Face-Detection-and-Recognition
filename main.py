import sys
import os
import numpy as np
from PIL import Image, ImageDraw
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication, QButtonGroup
from PCA import * # Assuming PCA is defined in a separate file named PCA.py
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

    #    /////////////////////////////////////////////////////////
            # Create a button group and add the radio buttons
        button_group = QButtonGroup(self)
        button_group.addButton(self.RadioButton_Recog)
        button_group.addButton(self.RadioButton_detect)
        button_group.addButton(self.RadioButton_C_matrix)
        button_group.addButton(self.RadioButton_Roc)
        
        # Connect the buttons to a function to handle their state change
        button_group.buttonClicked.connect(self.on_radio_button_clicked)




    def setup_connections(self):
        """Set up UI element connections"""
        # Connect the double-click event on Widget_Org_Image to load image function
        try:
            self.Widget_Org_Image.mouseDoubleClickEvent = self.on_image_widget_double_click
        except AttributeError as e:
            print(f"Error connecting Widget_Org_Image: {e}")

        # Connect the radio button for detection
        try:
            self.RadioButton_detect.toggled.connect(self.on_radio_button_clicked)
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
                self.display_image_on_widget(cv_image,self.Widget_Org_Image)

                # Prepare image for processing
                resized_img = self.current_image.resize((self.img_size, self.img_size))
                self.current_image_array = np.array(resized_img).flatten()

                # Status update
                self.statusBar().showMessage(f"Loaded image: {os.path.basename(file_name)}")

#                # Call the radio button click event to trigger detection       

                self.on_radio_button_clicked()

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

            # If face detected, display the original image with face highlighted
            if self.is_face:
                self.display_face_result()
            else:
                # Clear the widget if no face is detected
                if hasattr(self.Widget_Org_Image, 'image_label'):
                    self.Widget_Org_Image.image_label.setPixmap(QPixmap())

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")
            print(f"Error in detection: {e}")

    def display_face_result(self):
        """Display the original image with the detected face highlighted"""
        try:
            # Load original image with OpenCV
            original_image = cv2.imread(self.current_image_path)
            if original_image is None:
                raise ValueError("Could not load original image with OpenCV")

            # Get original dimensions
            orig_height, orig_width = original_image.shape[:2]

            # Calculate scaling factors
            x_scale = orig_width / self.img_size
            y_scale = orig_height / self.img_size

            # Calculate face coordinates (using center region as in your PCA approach)
            center_x = orig_width // 2
            center_y = orig_height // 2
            face_size = min(orig_width, orig_height) // 2  # Adjust this based on your needs

            # Define face bounding box
            x_min = max(0, center_x - face_size // 2)
            y_min = max(0, center_y - face_size // 2)
            x_max = min(orig_width, center_x + face_size // 2)
            y_max = min(orig_height, center_y + face_size // 2)

            # Draw rectangle around the face
            cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Display the result
            self.display_image_on_widget(original_image,self.Widget_Output)

        except Exception as e:
            print(f"Error displaying face result: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to display face result: {str(e)}")

    def display_image_on_widget(self, image, widget):
        """Display a numpy array image on the given widget"""
        if image is None or image.size == 0:
            print("Empty or invalid image")
            return

        try:
            if len(image.shape) == 2:  # Grayscale
                height, width = image.shape
                bytes_per_line = width
                q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR
                height, width = image.shape[:2]
                bytes_per_line = 3 * width
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                q_img = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                raise ValueError("Unsupported image format!")

            pixmap = QPixmap.fromImage(q_img)
            widget_size = widget.size()
            scaled_pixmap = pixmap.scaled(widget_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

            # Create image_label once
            if not hasattr(widget, 'image_label'):
                widget.image_label = QtWidgets.QLabel(widget)
                widget.image_label.setAlignment(QtCore.Qt.AlignCenter)
                layout = QtWidgets.QVBoxLayout(widget)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(widget.image_label)
            else:
                # Remove previous pixmap if needed (not required strictly, but ensures clean updates)
                widget.image_label.clear()

            widget.image_label.setPixmap(scaled_pixmap)

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

# /////////////////////////////////////////////////////////////////////////////////////////////

    def on_radio_button_clicked(self):
       
        if self.RadioButton_Recog.isChecked():
            #  self.current_image_path
            recognizer = EigenFaceRecognizer()
            recognizer.load_models()  # load trained models

            
            stacked_img = recognizer.get_neighbors_images( self.current_image_path)
            self.display_image_on_widget(stacked_img, self.Widget_Output)
            print("Recognition mode selected")
        
        elif self.RadioButton_detect.isChecked():
            print("Recognition mode selected")
            self.detect_and_locate_face()
        
        elif self.RadioButton_C_matrix.isChecked():
           pass
        
        
        elif self.RadioButton_Roc.isChecked():
           pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec_())