import cv2
from data_loader import load_georgia_tech_dataset
from haar_features import generate_haar_features
from adaboost import adaboost_train, detect_faces, non_max_suppression


def main():
    # Paths to dataset
    images_root = r"D:\College\Year Three\Second Term\Computer Vision\Face-Detection-and-Recognition\data\images"
    labels_folder = r"D:\College\Year Three\Second Term\Computer Vision\Face-Detection-and-Recognition\data\labels"

    # Step 1: Load and preprocess dataset
    print("Loading dataset...")
    try:
        train_images, train_labels = load_georgia_tech_dataset(images_root, labels_folder)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Step 2: Generate Haar features
    print("Generating Haar features...")
    features = generate_haar_features(window_size=24)

    # Step 3: Train AdaBoost classifier
    print("Training AdaBoost classifier...")
    try:
        classifiers = adaboost_train(train_images, train_labels, features, num_iterations=10)
    except Exception as e:
        print(f"Error training classifier: {e}")
        return

    # Step 4: Detect faces in a test image
    print("Detecting faces...")
    test_image = cv2.imread('../data/images/s01/01.jpg', cv2.IMREAD_GRAYSCALE)
    if test_image is None:
        print("Error: Failed to load test image data/images/s01/01.jpg")
        return

    stage_thresholds = [0.5]  # Tune this
    detections = detect_faces(test_image, [classifiers], stage_thresholds)
    detections = non_max_suppression(detections)

    # Draw detections
    for (x, y, w) in detections:
        cv2.rectangle(test_image, (x, y), (x+w, y+w), (0, 255, 0), 2)

    # Save and display
    cv2.imwrite('output/detected_faces.jpg', test_image)
    cv2.imshow('Detected Faces', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
