import cv2  # Import OpenCV library
import numpy as np  # Import NumPy library
from sklearn.neighbors import KNeighborsClassifier  # Import KNN classifier from scikit-learn
import os  # Import OS library for file operations

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to load and preprocess the dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    label_dict = {}
    label_counter = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                face = image[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (100, 100))  # Resize face to a fixed size
                images.append(face_resized.flatten())
                if person_name not in label_dict:
                    label_dict[person_name] = label_counter
                    label_counter += 1
                labels.append(label_dict[person_name])
    
    return np.array(images), np.array(labels), label_dict

# Function to train the KNN classifier
def train_knn_classifier(images, labels, n_neighbors=3):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(images, labels)
    return knn_classifier

# Function to predict the label of a given face
def predict_face(knn_classifier, face_image, label_dict):
    face_resized = cv2.resize(face_image, (100, 100)).flatten().reshape(1, -1)
    label_index = knn_classifier.predict(face_resized)[0]
    for name, index in label_dict.items():
        if index == label_index:
            return name
    return "Unknown"

# Main function to execute the face recognition system
def main():
    dataset_path = "path_to_dataset"  # Set the path to the dataset
    images, labels, label_dict = load_dataset(dataset_path)
    knn_classifier = train_knn_classifier(images, labels)

    cap = cv2.VideoCapture(0)  # Capture video from the webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face = gray_frame[y:y + h, x:x + w]
            label = predict_face(knn_classifier, face, label_dict)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
