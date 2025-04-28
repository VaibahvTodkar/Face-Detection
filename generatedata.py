import cv2  # Import OpenCV library
import os  # Import OS library for file operations

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to capture and save face images
def capture_face_images(person_name, num_images=100, dataset_path='dataset'):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.exists(person_path):
        os.makedirs(person_path)
    
    cap = cv2.VideoCapture(0)  # Capture video from the webcam
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face = gray_frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (100, 100))
            face_filename = os.path.join(person_path, f'{person_name}_{count}.jpg')
            cv2.imwrite(face_filename, face_resized)
            count += 1
            if count >= num_images:
                break
        
        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to capture dataset images
def main():
    person_name = input("Enter the person's name: ")
    num_images = int(input("Enter the number of images to capture: "))
    capture_face_images(person_name, num_images)

if __name__ == "__main__":
    main()
