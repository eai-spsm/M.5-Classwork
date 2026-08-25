# Lesson step 1/3: capture N face crops from the webcam for one user ID -> data/user.<id>.<n>.jpg. Run step2 next.
import cv2 as cv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
DATA_DIR = os.path.join(BASE_DIR, 'data')
SAMPLE_COUNT = 100

os.makedirs(DATA_DIR, exist_ok=True)

def capture(user_id):
    face_cascade = cv.CascadeClassifier(XML_PATH)
    if face_cascade.empty():
        print(f"Error: Could not load Haar Cascade from {XML_PATH}")
        return

    cap = cv.VideoCapture(0)
    print(f"\nLook at the camera. Capturing {SAMPLE_COUNT} faces for user {user_id}...")

    count = 0
    while cap.isOpened() and count < SAMPLE_COUNT:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            file_name = f"user.{user_id}.{count}.jpg"
            cv.imwrite(os.path.join(DATA_DIR, file_name), gray[y:y+h, x:x+w])
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, f"Captured: {count}/{SAMPLE_COUNT}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if count >= SAMPLE_COUNT:
                break

        cv.imshow("Capturing Face Data", frame)
        if cv.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    print(f"Finished capturing {count} images to {DATA_DIR}")

if __name__ == "__main__":
    try:
        user_id = int(input("Enter numeric User ID (e.g., 1, 2, 3): "))
    except ValueError:
        print("Invalid ID. Please enter a number.")
    else:
        capture(user_id)
