import threading
import cv2

CAMERA_URL = 'http://192.168.227.20:8080/video'
cap = cv2.VideoCapture(CAMERA_URL)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame = None
lock = threading.Lock()

def capture_frame():
    global frame
    while True:
        ret, temp_frame = cap.read()
        if ret:
            with lock:
                frame = temp_frame

def detect_faces():
    global frame
    while True:
        if frame is not None:
            with lock:
                current_frame = frame.copy()

            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(current_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('IP Webcam - Видеопоток с распознаванием лиц', current_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Запускаем два потока
capture_thread = threading.Thread(target=capture_frame)
detection_thread = threading.Thread(target=detect_faces)

capture_thread.start()
detection_thread.start()

capture_thread.join()
detection_thread.join()

cap.release()
cv2.destroyAllWindows()