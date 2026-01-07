import os
import time
import math
import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, Response, request, jsonify


CAMERA_URL = 'http://192.168.227.20:8080/video'
UNKNOWN_DIR = 'static/unknown_faces'
PER_PAGE = 16
DELAY = 2  # секунды между кадрами
FRAME_THICKNESS = 2


deploy_prototxt = 'C:/Users/User/PycharmProjects/Hackaton/deploy.prototxt'  # Путь к конфигурационному файлу
model_caffemodel = 'C:/Users/User/PycharmProjects/Hackaton/res10_300x300_ssd_iter_140000_fp16.caffemodel'

os.makedirs(UNKNOWN_DIR, exist_ok=True)

app = Flask(__name__)


net = cv2.dnn.readNetFromCaffe(deploy_prototxt, model_caffemodel)
video_capture = cv2.VideoCapture(CAMERA_URL)
last_capture_time = 0.0


known_face_encodings = []
known_face_names = []


def add_face_to_database(image_path, name):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]  # Кодировка лица
    known_face_encodings.append(encoding)
    known_face_names.append(name)


add_face_to_database("Students/dastan.jpg", "dastan")  # Пример добавления лица студента

def generate_frames():
    global last_capture_time
    frame_count = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 3 == 0:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    faces.append((startX, startY, endX - startX, endY - startY))  # x,y,w,h

            now = time.time()
            if faces and now - last_capture_time >= DELAY:
                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    timestamp = int(now * 1000)
                    cv2.imwrite(os.path.join(UNKNOWN_DIR, f"{timestamp}.jpg"), face_img)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), FRAME_THICKNESS)


                    rgb_frame = frame[:, :, ::-1]
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                        name = "Unknown"

                        if True in matches:
                            first_match_index = matches.index(True)
                            name = known_face_names[first_match_index]


                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


                        if name == "Unknown":
                            print(f"Неизвестное лицо найдено: {name}")

                            timestamp = int(now * 1000)
                            cv2.imwrite(os.path.join(UNKNOWN_DIR, f"unknown_{timestamp}.jpg"), face_img)



                last_capture_time = now

            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), FRAME_THICKNESS)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/unknown_images')
def unknown_images():
    page = int(request.args.get('page', 1))
    per_page = PER_PAGE
    all_imgs = sorted(os.listdir(UNKNOWN_DIR))
    total = len(all_imgs)
    total_pages = math.ceil(total / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    imgs = all_imgs[start:end]

    return jsonify({
        'images': imgs,
        'page': page,
        'total_pages': total_pages
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
