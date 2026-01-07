AI_Guard (HACKATHON Project)

AI_Guard is a hackathon project designed for real-time face recognition and access control using a live camera stream.
The system continuously captures video from a camera and compares detected faces with a registered database.

If a person exists in the database, the system marks them as Authorized.
If the person is not found, they are marked as Unauthorized, meaning they do not belong to the company, university, or organization.

This solution can be used in government buildings, commercial companies, and universities as a physical security and access control system based on face recognition.

Project Files

app.py: Main file for running the web application.
camera_stream.py: Handles real-time camera streaming.
face_recognition_models/: Pre-trained face recognition models.
dlib/: Dlib files for face recognition.
deploy.prototxt: Face detector configuration file.
opencv_face_detector.pbtxt: OpenCV face detector config.
res10_300x300_ssd_iter_140000_fp16.caffemodel: Pre-trained face detection model.
index.html: Web interface for the system.
style.css: Main CSS styles.
style11.css: Additional styles.
captured_face.jpg: Sample captured face image.
README.md: Project documentation.

Project Structure

AI.hackathon/
├── app.py
├── camera_stream.py
├── face_recognition_models/
├── dlib/
├── templates/
├── static/
│   └── unknown_faces/
├── deploy.prototxt
├── opencv_face_detector.pbtxt
├── res10_300x300_ssd_iter_140000_fp16.caffemodel
├── index.html
├── style.css
├── style11.css
├── captured_face.jpg
├── README.md

Technologies Used

Python: Main programming language used for the project.
OpenCV: Library for real-time computer vision.
Dlib: Face recognition library.
Flask: Framework for backend and web interaction.
HTML, CSS: Technologies for building the user interface.
Machine Learning: Face recognition and identification.

How to Run

Clone the repository:

git clone https://github.com/dastanmukhan/AI.hackathon.git


Install dependencies:

pip install -r requirements.txt


Run the web application:

python app.py


Open the web interface:
Open your browser and go to:

http://127.0.0.1:5000

HACKATHON Project

Face Recognition & Backend: Dastanmukhan
AI/ML / Computer Vision: Alikhan Zinetov


