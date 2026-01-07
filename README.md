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

Crime Prediction System (HACKATHON Project Challenge #20)
The Crime Prediction System is a project developed as part of HACKATHON Project Challenge #20. It is designed to analyze crime data and create a model that predicts the likelihood of a crime based on given parameters. The project includes backend logic for data processing and predictions, as well as a frontend interface for user interaction.

Project Files
app.py: Main file for running the web application.
Project20.ipynb: Jupyter Notebook for data analysis and model training.
crime_model.pkl: Saved model for crime prediction.
scaler.pkl: Saved StandardScaler for data preprocessing.
crimedata.csv: Crime data used for training the model.
index.html: HTML file for the web application interface.
README.md: Project documentation.
Project Structure
project-folder/
├── app.py                # Web application
├── Project20.ipynb       # Jupyter Notebook for analysis and training
├── crime_model.pkl       # Saved prediction model
├── scaler.pkl            # Saved StandardScaler
├── crimedata.csv         # Dataset for training
├── index.html            # Web application interface
├── README.md             # Project documentation
Technologies Used
Python: The main programming language used for the project.
Pandas, NumPy: Libraries for data processing and analysis.
Scikit-learn: Tools for training and evaluating machine learning models.
Matplotlib, Seaborn: Libraries for data visualization.
Flask: Framework for building the API and frontend interaction.
HTML, CSS, JS: Technologies for developing the user interface.
How to Run
Clone the repository:

git clone git@github.com:Bublik-05/crime-prediction.git
Install dependencies:

pip install -r requirements.txt
Run the Jupyter Notebook:
Execute all the cells in Project20.ipynb sequentially.

Run the web application:

python app.py
Open the web interface:
Launch index.html in your browser.


