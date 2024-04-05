from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)

camera = cv2.VideoCapture('rdjxcm.mp4')

# Global variables to store uploaded image and name
uploaded_image = None
uploaded_name = None

kernel = np.ones((5,5),np.float32)/25

def gen_frames():
    global uploaded_image, uploaded_name
    
    if uploaded_image is not None and uploaded_name is not None:
        first_image = face_recognition.load_image_file(uploaded_image)
        first_face_encoding = face_recognition.face_encodings(first_image)[0]

        known_face_encodings = [first_face_encoding]
        known_face_names = [uploaded_name]
    else:
        # Default image and name if none is uploaded
        known_face_encodings = []
        known_face_names = []

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                if name == uploaded_name:
                    face_region = frame[top:bottom, left:right]
                    blurred_face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
                    frame[top:bottom, left:right] = blurred_face_region

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/add_person', methods=['POST'])
def add_person():
    global uploaded_image, uploaded_name
    
    if request.method == 'POST':
        img = request.files['image']
        img_name = request.form['name']
        img.save(f'uploads/{img.filename}')
        
        # Update global variables
        uploaded_image = f'uploads/{img.filename}'
        uploaded_name = img_name
        
        return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
