import cv2
import numpy as np
import face_recognition
import os
import pygame
from pathlib import Path
from datetime import datetime
from flask import Flask, flash, request, redirect, url_for, render_template, Response
#import requests
from werkzeug.utils import secure_filename
#from sinch import Client
#import clx.xms

UPLOAD_FOLDER = os.getcwd() + '/trainers'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

'''
#client is a object that carries your unique token.
client = clx.xms.Client(service_plan_id='ad16c74c87ba4bc59e1c3ed98b40b881', token='23e9b885134642d6acc997bcfc816113')
  
create = clx.xms.api.MtBatchTextSmsCreate()
create.sender = '447520651119'
create.recipients = {'85516629191'}
create.body = 'This is a test message from your Sinch account'
'''

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)

app.secret_key = "123456789"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/success', methods=['GET', 'POST'])
def success():
    if 'file' not in request.files:
        flash('No file part')
        return render_template('upload.html')
    
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return render_template('upload.html')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html')
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return render_template('upload.html')


@app.route('/index')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
    IMAGE_FILES = []
    filename = []
    dir_path = os.getcwd() + '/trainers'
    #print('dir_path=========%s' %(dir_path))
    for imagess in os.listdir(dir_path):
        img_path = os.path.join(dir_path, imagess)
        #print('imagess=========%s' %(imagess))
        img_path = face_recognition.load_image_file(img_path)  # reading image and append to list

        #print('img_path=========%s' %(img_path))
        IMAGE_FILES.append(img_path)
        filename.append(imagess.split(".", 1)[0])

    def encoding_img(IMAGE_FILES):
        encodeList = []
        for img in IMAGE_FILES:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def takeAttendence(name):
        with open('attendence.csv', 'r+') as f:
            mypeople_list = f.readlines()
            nameList = []
            for line in mypeople_list:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                datestring = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{datestring}')
            else:
                print(len('Not registered'))

    encodeListknown = encoding_img(IMAGE_FILES)
    #print(len('success'))

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgc = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        # converting image to RGB from BGR
        imgc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #0, 255, 0 #cv2.COLOR_BGR2RGB

        # Convert logical matrix to uint8
        #gray = gray.astype(np.uint8)*255
        
        fasescurrent = face_recognition.face_locations(imgc)
        encode_fasescurrent = face_recognition.face_encodings(imgc, fasescurrent)

        # faceloc- one by one it grab one face location from fasescurrent
        # than encodeFace grab encoding from encode_fasescurrent
        # we want them all in same loop so we are using zip
        for encodeFace, faceloc in zip(encode_fasescurrent, fasescurrent):
            matches_face = face_recognition.compare_faces(encodeListknown, encodeFace)
            #print('====matches_face====: %s' %(matches_face))
            face_distence = face_recognition.face_distance(encodeListknown, encodeFace)
            #print('====face_distence====: %s' %(face_distence))
            # finding minimum distence index that will return best match
            matchindex = np.argmin(face_distence)
            #print('====matchIndex====: %s' %(matchindex))

            if matches_face[matchindex]:
                name = filename[matchindex].upper()
                # print(name)
                y1, x2, y2, x1 = faceloc
                # multiply locations by 4 because we above we reduced our webcam input image by 0.25
                # y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), 2, cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                takeAttendence(name)  # taking name for attendence function above
            else:
                print('====Not recognized====: %s' %(matches_face))
                try:
                    pygame.init()
                    pygame.mixer.init()
                    sounda= pygame.mixer.Sound( "./sounds/mixkit-facility-alarm-sound.wav")
                    sounda.play()
                   #batch = client.create_batch(create)
                #except (requests.exceptions.RequestException, clx.xms.exceptions.ApiException) as ex:
                except FileNotFoundError:
                    print('Failed to communicate with XMS: %s' % str(FileNotFoundError))

        cv2.imshow("campare", img)
        # cv2.waitKey(0)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break


@app.route('/video_capturing')
def video_capturing():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)