from ..libs import *
from project import app
from flask import render_template, flash, request, redirect, url_for, Response, session

import cv2
import numpy as np
import face_recognition
import pygame
import os
import datetime, time
from threading import Thread
import datetime, time

global capture,rec_frame, grey, switch, face, rec, out, fullName, verify
capture=0
grey=0
face=0
switch=1
rec=0
fullName=''
verify=0

UPLOAD_FOLDER = os.getcwd() + '/trainers'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe( os.getcwd() +'/saved_model/deploy.prototxt.txt', os.getcwd() +'/saved_model/res10_300x300_ssd_iter_140000.caffemodel')

# initialize Camera Open
camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

# Import models here

@app.route('/face-detection', methods = ['GET'])
def faceDetection():
    data = {
        "title": "Register Face Detection",
        "body": "Register Body"
    }
    return render_template('register.html', data= data)

@app.route('/requests', methods=['POST', 'GET'])
def requestActions():
    global switch, camera, face_verify
    data = {
        "title": "Register Face Detection",
        "body": "Register Body"
    }
    print('rec : %s'%(request.form.get('rec')))   
    if request.form.get('click') == 'Capture':
        global capture, fullName
        capture=1
        fullName=request.form.get('fullName') or 'Photo'   

    elif  request.form.get('face') == 'Face Only':
        global face
        face=not face 
        if(face):
            time.sleep(4)   

    elif  request.form.get('stop') == 'Stop/Start':
        if(switch==1):
            switch=0
            # Close
            camera.release()
            cv2.destroyAllWindows()

        else:
            # Open
            camera = cv2.VideoCapture(0)
            switch=1

    elif  request.form.get('rec') == 'Start/Stop Recording':
        global rec, out
        rec= not rec
        
        #recordVideo(rec)
        print('IF rec : %s'%(rec)) 
        if(rec):
            
            now=datetime.datetime.now() 
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))

            #Start new thread for recording the video
            thread = Thread(target = record, args=[out,])
            thread.start()

        elif(rec==False):
            out.release()

    elif request.form.get('face_verify') == 'Face Recognize':
        global verify
        if(verify==1):
            verify=0
            # camera.release()
            # cv2.destroyAllWindows()
        else:
            verify=1    
            # Start new thread for verify face
            # thread = Thread()
            # thread.start()
            camera.release()
            cv2.destroyAllWindows()
            camera = cv2.VideoCapture(0)       

    elif request.method=='GET':
        return render_template('register.html', data=data)
    
    print('register.html : %s'%(request.method))   
    return render_template('register.html', data=data)

@app.route('/video_capturing')
def video_capturing():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_livestream_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Camera LiveStream
def generate_livestream_frames():
    global out, capture, rec_frame, fullName, verify
    while True:
        # generate frame by frame from camera
        success, frame = camera.read()     
        if success:

            if(face):
                frame = detect_face_area(frame)

            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if(capture):
                capture = 0
                now = datetime.datetime.now()   
                photoPath = os.path.sep.join([UPLOAD_FOLDER, "{}_{}.jpg".format(fullName, str(now).replace(":",''))])
                cv2.imwrite(photoPath, frame)

            if(verify):
                verify = 0
                IMAGE_FILES = []
                filename = []
                dir_path = UPLOAD_FOLDER
                for imagess in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, imagess)
                    img_path = face_recognition.load_image_file(img_path)  # reading image and append to list
                    IMAGE_FILES.append(img_path)
                    filename.append(imagess.split(".", 1)[0])

                def encoding_img(IMAGE_FILES):
                    encodeList = []
                    for img in IMAGE_FILES:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        encode = face_recognition.face_encodings(img)[0]
                        encodeList.append(encode)
                    return encodeList
                
                encodeListknown = encoding_img(IMAGE_FILES)

                imgc = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

                # converting image to RGB from BGR
                imgc = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #0, 255, 0 #cv2.COLOR_BGR2RGB
                
                fasescurrent = face_recognition.face_locations(imgc)
                encode_fasescurrent = face_recognition.face_encodings(imgc, fasescurrent)
                # faceloc- one by one it grab one face location from fasescurrent
                # than encodeFace grab encoding from encode_fasescurrent
                # we want them all in same loop so we are using zip
                for encodeFace, faceloc in zip(encode_fasescurrent, fasescurrent):
                    matches_face = face_recognition.compare_faces(encodeListknown, encodeFace)
                    face_distence = face_recognition.face_distance(encodeListknown, encodeFace)
                    # finding minimum distence index that will return best match
                    matchindex = np.argmin(face_distence)

                    if matches_face[matchindex]:

                        name = filename[matchindex].upper()
                        # print(name)
                        y1, x2, y2, x1 = faceloc
                        # multiply locations by 4 because we above we reduced our webcam input image by 0.25
                        # y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (255, 0, 0), 2, cv2.FILLED)
                        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        #takeAttendence(name)  # taking name for attendence function above

                    else:
                        try:
                            pygame.init()
                            pygame.mixer.init()
                            sounda= pygame.mixer.Sound( os.getcwd() + "/sounds/mixkit-facility-alarm-sound.wav")
                            sounda.play()
                        except FileNotFoundError:
                            print('Failed to communicate with XMS: %s' % str(FileNotFoundError))

                cv2.imshow("campare", frame)        
                #verifyFace(frame)

            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

            
        else:
            pass

def detect_face_area(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
        return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame
    
