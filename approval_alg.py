########### IMPORTS ##########
import cv2
import numpy as np
import os
import matplotlib.image as img
import matplotlib.pyplot as plt
import math
import logging
import random
import string
import shutil

######## REQUIRED LOADS ########
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')

#params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#path to face cascde
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#function to get coordinates
def get_coords(p1):
    try: return int(p1[0][0][0]), int(p1[0][0][1])
    except: return int(p1[0][0]), int(p1[0][1])
#define font and text color
font = cv2.FONT_HERSHEY_SIMPLEX


####### DEFINITIONS ############
def distance(x,y):
    return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

def getFrame(sec, invid):
    smilecount = 0
    invid.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = invid.read()
    if hasFrames:
        # cv2.imwrite(os.path.join(out_path , vid_name + '_'+str(count)+".jpg"), cv2.resize(image[:, :], (250, 250)))
        # cv2.imwrite(os.path.join(out_path , vid_name + '_f' + str(1/frameRate) + '_'+str('{0:03}'.format(count))+".jpg"), image)
        # frame1 = img.imread(folder_path)
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smilecount += detect_smile(gray1, image)
    return hasFrames, smilecount

def detect_smile(gray, frame):
    scount=0
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 2.05, 25)
        for (sx, sy, sw, sh) in smiles:
            scount = 1
    return scount

def vid_to_smile(path, filename):
    vidcap = cv2.VideoCapture(path + "/"+ filename + '.mp4') #INCLUDE THE PATH HERE, NOT IN THE
    sec = 0
    count=1
    frameRate = 1
    smile_count = 0
    success, sc = getFrame(sec, vidcap)
    if sc: smile_count += 1
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success, sc = getFrame(sec, vidcap)
        if sc: smile_count += 1
    return smile_count


def vid_to_nod(path, filename):
    cap = cv2.VideoCapture(path + "/" + filename + ".mp4")
    face_found = False
    frame_num = 0
    while not face_found:
        frame_num += 1
        ret, frame = cap.read()
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except (cv2.error):
            face_found = True
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            face_found = True
        # cv2.imshow('image',frame)
        # out.write(frame)
        cv2.waitKey(1)
    face_center = x+w/2, y+h/3
    p0 = np.array([[face_center]], np.float32)

    #define movement threshodls
    hr = 0.025
    disp_threshold = hr * h

    vid_works = 1
    nod_count = 0
    fc = 0
    while vid_works:
        ret,frame = cap.read()
        old_gray = frame_gray.copy()
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except (cv2.error):
            # print("No more frames!", fc)
            vid_works = 0

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        #get the xy coordinates for points p0 and p1
        nc = nod_count
        displacement = distance(get_coords(p0), get_coords(p1))
        if displacement >= disp_threshold:
            nod_count += 1
        p0 = p1
        fc += 1

    cv2.destroyAllWindows()
    cap.release()
    return int(nod_count/2) + 1


########## OUTPUT ############
videoname = 'nod_far_12'
path = 'video_folder'
print("SMILE VID TO FILE:",vid_to_smile(path, videoname))
print("NOD VID TO FILE:", vid_to_nod(path, videoname) )
