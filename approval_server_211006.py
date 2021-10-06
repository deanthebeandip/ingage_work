############## WORD CLOUD IMPORTS ###############
#!/usr/bin/env python3
import asyncio
import json
import string
import websockets
from aiofile import async_open
import random
import pymssql
import os
import logging

#base64 to JPG
from PIL import Image
import io
import base64

COUNTER = 0
import datetime
from datetime import date
import time
# import azure.cognitiveservices.speech as speechsdk
from database.tables import table
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter, defaultdict
#=========================================================#

############ BLOB IMPORTS #####D#############
import asyncio
#=========================================================#

############ H TO MP4IMPORTS #####D#############
from subprocess import call
#=========================================================#

############ APPROVAL IMPORTS ##################
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

from moviepy.editor import *
from subprocess import call
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from datetime import timezone
from time import sleep



import os
import time
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.storage.blob import ContentSettings, ContainerClient
import sys
#=========================================================#
### Vatsal's Stuff ###
os.chdir('/home/azureuser/ingage_attention')

### GLOBAL VARIABLE###
participant_init = {}
############ SQL INFORMATION ##################
SERVER = 'ingagedevdb.database.windows.net'
USER = 'ingagedevadmin@ingagedevdb.database.windows.net'
PASSWORD = 'HotForBotsDevDB1!'
DATABASE = 'ingagedevdb'
############ SQL INFORMATION ##################


############### BLOB CLASS ####################
connection_string = "DefaultEndpointsProtocol=https;AccountName=ingagedevvideostorage;AccountKey=p/M3gDo57yDtVA0z8RGJqqYd78E9F3dTUlIW3CFG0PztYpp1ngUFF1uRf0kcz/GDSFL9p96+Az7mTSSRPC8a7g==;EndpointSuffix=core.windows.net"


h_path = "h264_folder"
h2_path = "h264_folder_2"
v_path = "video_folder"
s_path = "split_folder"



# Replace with the local folder where you want downloaded files to be stored

############### BLOB CLASS ####################
class BlobSamplesAsync(object):

    async def stream_block_blob(self, container_name, video_name, p_id):

        # Instantiate a new BlobServiceClient using a connection string - set chunk size to 1MB
        from azure.storage.blob import BlobBlock
        from azure.storage.blob.aio import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        async with blob_service_client:
            # Instantiate a new ContainerClient
            container_client = blob_service_client.get_container_client(container_name)
            # Generate 4MB of data
            # data = b'a'*4*1024*1024


            download_file_path = os.path.join(h_path, video_name)

            try:
                    i = 0
                    download_complete = 0

                    while i < 5 and download_complete == 0:
                        try:
                            source_blob_client = container_client.get_blob_client(video_name)
                            stream = await source_blob_client.download_blob(validate_content = False)

                            with open(download_file_path, 'wb') as fp:
                                data = await stream.readall()
                                fp.write(data)
                                # print("Async Chunk received")
                                # print(time.perf_counter())
                                append_to_file_multiple("Async Chunk received: ", time.perf_counter(), " Participant ID: ", p_id)
                                fp.close()
                                download_complete = 1

                        except Exception as e:
                            append_to_file_multiple(e, "This is exception number", i+1, " time: ", time.time(), " Participant ID: ", p_id)
                            await asyncio.sleep(5)

                        i += 1

                # Upload the whole chunk to azure storage and make up one blob
                # await destination_blob_client.commit_block_list(block_list)
            finally:
                # Delete container
                # await container_client.delete_container()
                # print("Entered finally")
                append_to_file_multiple("Download Complete ", " Participant ID: ", p_id)

                participant_init[p_id] = 0
                #os.remove(download_file_path)



############ APPROVAL fUNCTIONS ##################
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

def vert_dist(x, y):
    return y[1] - x[1]

def num_sign(x):
    if x < 0:
        return 0
    else:
        return 1

def getFrame(sec, invid):
    smilecount = 0
    invid.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = invid.read()
    if hasFrames:
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smilecount += detect_smile(gray1, image)
    return hasFrames, smilecount
def detect_smile(gray, frame):
    scount=0
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.3, 25)
        for (sx, sy, sw, sh) in smiles:
            scount = 1
    return scount

def vid_to_smile(path, filename, p_id_smile):
    vidcap = cv2.VideoCapture(path + "/"+ filename) #INCLUDE THE PATH HERE, NOT IN THE
    sec = 0
    frameRate = 1
    smile_count = 0
    success, sc = getFrame(sec, vidcap)
    if sc: smile_count += 1
    while success:
        sec = sec + frameRate
        sec = round(sec, 2)
        success, sc = getFrame(sec, vidcap)
        if sc:
            smile_count += 1
            sec += 4
            print("Smile detected, moving forward 4 seconds!", sec)
    return smile_count

def vid_to_nod(path, filename, p_id_vid):
    cap = cv2.VideoCapture(path + "/" + filename)
    face_found = False
    frame_num = 0
    while not face_found:
        frame_num += 1
        ret, frame = cap.read()
        if not ret:
            # print("Return not found!--------------------")
            append_to_file_multiple("Return not found!----------- Participant ID: ", p_id_vid)
            return 0
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        except (cv2.error):
            # print("CV2 Error! No frame detected")
            append_to_file_multiple("CV2 Error! No frame detected Participant ID: ", p_id_vid)
            continue
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            face_found = True
        cv2.waitKey(1)

    face_center = x+w/2, y+h/3
    p0 = np.array([[face_center]], np.float32)

    #define movement threshodls
    hr = 0.0225
    disp_threshold = hr * h

    vid_works = 1
    nod_count = 0
    fc = 0
    up = 0

    while vid_works:
        ret,frame = cap.read()
        old_gray = frame_gray.copy()
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except (cv2.error):
            # print("No more frames!", fc)
            append_to_file_multiple("No more frames! Participant ID:",  p_id_vid)
            vid_works = 0

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        #get the xy coordinates for points p0 and p1

        displacement = vert_dist(get_coords(p0), get_coords(p1))

        #print(disp_threshold, displacement)

        if (abs(displacement) >= disp_threshold) :

            if num_sign(displacement) != up:
                nod_count += 1

            up = num_sign(displacement) #update the direction


        p0 = p1
        fc += 1

    cv2.destroyAllWindows()
    cap.release()
    return int(nod_count/2)


############ H TO MP4 fUNCTIONS ##################
def convert(file_h264, file_mp4):
    command = "ffmpeg -framerate 24 -y -i " + file_h264 + " -c copy " + file_mp4
    call([command], shell=True)

def concat_vid(vid_1, vid_2, vid_3):
    #command = ffmpeg -i "concat:input1.mp4|input2.mp4" -c copy output.mp4
    command = 'ffmpeg -i "concat:' + vid_1 + "|" + vid_2 + '" -c copy ' + vid_3
    call([command], shell=True)



def update_attention_to_DB(participantID, meetingId, attention, attentionTime) :
    print("Entering update_attention with", meetingId)
    print("Attention", attention)
    with pymssql.connect(SERVER, USER, PASSWORD, DATABASE) as CONN:
        CONN.autocommit(True)
        with CONN.cursor(as_dict=True) as CURSOR:
            CURSOR.callproc('SP_InsertParticipantAttentionData', (participantID, meetingId, attention, attentionTime))
    append_to_file_multiple("SP_InsertParticipantAttentionData Call completed for ---" + str(participantID))
    print("Exiting update_attention after SP_InsertParticipantAttentionData")

'''
SP_InsertParticipantAttentionData
@participantID varchar(50),
    @meetingId varchar(50),
    @attention    bit,
    @attentionTime    datetime
'''

def append_to_file_multiple(*args):
    mess = ""
    for arg in args:
        mess = mess + str(arg) + " "
    append_to_file_sync(mess)

def append_to_file_sync(content):
    current_date_and_time = datetime.datetime.now()
    current_date_and_time_string = str(current_date_and_time)

    content = current_date_and_time_string + "--" + content + "\n"
    current_date = date.today()
    current_date = str(current_date)
    path = "AttentionSingleFile" + str(current_date) + ".log"

    mode = 'a' if os.path.exists(path) else 'w'

    with open(path, mode) as f:
        f.write(content)

async def process_video(meetingMetaData, incoming_time):
    # starting time
    start = time.time()
    global COUNTER
    COUNTER = COUNTER + 1

    #get the container name and h264 file name
    meeting_container_name = meetingMetaData['ContainerName']
    meeting_file_name = meetingMetaData['FileName']


    file_name_mp4 = meeting_file_name.replace(".h264", ".mp4") #change name from h264 to mp4
    h_path_long = os.path.join(h2_path, meeting_file_name)
    v_path_long = os.path.join(v_path, file_name_mp4)
    convert(h_path_long, v_path_long) #convert h264 to mp4 here

    end = time.time()
    append_to_file_multiple("Total time converting video h264->mp4: ", end - start, ", Participant_ID:", meetingMetaData['ParticipantID'])

    clip = VideoFileClip(v_path_long) #grab the mp4 video using moviepy
    # print("Total time for incoming mp4:", clip.end) #time of video
    append_to_file_multiple("Total length of mp4 video:", clip.end, " Participant ID: ", meetingMetaData['ParticipantID'])

    #Turn the mp4 video into a 1 minute clip
    t1 = str(datetime.timedelta(seconds = clip.end - 60))
    t2 = str(datetime.timedelta(seconds = clip.end))

    cminname = "clip_min_" + str(meetingMetaData['ParticipantID']) + ".mp4" #clip min name
    s_path_long = os.path.join(s_path, cminname) #make the long path for split folder + mp4 name


    if clip.end > 60: #check to see video is longer than 60 seconds
        #command = "ffmpeg -y -i " + v_path_long + " -ss 0" + t1 +" -t 0" + t2 + " -async 1 " + s_path_long
        #command = "ffmpeg -y -i " + v_path_long + " -ss 0" + t1 +" -t 0" + t2 + " -c copy " + s_path_long
        #command = "ffmpeg -y -ss 0" + t1 + " -to 0" + t2+ " -i " + v_path_long + " -ss 0" + t1 + " -to 0" + t2 + " -c copy " + s_path_long
        command = "ffmpeg -sseof -60 -y -i " + v_path_long + " " + s_path_long
        call([command], shell=True)
    else:
        command = "ffmpeg -y -i " + v_path_long + " -ss 0" + t1 +" -t 0" + t2 + " -c copy " + s_path_long
        call([command], shell=True)

    end = time.time()
    append_to_file_multiple("Total time after clipping: ", end - start, ", Participant_ID:", meetingMetaData['ParticipantID'])

    smile_num = 0
    nod_num = 0

    smile_num  = vid_to_smile(s_path, cminname, meetingMetaData['ParticipantID'])
    end = time.time()
    append_to_file_multiple("Total time smile alg - start: ", end - start, ", Participant_ID:", meetingMetaData['ParticipantID'])
    #print("Smiles Detected:", smile_num)
    nod_num  = vid_to_nod(s_path, cminname, meetingMetaData['ParticipantID'])
    end = time.time()
    append_to_file_multiple("Total time nod alg - start: ", end - start, ", Participant_ID:", meetingMetaData['ParticipantID'])
    #print("Nods Detected:", nod_num)
    append_to_file_multiple("Smiles + Nods:", smile_num , nod_num, ", Participant_ID:", meetingMetaData['ParticipantID'])

    smile_score = 0
    nod_score = 0

    if smile_num == 0: #0, 1-3, 4+
        smile_score = 0
    elif smile_num >= 1 and smile_num <= 3:
        smile_score = 1
    elif smile_num > 3:
        smile_score = 2

    if nod_num == 0 or nod_num == 1: #0-1, 2-6, 7+
        nod_score = 0
    elif nod_num > 1 and nod_num <= 6:
        nod_score = 1
    elif nod_num > 6:
        nod_score = 2



    update_attention_to_DB(meetingMetaData['ParticipantID'], meetingMetaData['MeetingID'],  nod_score + smile_score , incoming_time)
    append_to_file_multiple("Total Attention Score", smile_score + nod_score, ", Participant_ID:", meetingMetaData['ParticipantID'])

    #time tracker...
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # file_name = 'callimage_' + str(COUNTER) + "_" + timestr + '.jpg'
    global last_video_file_name
    last_video_file_name = 'callvideo_' + str(COUNTER) + "_" + timestr + '.txt'

    # total time taken
    print("AttentionTime-", meetingMetaData['AttentionTime'])
    AttentionTime = meetingMetaData['AttentionTime'].replace("T", " ")

    #Remove the created files
    for f1 in os.listdir(s_path):
        os.remove(os.path.join(s_path, f1))
    for f2 in os.listdir(v_path):
        os.remove(os.path.join(v_path, f2))

    print("Files Have been Deleted")
    end = time.time()
    # print(f"Total Time Spent in processing the request is {end - start}")
    append_to_file_multiple("Total Time Spent in processing the request is: ", end - start, ", Participant_ID:", meetingMetaData['ParticipantID'])

async def echo(websocket, path):
    data = {}
    print("Echo request received-")
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)

    async for message in websocket:
        if isinstance(message, str):
            print(message)
            await download_once(message)
			# {message looks like:
                    # MeetingID = "abc",
                    # ParticipantID = "abc",
                    # AttentionTime = "2021-07-12 12:11:12.123",
                    # Seq = 1,
                    # Image = Base64 string
            # }

async def download_once(message):
    messageMetaData = {}
    messageMetaData = json.loads(message)
    append_to_file_multiple("Entered DL_Once, Message: ", message)
    #What if I initialize participant init to 0 here
    if messageMetaData["MeetingID"]:
        #Check to see if part_init has key already, if yes then set it to 0, if not then create new with 0
        if messageMetaData["ParticipantID"] not in participant_init:
            participant_init[messageMetaData["ParticipantID"]] = 0
        #print("Participant init has been set to 1")
        meeting_container_name = messageMetaData['ContainerName']
        meeting_file_name = messageMetaData['FileName']
        #print("Info about the participant inits:")
        #print("The Init is currently: ", type(participant_init[messageMetaData["ParticipantID"]] ), participant_init[messageMetaData["ParticipantID"]] )
        #if (participant_init[messageMetaData["ParticipantID"]] == 0) :
        #    participant_init[messageMetaData["ParticipantID"]] = 1
        # print("Before the stream begins")
        append_to_file_multiple("Before the stream begins... Participant ID: ", messageMetaData["ParticipantID"])

        try :
            sample = BlobSamplesAsync()
            await sample.stream_block_blob(meeting_container_name, meeting_file_name, messageMetaData["ParticipantID"])
            #here, it will download the newest dean-1.h264 and that's the meeting filename
        except Exception as e:
            append_to_file_multiple(e, "Participant ID: ", messageMetaData["ParticipantID"])

        print("MESSAGE RECEIVED-----", type(message))

        #Check to see if this video is the first minute:
        if meeting_file_name.find("_1.") > 0: #First minute detected

            copy_name = meeting_file_name.replace("_1.", "_1m.")
            orig_path = os.path.join(h_path, meeting_file_name)
            dest_path = os.path.join(h2_path, copy_name)
            shutil.copyfile(orig_path, dest_path) #create a master copy in the second folder
            append_to_file_multiple("The first minute has been copied!:", copy_name, "Participant ID: ", messageMetaData["ParticipantID"])
            messageMetaData['FileName'] = copy_name

            os.remove(orig_path) #delete _1 in folder 1
            
            
        else: #Not the first one!, so simply concat new  meeting_filename
            underscore_pos = meeting_file_name.find("_", 30)
            pid_meeting_name = meeting_file_name[:underscore_pos]  #got the pid_424-4324_5.h264 -> pid_424-4324
            current_min = int(meeting_file_name[underscore_pos + 1 : meeting_file_name.find(".")])
            append_to_file_multiple("The later minutes are being processed! Current min:", current_min, "Participant ID: ", messageMetaData["ParticipantID"])
            
            old_video_name = pid_meeting_name + "_" + str(current_min - 1) + "m.h264" #the last
            out_video_name = meeting_file_name.replace(".h264", "m.h264")
            
            old_vid_long = os.path.join(h2_path, old_video_name) #
            pid_vid_long = os.path.join(h_path, meeting_file_name)
            out_vid_long = os.path.join(h2_path, out_video_name)
            concat_vid(old_vid_long, pid_vid_long, out_vid_long)
            append_to_file_multiple("The later minutes are being processed! Name of output file:", out_video_name, "Participant ID: ", messageMetaData["ParticipantID"])

            messageMetaData['FileName'] = out_video_name

            os.remove(pid_vid_long) #remove _2 folder 1
            os.remove(old_vid_long) #remove _1 folder 2


        #Now that the video has been added, process the video with the correct name
        await process_video(messageMetaData, messageMetaData["AttentionTime"])


def main():
    print("STARTING APPROVAL SERVER .........")
    try:
        asyncio.get_event_loop().run_until_complete(
          websockets.serve(echo, '0.0.0.0', 4569, max_size=1380891))
        asyncio.get_event_loop().run_forever()
    except websockets.ConnectionClosedError as ex:
        print("Connection Closed", ex)

    print("FOREVER .........")

if __name__ == "__main__":
    main()
