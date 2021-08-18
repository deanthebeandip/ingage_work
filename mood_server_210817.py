#!/usr/bin/env python3
import asyncio
import json
#import librosa

import websockets
from aiofile import async_open

import pymssql
import os
import subprocess

from datetime import datetime
from datetime import date
import time
import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf



#### Mood Imports #####
import parselmouth
from parselmouth.praat import call, run_file
import glob
import pandas as pd
import numpy as np
import scipy
from scipy.stats import binom
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
import os

import myspsolution as mysp
import csv
import sys
#### Mood Imports #####




messageMetaData = {}
created_wave_files = {}

COUNTER = 0

my_classes = ['positive', 'neutral', 'negative']
emotion_audio_subdirectory = "mood_wave"
emotion_full_path=r"/home/azureuser/mood_detector/mood_wave/"

SERVER = 'ingagedevdb.database.windows.net'
USER = 'ingagedevadmin@ingagedevdb.database.windows.net'
PASSWORD = 'HotForBotsDevDB1!'
DATABASE = 'ingagedevdb'



def myspgend(m):
    print("Entered the alg !!!")
    #sound=p+"/"+m
    print(m)

    sourcerun=emotion_full_path+"/myspsolution.praat"
    #path=p+"/"
    #sound = sound.replace("/AudioEmotion/AudioEmotion", "/AudioEmotion")
    print("All the stuff that's coming into mspgend")

    #print("The sound is : " , sound)
    #print("The path is p: " , path)
    try:
        
        print ("AUDIO WAS CLEAR")
        objects= run_file(sourcerun, -20, 2, 0.3, "yes", m, emotion_full_path, 80, 400, 0.01, capture_output=True)
        print ("Passed Objects")
        
        #print (objects[0]) # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
        z1=str( objects[1]) # This1 will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
        z2=z1.strip().split()
        z3=float(z2[8]) # will be the integer number 10
        z4=float(z2[7]) # will be the floating point number 8.3
        print ("Passed Zs")
        if z4>97 and z4<=114:
            return "Neutral"
        elif z4>114 and z4<=135:
            return "Negative"
        elif z4>135 and z4<=163:
            return "Positive"
        elif z4>163 and z4<=197:
            return "Neutral"
        elif z4>197 and z4<=226:
            return "Negative"
        elif z4>226 and z4<=245:
            return "Positive"
        else:
            print("Voice not recognized")
            return "Neutral"
    except :
        print ("Try again the sound of the audio was not clear")
        return "Neutral"


def append_to_file_sync(content):
    current_date_and_time = datetime.now()
    current_date_and_time_string = str(current_date_and_time)

    content = current_date_and_time_string + "--" + content + "\n"
    current_date = date.today()
    current_date = str(current_date)
    path = "EnergyServer" + str(current_date) + ".log"

    mode = 'a' if os.path.exists(path) else 'w'

    with open(path, mode) as f:
        f.write(content)


async def append_to_file(content):
    current_date_and_time = datetime.now()
    current_date_and_time_string = str(current_date_and_time)

    content = current_date_and_time_string + "--" + content + "\n"
    current_date = date.today()
    current_date = str(current_date)
    path = "EnergyServer" + str(current_date) + ".log"

    mode = 'a' if os.path.exists(path) else 'w'

    async with async_open(path, mode) as f:
        await f.write(content)


def update_energy_level(meetingId, participantId, energyLevel):
    print("Entering update_energy_level with", meetingId)
    print("Energy level: ", energyLevel)

    with pymssql.connect(SERVER, USER, PASSWORD, DATABASE) as CONN:
        CONN.autocommit(True)
        with CONN.cursor(as_dict=True) as CURSOR:
            CURSOR.callproc('SP_UpdateEnergyLevels', (meetingId, participantId, energyLevel))
    append_to_file_sync("SP_UpdateEnergyLevels Call completed for ---" + str(participantId)+"-MeetingID"+ meetingId+"energyLevel"+energyLevel)
    print("Exiting update_energy_level after SP_UpdateEnergyLevels")


async def create_wave_file(responseData, meetingMetaData, md5_hash=''):
    # starting time
    start = time.time()

    print("Emotion Server Wavefile....")
    global COUNTER
    COUNTER = COUNTER + 1

    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = 'callaudio_' + str(COUNTER) + "_" + timestr + '.wav'
    global last_wave_file_name
    last_wave_file_name = 'callaudio_' + str(COUNTER) + "_" + timestr + '.txt'

    print("----------Write wave file-", file_name)
    '''
    var additionalData = new
                {
                    MeetingID = "meet",
                    ParticipantID = "abc",
                    Seq = 1,
                    MD5Sum = "abcd"
                };
    '''
    print("MeetingID =", meetingMetaData['MeetingID'])
    print("ParticipantID =", meetingMetaData['ParticipantID'])
    print("Seq=", meetingMetaData['Seq'])
    print("MD5Sum=", meetingMetaData['MD5Sum'])



    wavefilename_metadata = "wave file-" + "" + file_name + "\n" + json.dumps(messageMetaData)


    async with async_open(os.path.join(emotion_audio_subdirectory, file_name), mode='bx') as f:
        await f.write(responseData)

    await asyncio.sleep(2)

    md5str = await get_MD5_commandline(os.path.join(emotion_audio_subdirectory, file_name))
    print(md5str.lower())
    print(meetingMetaData['MD5Sum'].lower())

    if  md5str.lower() == meetingMetaData['MD5Sum'].lower():
        print("Correct Wave File in process!")
        print("Passing filenmae to alg:", file_name)
        await check_energy(file_name, meetingMetaData['MeetingID'], meetingMetaData['ParticipantID'])
    else:
        print("WRONG Wave File in process!")

    return file_name
    # total time taken
    # end time
    end = time.time()
    print(f"Total Time Spent in Writing Wave file  is {end - start}")

async def check_energy(filename, meetingId, participantId):
    start = time.time()
    print("Entered check ENERGY, FILENAME IS-----***:", filename)
 
    print("Alg entering mood:-------------------------------------")
    mood = myspgend(filename)
    print("Alg returned mood:-------------------------------------", mood)

    update_energy_level(meetingId, participantId, mood)

    end = time.time()
    print(f"Total Time Spent in Performing Inference  is -- {end - start}")


async def get_MD5_commandline(filename):
    data = subprocess.Popen(['md5sum', filename], stdout=subprocess.PIPE)

    #proc = Popen("cat /etc/group", shell=True, stdout=subprocess.PIPE)
    out = data.stdout.read().decode("utf-8")

    print("Command output type-", type(out))
    print("Checksum Found---->", out[0:32])

    print(out)
    return out[0:32]


async def get_md5_checksum(filename):
    #md5_hash = hashlib.md5()
    async with async_open(os.path.join(emotion_audio_subdirectory, filename), mode='rb') as f:
        bytes = f.read()
        #for byte_block in iter(lambda: f.read(4096), b""):
        #  md5_hash.update(byte_block)
        print(hashlib.md5(bytes).digest())

def checkAudioSeq(audioProcDict, key, seq_num):
    if key in audioProcDict.keys():
        for seq_num in audioProcDict.get(key):
            return True
    else:
        return False

async def echo(websocket, path):
    data = {}
    print("Echo request received-")
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)

    global messageMetaData

    async for message in websocket:

        if isinstance(message, str):

            print("STRING RECEIVED-----", type(message))
            print(message)
            messageMetaData = {}
            messageMetaData = json.loads(message)

            print(messageMetaData)

        else:
            print("messageMetaData in Context---", str(messageMetaData))

            messageDataCopy = messageMetaData.copy()

            md5_content = hashlib.md5(message).digest()
            print("md5_content--", md5_content)

            #if md5_content in created_wave_files.keys():
            #    print("Request with this content already came-", md5_content)
            #else:
            fname = await create_wave_file(message, messageDataCopy)
            await websocket.send(str(messageDataCopy))





def main():
    print("STARTING Emotion File Receiver Server .........")
    try:
        asyncio.get_event_loop().run_until_complete(
            websockets.serve(echo, '0.0.0.0', 4568, max_size= 1380891))
        asyncio.get_event_loop().run_forever()
    except websockets.ConnectionClosedError as ex:
        print("Connection Closed", ex)


if __name__ == "__main__":
    # static_test()
    # static_test_event_loop_creator()
    main()
