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


def myspgend(m,p):
    sound=p+"/"+m+".wav"
    sourcerun=p+"/myspsolution.praat"
    path=p+"/"
    try:
        objects= run_file(sourcerun, -20, 2, 0.3, "yes",sound,path, 80, 400, 0.01, capture_output=True)
        #print (objects[0]) # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
        z1=str( objects[1]) # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
        z2=z1.strip().split()
        z3=float(z2[8]) # will be the integer number 10
        z4=float(z2[7]) # will be the floating point number 8.3

        if z4>97 and z4<=114:
            return 1
        elif z4>114 and z4<=135:
            return 2
        elif z4>135 and z4<=163:
            return 3
        elif z4>163 and z4<=197:
            return 1
        elif z4>197 and z4<=226:
            return 2
        elif z4>226 and z4<=245:
            return 3
        else:
            print("Voice not recognized")
            return 2
    except:
        print ("Try again the sound of the audio was not clear")
        return 2
