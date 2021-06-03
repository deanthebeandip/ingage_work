#Goes into each person's folder
#grabs the png
#moves it into the target directory

import shutil
import os

'''
rootdir = 'C://Users//DeanTheBean//Dean_Main//github//dean_testbench'
target_dir = 'C://Users//DeanTheBean//Dean_Main//github//dean_testbench//all_photos'
'''


rootdir = r"C:\Users\DeanTheBean\Dean_Main\UCLA_migrate_201118\Dean_Transfer\Work\Ingage\Data_Hub\Face_Data\lfw"
target_dir =  r"C:\Users\DeanTheBean\Dean_Main\UCLA_migrate_201118\Dean_Transfer\Work\Ingage\Data_Hub\Face_Data\all_pictures"


for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith(".jpg"):
            print (os.path.join(subdir, file))
            shutil.move(os.path.join(subdir, file) , target_dir)
