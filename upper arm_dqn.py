import AIO
import motor
import sys
import os
import csv
from collections import deque
import pandas as pd
import numpy as np
import serial
import scipy
import collections
import datetime
import time


##############################################################################
name='デモ'
gishuname='上島L'


CONTEC=b'AIO003'
COM_number='COM6'
XIAO=True           #XIAOをつなげるか

num_of_joints  = 3
num_of_channels= 6  #計測チャンネル数

handmotions  = ["安静", "握り", "開き"]
wristmotions = ["安静", "回内", "回外"]
elbowmotions = ["安静", "屈曲", "伸展"]
motions = []
count=0
for handmotion in handmotions:
    for wristmotion in wristmotions:
        for elbowmotion in elbowmotions:
            motions.append("["+str(count)+"]"+handmotion+wristmotion+elbowmotion)
            count+=1
#######################################################################################################
#######################################################################################################