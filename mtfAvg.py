import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import enum
import math
import matplotlib.cm as cm
FREQ= 1
IMG_WIDTH_REAL = 6024
IMG_WIDTH_CG   = 1200
SENSOR_WIDTH   = 35.9
LENS_FOCUS_DISTANCE = 35
CAM_TO_IMG_DISTANCE = 300
class ImgType(enum.IntEnum):
    REAL = 0
    CG = 1

# cycles/px -> lp/mm
def cpm2lppmm(x, type):
    return 1.0*x * (IMG_WIDTH_REAL if type==ImgType.REAL else IMG_WIDTH_CG) / (1.0* SENSOR_WIDTH * CAM_TO_IMG_DISTANCE / LENS_FOCUS_DISTANCE)

def getData(path, freq, ImgType):
    idx = -1
    f = np.linspace(0, 1, 1024)
    f_lpmm = cpm2lppmm(f, ImgType)
    for i in range(len(f_lpmm)):
        if f_lpmm[i]>freq:
                idx = i
                break
    
    y = [[]]
    count = 0
    
    with open(path, 'r') as f:
        csvreader = csv.reader(f)
        firstRow = False
        for row in csvreader:
            if firstRow:
                # 最初1行は取り除く
                # firstRow=False
                continue
            ytmp = row
            ytmp = list(map(float, ytmp))
            y.append(ytmp)
            count += 1
            if idx>-1 and count == idx:
                # indexの個数分データとったので抜けだす
                break
    return y[1:]

def trimData(data, delArray):
    return np.delete(data, delArray, 1)

def extractData(data, ext):
    return np.array(data)[np.ix_(list(range(np.array(data).shape[0])), ext)]


def main():
    dataPath = sys.argv[1]
    
    data = getData(dataPath, FREQ, ImgType.REAL)
    
    print(np.average(data, axis=0
                     ))
    


if __name__ == "__main__":
    main()