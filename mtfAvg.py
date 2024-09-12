import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import enum
import math
import matplotlib.cm as cm
import glob
import os

FREQ= 1
IMG_WIDTH_REAL = 6024 #画像の横幅
IMG_WIDTH_CG   = 1080#1080 #画像の横幅
IMG_WIDTH = 200
SENSOR_WIDTH   = 36 #35.6
LENS_FOCUS_DISTANCE = 35
CAM_TO_IMG_DISTANCE = 300



class ImgType(enum.IntEnum):
    REAL = 0
    CG = 1

# cycles/px -> lp/mm
def cpm2lppmm(x, type):
    return 1.0*x * (IMG_WIDTH_REAL if type==ImgType.REAL else IMG_WIDTH_CG) / (1.0* SENSOR_WIDTH * CAM_TO_IMG_DISTANCE / LENS_FOCUS_DISTANCE)
    #return 1.0*x * IMG_WIDTH / (1.0* SENSOR_WIDTH * CAM_TO_IMG_DISTANCE / LENS_FOCUS_DISTANCE)

def getData(path, freq, ImgType):
    idx = -1
    f = np.linspace(0, 1, 512)
    f_lpmm = cpm2lppmm(f, ImgType)
    #print(f_lpmm)
    
    #print(path.replace('.csv',"")+"_lp.csv")
    #print(path)
    #quit()
    
    for i in range(len(f_lpmm)):
        if f_lpmm[i]>freq:
                idx = i
                break
    
    y = [[]]
    count = 0
    
    with open(path, 'r') as f:
        csvreader = csv.reader(f)
        firstRow = True
        for row in csvreader:
            if firstRow:
                # 最初1行は取り除く
                firstRow=False
                continue
            ytmp = row
            ytmp = list(map(float, ytmp))
            y.append(ytmp)
            count += 1
            if idx>-1 and count == idx:
                # indexの個数分データとったので抜けだす
                break
            #print(ytmp)
    #quit()
    addlpdata(path, f_lpmm)
    
            
    return y[1:]

def addlpdata(path, f_lpmm):
    with open(path, 'r', newline='') as inf, open(path.replace('.csv',"")+"_lp.csv", 'w', newline='') as ouf:
        
        csvreader = csv.reader(inf)
        csvwriter = csv.writer(ouf)
        
        #headers = next(csvreader)
        #headers.append("lp/mm")
        #csvwriter.writerow(headers)
        firstRow = True
        for i, row in enumerate(csvreader):
            #print(row)
            if firstRow:
                # 最初1行は取り除く
                row.append("lp/mm")
                csvwriter.writerow(row)
                firstRow=False
                
                continue
            #print(row)
            row.append(f_lpmm[i-1])
            csvwriter.writerow(row)
            

def trimData(data, delArray):
    return np.delete(data, delArray, 1)

def extractData(data, ext):
    return np.array(data)[np.ix_(list(range(np.array(data).shape[0])), ext)]

def multiget(datapaths, Type):
    datas = glob.glob(datapaths + "/*.csv")
    lpdatas = []
    #print(datas)
    for dataPath in datas:
        if (not '_lp' in dataPath) and (not 'avgmtfs' in dataPath):
            #print(dataPath)
            #quit()
            #data = getData(dataPath, FREQ, ImgType.REAL)
            data = getData(dataPath, FREQ, Type)
            lpavg = np.average(data, axis=0)
            #print(os.path.splitext(os.path.basename(dataPath))[0])
            #print([os.path.splitext(os.path.basename(dataPath))[0], lpavg[1]])
            lpdatas.append([os.path.splitext(os.path.basename(dataPath))[0], lpavg[1]])
            print(lpavg)
    #print(lpdatas)
    createavgmtfs(datapaths, lpdatas)

def createavgmtfs(datapath, lpdatas):
    outputfile=datapath + "/avgmtfs.csv"
    with open(outputfile, 'w', newline='') as ouf:
        csvwriter = csv.writer(ouf)
        csvwriter.writerows(lpdatas)
        
        #firstRow = True
        #for i, row in enumerate(lptadas):
        #    #print(row)
        #    if firstRow:
        #        # 最初1行は取り除く
        #        row.append(["filename","avgmtf"])
        #        csvwriter.writerow(row)
        #        firstRow=False    
        #        continue
        #    #print(row)
        #    row.append(lptadas[i-1])
        #    csvwriter.writerow(row)
    

def main():
    dataPath = sys.argv[1]
    #multiget(dataPath, ImgType.REAL)
    multiget(dataPath, ImgType.CG)
    
    #data = getData(dataPath, FREQ, ImgType.REAL)
    #data = getData(dataPath, FREQ, ImgType.CG)
    
    #print(np.average(data, axis=0))
    


if __name__ == "__main__":
    main()