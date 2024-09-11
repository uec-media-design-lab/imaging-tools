import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import enum
import math
import matplotlib.cm as cm
FREQ= 1
IMG_WIDTH_REAL = 6024
IMG_WIDTH_CG   = 200
SENSOR_WIDTH   = 35.9
LENS_FOCUS_DISTANCE_REAL = 35
LENS_FOCUS_DISTANCE_CG = 250
CAM_TO_IMG_DISTANCE = 300
class ImgType(enum.IntEnum):
    REAL = 0
    CG = 1

realRRs=["rfay", "rfan"]

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
plot_x = [0, 10, 20, 30]
# plot_x = [0, 5, 10, 15, 20, 25, 30]
# plot_x = [0, 5, 10, 15, 20, 25, 30,35,40,45]    # theta
plot_x = [150, 175, 200, 225, 250, 275, 300]    # dist
# plot_x = [0, 10,20,30,40,50,60,70,80,90]    # phi
plotNum = len(plot_x)

def MAPE(data1, data2):
    return np.mean(np.abs((data2 - data1) / data1))

def MAPEcomp(data1, data2):
    return MAPE(np.mean(data1, axis=0), np.mean(data2, axis=0))

# cycles/px -> lp/mm
def cpm2lppmm(x, type):
    return 1.0*x * (IMG_WIDTH_REAL if type==ImgType.REAL else IMG_WIDTH_CG) / (1.0* SENSOR_WIDTH * CAM_TO_IMG_DISTANCE / (LENS_FOCUS_DISTANCE_REAL if type==ImgType.REAL else LENS_FOCUS_DISTANCE_CG))

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

def mtfMeanPlot(data, color, label, marker, line):
    # chain line : 多項式フィッティング結果
    
    means = np.mean(data, axis=0)
    res1 = np.polyfit(plot_x, means, 1)
    res2 = np.polyfit(plot_x, means, 2)
    res3 = np.polyfit(plot_x, means, 3)
    fit1 = np.poly1d(res1)
    fit2 = np.poly1d(res2)
    fit3 = np.poly1d(res3)

    plt.plot(plot_x, means, color=color, linestyle=line, marker=marker, label=label, markersize=8)
    plt.plot(plot_x, fit1(plot_x), color=color, linestyle="-.")
    print(f"label:{label}\tcoef:{res1}")
    

def main():
    suff = "exr/Result_csv/all.csv"
    realDataDirectory = sys.argv[1]
    addSim1 = sys.argv[2]
    addSim2 = sys.argv[3]
    addSim3 = sys.argv[4]
    addSim4 = sys.argv[5]
    
    # data_rfay = getData(realDataDirectory+"all-{}.csv".format(realRRs[0]), FREQ, ImgType.REAL)
    # data_rfan = getData(realDataDirectory+"all-{}.csv".format(realRRs[1]), FREQ, ImgType.REAL)
    data_rfay = getData(sys.argv[1], FREQ, ImgType.REAL)
    data_rfan = getData(sys.argv[2], FREQ, ImgType.REAL)
    data_sim1 = getData(sys.argv[3], FREQ, ImgType.CG)
    data_sim2 = getData(sys.argv[4], FREQ, ImgType.CG)
    data_sim3 = getData(sys.argv[5], FREQ, ImgType.CG)
    data_sim4 = getData(sys.argv[6], FREQ, ImgType.CG)
    
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    trim_real = [0, 2, 4, 6]
    # trim_real = [0,1,2,3,4,5,6]
    # trim_real = [0,1,2,3,4,5,6,7,8,9]
    trim_real = [0,1,2,3,4,5,6]   # dist
    data_rfay = extractData(data_rfay, trim_real)
    data_rfan = extractData(data_rfan, trim_real)
    
    # data_sim4 = extractData(data_sim4, trim_real)
    
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    # xlabel = "Incident angle [°]"
    # xlabel = "Azimuth angle [°]"
    xlabel = "Floating distance [mm]"
    
    fig = plt.figure(figsize=(10,8))
    plt.rcParams["font.family"] = "Source Han Sans JP"
    plt.grid(color="silver", linestyle="dotted", linewidth=1)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel("Average of MTF [au]", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # plt.ylim(0, 0.6)
    
    mtfMeanPlot(data_rfay, "black", "RF-Ay", "o", "-")
    mtfMeanPlot(data_rfan, "black", "RF-AN", "o", ":")
    mtfMeanPlot(data_sim1, cm.hsv(90), "BSSRDF (RF-Ay)", "D", "-")
    mtfMeanPlot(data_sim2, cm.hsv(90), "BSSRDF (RF-AN)", "D", ":")
    mtfMeanPlot(data_sim3, cm.hsv(150), "BRDF (RF-Ay)", "s", "-")
    mtfMeanPlot(data_sim4, cm.hsv(150), "BRDF (RF-AN)", "s", ":")
    # mtfMeanPlot(data_sim1, cm.hsv(30), "sim1", "o")
    # mtfMeanPlot(data_sim2, cm.hsv(90), "sim2", "o")
    # mtfMeanPlot(data_sim3, cm.hsv(150), "sim3", "o")
    # mtfMeanPlot(data_sim4, cm.hsv(210), "sim4", "o")
    
    ## calculate MAPEs
    print("="*20)

    print("MAPE RFAY")
    print(f"RFAy-Ours:\t{MAPEcomp(data_rfay, data_sim1):.4f}")
    print(f"RFAy-Guos:\t{MAPEcomp(data_rfay, data_sim3):.4f}")
    print("MAPE RFAN")
    print(f"RFAN-Ours:\t{MAPEcomp(data_rfan, data_sim2):.4f}")
    print(f"RFAN-Guos:\t{MAPEcomp(data_rfan, data_sim4):.4f}")
    print("="*20)
    
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    fig.savefig("./expMTF2dist-fit1BSSRDF.png")
    # fig.savefig("./expMTF2theta-fit1BSSRDF.png")
    # fig.savefig("./expMTF2phi.png")


if __name__ == "__main__":
    main()