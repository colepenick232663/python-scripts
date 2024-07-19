import os
import numpy as np
import csv
import math
import matplotlib.pyplot as plt

summary1 = r"Z:\DRAWING\WIP\Projects\RS-9 Series\Testing - RS-9\RS-9-1 demo units\Demo#1\verification\channels at 100%\archive (LT3)\summary.csv"
#summary2 = r"Z:\DRAWING\WIP\Projects\RS-9 Series\Testing - RS-9\RS-9-1 demo units\Demo#1\verification\channels at 100%\archive (LT3)\summary.csv"
filepath = r"Z:\DRAWING\WIP\Projects\RS-9 Series\Testing - RS-9\RS-9-1 demo units\Demo#1\verification\spectral peaks (line item 15)"
os.chdir(filepath)
        
def read_csv(filename):
    channels=[]
    with open(filename) as summary:
        next(summary)
        for row in summary:
            channels.append(row)
    return channels

def unfuck_channel_data(channels):
    channels_new = []
    for ch in channels:
        channels_new.append(ch.replace("(","").replace(")","").replace('"', '').strip('\n'))
    return channels_new

def split_lines(channels_new):
    return [x.split(",") for x in channels_new]

def convert_vals(line):
    new_line = []
    for x in line:
        if '.csv' in x:
            new_line.append(x)
        else:
            new_line.append(float(x))
            
    return new_line


def make_cole_list(filename):
    data = read_csv(filename)
    data = unfuck_channel_data(data)
    data = split_lines(data)
    data = [convert_vals(x) for x in data]
        
    return data

nominal_peaks = [385, 637, 720, 590, 420, 6500, 430, 630, 970, 780, 590, 470, 6500, 700, 405, 750, 850, 760, 2700, 365, 525, 720, 599, 375, 395, 680, 525, 615, 460, 690, 450, 735, 940, 700, 539, 415, 568, 590, 660, 505, 810, 490, 530]
channel_num = [1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]

wl_deltas = []
actual_peaks = [] 
summ = make_cole_list(summary1)
failure_channels=[]

for i in summ[:][:]:
    wl_deltas.append(nominal_peaks[summ.index(i)]-i[1])
    actual_peaks.append(i[1])
    if wl_deltas[summ.index(i)] > 10 or wl_deltas[summ.index(i)] < -10:
        failure_channels.append(channel_num[summ.index(i)])
        
#remove wl_deltas for white channels because nominal peak value is stored as CCT not wl        
for i in (5,12,18,25,39):
    wl_deltas[i]=0
    
#write report displaying the channels that failed, the nominal peaks, the actual peaks, and their difference
report_file = open(os.path.join(filepath, "report.txt"), 'w')
report = str("channels whose peak wavelength is > +-10nm from the nominal peak wavelength: " + str(failure_channels) +
             "\n \n nominal peaks: " + str(nominal_peaks) + "\n \n measured peaks: " + str(actual_peaks) + "\n \n differences: " + str(wl_deltas) +
             "\n \n 2700K CCT warm white x,y: " + str(summ[18][-2]) + " " + str(summ[18][-1]) +
             "\n \n 6500K CCT cool white CH7 x,y: " + str(summ[5][-2]) + " " + str(summ[5][-1]) +
             "\n \n 6500K CCT cool white CH14 x,y: " + str(summ[12][-2]) + " " + str(summ[12][-1]))
n = report_file.write(report)
report_file.close()


plt.scatter(channel_num, wl_deltas)
plt.hlines(10,0, len(channel_num), color = 'red')
plt.hlines(-10,0,len(channel_num), color = 'red')
plt.xticks(np.arange(min(channel_num),max(channel_num)+1,1.0))
for i,(x,y) in enumerate(zip(channel_num,wl_deltas)):
    plt.annotate(f"CH{channel_num[i]} \n {nominal_peaks[i]}nm", xy = (channel_num[i],wl_deltas[i]), fontsize = 7)
    if wl_deltas[i] < -10 or wl_deltas[i]> 10:
        plt.scatter(channel_num[i], wl_deltas[i], color = "red")
plt.title("wavelength deltas (white CCTs CH7,14,20; CH25, 39 omitted)")
plt.xlabel("channel")
plt.ylabel("difference between nominal and measured (nm)")
plt.savefig(os.path.join(filepath,"wavelength deltas (white CCTs, CH7, 14, 20; CH39 omitted"))
plt.show()

#percent_error = []
#for data in (range(len(channels1))):
#    for spec in [x for x in range(len(channels1[data][:])) if x != 0]:

#        channels1[data][spec]=channels1[data][spec][:10]
#        channels2[data][spec]=channels2[data][spec][:10]
#        percent_error.append(((float(channels2[data][spec].strip('"()\n'))-float(channels1[data][spec].strip('"()\n')))/(float(channels1[data][spec].strip('"()\n')))*100))
        
#print(percent_error)

