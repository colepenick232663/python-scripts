import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import statistics
from math import log10, floor

filepath = r"Z:\DRAWING\WIP\Projects\191 Semi-Auto\testing\repeatability\other glass"
os.chdir(filepath)

def round_sig(x, sig=3):
    return round(x, sig-int(floor(log10(abs(x))))-1)

def truncate(n):
    return (n*1000)/1000

graphs = []
for scans in os.listdir():
    graphs.append(np.genfromtxt(scans, delimiter = ',', skip_header = 13))
  
mean = []
stdev = []
graphs=np.asarray(graphs,float).reshape(100, 401, 2)
csv_graphs=zip(*graphs)
std_array = []
for i in range(len(graphs[:,:,-1])):
    std_array.append(graphs[:,:,-1][i])
    plt.plot(graphs[:,40:301,0][1], [x for x in graphs[:,40:301,-1][i]])
for i in range(len(graphs[:][0])):
    mean.append(statistics.mean(graphs[:,i,-1]))
    stdev.append(np.std(graphs[:,i,-1]))

filename = "repeatability scan report.csv"
fields = ["wavelength (nm)", "reflectance(s) (%)"]
with open(filename, 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(graphs[:,:,0][1])
    for i in range(len(graphs[:,:,-1])):
        writer.writerow(csv_graphs[:,:,-1][i])
#    writer.writerow(mean)
#    writer.writerow(stdev)
    

repeatability = (max(stdev[20:301])/max(mean[20:301]))

plt.title("100 reflectance scans for repeatability of *alternate* glass \n \n coefficient of variation (420-680nm): " + str(round_sig(repeatability)*100) + "%")
plt.xlabel("wavelength (nm)")
#plt.xlim((420,680))
#plt.ylim((0,1))
plt.ylabel("reflectance (%)")
plt.show()