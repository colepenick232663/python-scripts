# Takes a bunch of .csv files and compiles them into one XLSX without displaying anything.  Used for file consolidation

import numpy as np
import pandas as pd
import glob, os
import xlsxwriter
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

path = 'Z:\\DRAWING\\WIP\\Projects\\191 Semi-Auto\\testing\\spot vs slit\\3x3 slit\\'

#finds all filetypes between the '* ' statement and makes them into a list
csvfiles = [f for f in glob.glob(path+'*.csv')]

summary = pd.DataFrame(index=None)  #creates an empty dataframe
summary_file = pd.ExcelWriter(path+'summary.xlsx',engine='xlsxwriter')  #the engine used to build a new excel file
counter = 1  #used for enumeration later

header = ['wavelength','A1','C1','E1','A3','C3','E3','A5','C5','E5']
avg_intensity = []
for file in csvfiles:  #iterates through the list made on line 11
    scan_data = pd.read_csv(file,delimiter=',', header=None, skiprows=13, on_bad_lines='skip')  #imports csv content into dataframe separating commas into different cells
    scan_data.to_excel(summary_file, sheet_name = str(file.removeprefix(path).removesuffix('.CSV')), header = header)
    for i in scan_data:
        avg_intensity.append(sum(scan_data[i])/len(scan_data))
         
#    summary['Scan '+str(counter)] = specdata  #builds each column header to match each scan # sequentially
#    summary['Scan '+str(counter)] = summary['Scan '+str(counter)].astype(float)  #converts from an object into a number
    counter+=1  #sequentially enumerates to captures each unique scan

avg_intensity=np.asarray(avg_intensity).reshape(24, 10)
for i in range(1,len(avg_intensity)):
    np.delete(avg_intensity[i], 0, 0)
#    plt.bar(range(1,len(avg_intensity[0])), avg_intensity[i][1:], color = ['red','green','blue'])
    fig=plt.figure()
    ax=fig.add_subplot(111, projection = '3d')
    X=Y=np.arange(0,3,1)
#    X,Y = np.meshgrid(x,y)
    dxdy=[.5,.5,.5]
    ax.bar3d(X, Y, avg_intensity[i][1:].reshape(3,3), dxdy, dxdy, avg_intensity[i][1:].reshape(3,3))
#    plt.xticks(range(1,len(header)), labels=[x for x in header[1:]])
#    plt.xlabel('grid coordinates')
    plt.xticks(ticks=range(0,3),labels=['A','C','E'])
    plt.yticks(ticks=range(0,3),labels=['1','3','5'])
    plt.xlabel('horizontal position coordinate')
    plt.ylabel('vertical position coordinate')
    ax.set_zlabel('average reflectance (%/nm)')
    ax.view_init(elev=30,azim=45,roll=15)
#    plt.zlabel('average reflectance (%/nm)')
    plt.title('average reflectance on 9-point grid')
    plt.savefig(os.path.join(path+str(csvfiles[i].removeprefix(path).removesuffix('.CSV'))))
    plt.close()

#summary = summary.rename(index = lambda x: x+367)  #recasts column indices to match wavelengths
#summary.to_excel(summary_file,sheet_name = '100 scans')  #writes data into excel sheet and names tab
summary_file.close()  #saves the excel file
