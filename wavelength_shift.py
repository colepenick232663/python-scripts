#works as intended!
##change "path" in line 13 wherever calibration data is. Make sure the path name is correct
from msilib.schema import File
from re import I
from socket import NI_NOFQDN
import numpy as np
import gsmath
import os
import matplotlib.pyplot as plt
import shutil

#path containing scripts to be imported: saved to local for convenience. Data will also be saved to local path
path = 'C:\gstools\HX0717 calibration data'

#I had to change the names of the single digit channels to have a leading placeholder zero so that the script would process 
##each file in the correct order and assign it to its respective folder
filenames = [os.path.join(path, x) for x in os.listdir(path)]
#sorting files by modification time to fix 100% being the third value. Also ignores .txt file (log) populated during calibration
filenames=[i for i in sorted(filenames, key=os.path.getmtime) if i.endswith('csv')]

#all channels:
channels = ['03','04','05','06','07','08','11','13','14','16','19','20','21','22','23','24','26','27','28','29','30','31','33','34','36','37','38','39','41','42','43','45','46','47','49','51','52','53','54','55','57','59','60','61','62','63']
#nominal wavelength according to ccl
nominal_wl = [595,505,595,395,520,630,780,660,715,630,850,545,545,910,750,637,460,590,590,715,595,6500,940,420,685,620,2700,535,450,735,495,735,525,675,405,760,475,985,700,700,637,430,495,805,525,3000]
#makes a 2d list that assigns the nominal wavelength (y) to each channel (x)
channel_and_wl=[channels, nominal_wl]
#assign unit for displaying on graphs
nominal_wl_title=[f"{n}nm" for n in nominal_wl]

#makes a folder for each channel for which to store the calibration files to be called on by folder when making each graph
for n in channels:
    folder_path=os.path.join(path, f"CH{n}")
    os.mkdir(folder_path)

#defines all files in the path directory, and then ensures that the folders are defined by their directory type
all_files=[os.path.join(path,x) for x in os.listdir(path)]
folders = [i for i in all_files if os.path.isdir(i)==True]

#initiates initial number ni and final number nf to sort filenames into their respective folders in groups of 11. This counts 
##on each file being in sequential order (ascending channel, then by power level)
ni=0
nf=11
#initializes number of folders to go through when moving each channel power data table to the respective folder
n=0


#sorting calibration files into folders
for n in range(0,46,1):
    for i in filenames[ni:nf]:
        shutil.move(i, folders[n])
    ni+=11
    nf+=11

#makes a folder to store all the plots made below. NOTE: haven't figured out how to actually store the plots in this folder: 
##can't figure out how savefig command isn't putting the plots into this folder I created when the path is listed in the savefig syntax
graphs_folder=str('wavelength shift plots')
os.mkdir(graphs_folder)

#labels to display on wavelength shift graphs
annotation=["1%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]

#10 values equally spaced apart, then add the 1% at the 0th index
pwr = np.insert(np.linspace(10,100,num=10), 0, 1)

#make graphs for wavelength shift per channel

#initialize counting variable to assign calling index of nominal_wl
k=0
deltas=[]
dir_name="C:\gstools\wavelength shift plots"
for i in folders:
    k+=1
    wl=[]
    os.chdir(i)
    name=os.path.basename(i)
    for file in sorted(os.listdir(i), key=os.path.getmtime):
    
        data = np.genfromtxt(file, delimiter=',', skip_header=15)
        x = data[:,0]
        y = data[:,1]
        spectrum = gsmath.Analysis(x, y)
        wl += [spectrum.peak_wavelength]

    #assigns min and max wavelengths for each folder
    wl_first=wl[0]
    wl_last=wl[10]
    delta=(wl_last-wl_first)
    deltas.append(delta)
    plt.title('peak wavelength shift for different percent powers of '+str(name) + "\n nominal wavelength: " + str(nominal_wl[k-1])+"nm")
    plt.plot(pwr, wl)
    plt.scatter(pwr, wl)
    plt.ylabel('peak wavelength (nm)')
    plt.xlabel('% power')

    #add annotations. Directory needs to change back every time
    for j, label in enumerate(annotation):
        plt.annotate(label, (pwr[j], wl[j]))
    os.chdir('C:\gstools\wavelength shift plots')
    plt.savefig(str(name)+ ".png")
    plt.close()

#deltas_table is an array of three-tuple arrays whose first index is the nominal wavelength. The second index is the difference in peak 
##wavelength from 1% to 100%. The third value is the name of the channel to be stored and called when labeling graphed points.
deltas_table=list(zip(nominal_wl,deltas,channels))

#sorts deltas_table by nominal wavelength
deltas_table.sort(key=lambda i:i[0])
deltas_table=np.array(deltas_table)
deltas_table=deltas_table[:43]

#coordinates extracted from deltas_table
x=deltas_table[:,0]
y=[float(i) for i in deltas_table[:,1]]

labels=[f"CH{n}" for n in deltas_table[:,2]]

for k, label in enumerate(labels):
    plt.annotate(label,(x[k],y[k]))

#make plot of how much each wavelength changes from 1 to 100%.
plt.plot(x,y)
plt.scatter(x,y)
plt.title('wavelength shift per channel')
plt.ylabel('difference in wavelength from 1%-100% (nm)')
plt.xlabel('wavelength (nm)')
os.chdir('C:\gstools\wavelength shift plots')
plt.savefig(' wavelength shift per channel.png')
plt.show()
