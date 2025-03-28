import os
import matplotlib.pyplot as plt
import re
import numpy as np

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


tgr = []
mong = []
monr = []
scp = []
ccl = []
ch = []
active_ch = []
max_I = []
wl = []
grp = []
active_indices = []
active_wl = []
pwr = []
filepath = r"Z:\DRAWING\WIP\Projects\RS-9 Series\Testing - RS-9\monitor performance"
os.chdir(filepath)
with open('new monitor test.txt', 'r') as file:
    filelist=file.readlines()
    for line in filelist:
        # SCP
        if "SCP" in line and len(line)>8:
            line=line.split()
            scp.append(f"CH{line[3]} at {line[4]}% power")
        # MONG
        if "l(" in line:
            mong.append(str(line.replace('l(0)\n','')).split())
        # TGR
        if line.count(',')>5:
            line.split(',')
            tgr.append(re.sub(r'd[1-3]_gain:','',line).replace('\n','').replace('\t','').replace('nan','0'))
        # MONR
        if line.count(',')==2:
            monr.append(line.split(','))
        # CCL
        if line.count(',')==3:
            ccl.append(line.split(','))
        # USN
        if "HX" in line:
            usn = str(line)

for x in np.asarray(ccl, int).reshape(len(ccl), len(ccl[0])):
    ch.append(x[0])
    max_I.append(x[1])
    wl.append(x[2])
    grp.append(x[3])
    if x[2] != 0:
        active_ch.append(x[0])
        active_wl.append(x[2])

index_array=np.argsort(active_wl)
index_array=np.delete(index_array, [-3,-2,-1])
active_wl.sort()
chromatic = np.delete(active_wl,[-3,-2,-1])

monr=np.asarray(monr, float).reshape(len(active_ch),int(len(monr)/len(active_ch)),len(monr[0]))
tgr=np.fliplr(np.asarray(tgr[0].split(',')).reshape(len(monr[0][0]),int(len(tgr[0].split(','))/len(monr[0][0]))))
power = ['1','10','20','30','40','50','60','70','80','90','100']
mong=np.asarray(mong, int).reshape(len(active_ch),len(power))

#sort monr by wavelength for plotting
monr_sorted = []
i = 0;
for x in monr:
    while i < 40:
        monr_sorted.append(monr[index_array[i]])
        i+=1
monr_sorted = np.asarray(monr_sorted,float)

#each monitor value multiplied by the corresponding (tgr) gain ratio reported by mong, sampled into a new array. I think monr already does this so
##this code is meaningless
# monr_new = []
# for channel in monr:
#     for pwr in channel:
#          for det in pwr:
#             monr_new.append(det*float(tgr[int(np.where(monr==det)[2][0])][int(str(mong[int(np.where(monr==channel)[0][0])][int(np.where(monr==pwr)[1][0])])[int(np.where(monr==det)[2][0])])]))
#            
# monr_new=np.asarray(monr_new, float).reshape(len(monr),len(monr[0]),len(monr[0][0]))


channel_RS7 = ['3','4','5','6','7','8','11','13','16','19','20','21','22','23','26','27','28','29','30','31','33','34','36','37','38','39','41','42','43','45','46','47','49','51','52','53','54','55','57','59','60','61','62','63']
wl_RS7 = ['595','505','595','395','520','630','780','660','630','850','545','545','910','750','460','590','590','715','595','6500','940','420','685','620','2700','535','450','730','495','730','525','675','405','760','475','985','700','700','637','430','495','805','525','3000']

#creates a list based on RS9 that identifies the channel index of the corresponding wavelength in RS7
index_map = []
RS7_simWL = []
for wls in wl_RS7:
    if wls in active_wl:
        index_map.append(active_wl.index(wls))
        RS7_simWL.append(wl_RS7.index(wls))
    
RS9_WL = []
for index in index_map:
    RS9_WL.append(wl[index])

print(index_map)
print(RS7_simWL)

# with open("RS-7-1","a") as txt:
#     for ch in [24,34,46,31,53,23,19,33]:
#         txt.write("channel " + f"{ch}" + " monitor performance: \n" + str(monr[find_indices(active_ch, lambda e: e==ch)])+ "\n")

plt.plot(chromatic,[x for x in monr_sorted[:,:,1]], 'o')
plt.title("photodetector feedback ADC counts by wavelength")
plt.xlabel("wavelength (nm)")
#plt.xticks(x for x in chromatic)
#
#plt.yscale('log')
#
plt.ylabel("counts")
plt.legend([x+"%" for x in power], loc='upper left')
plt.show()
#plt.savefig(monitor_performance_graphs + "/graph name here")