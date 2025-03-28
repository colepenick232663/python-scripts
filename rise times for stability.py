import os
#make sure the interpreter is the same as what's installed
#see run -> configuration manager
import matplotlib.pyplot as plt
import numpy as np
import statistics
import math

def truncate(n):
    return (n*1000)/1000

filepath = "Z:/DRAWING/WIP/Projects/RS-9 Series/Testing - RS-9/RS-9-1 demo units/Demo#1/verification/illumination stability (line item 6)/all channels at 50%"
os.chdir(filepath)
#initialize list to collect data from .CSV files
rise_times = []
#listdir will sort wrong unless there's a 0 placeholder in the 10s place for the first 9 channels
for channels in os.listdir():
    rise_times.append(np.genfromtxt(channels, delimiter = ',', skip_header = 11, usecols = [0,1]))

#creates a "rise time graphs" folder to store all the rise time graphs. Delete current folder when running script
rise_time_graphs = os.path.join(filepath, "rise time graphs")
os.mkdir(rise_time_graphs)

#list to array for reshape
rise_times=np.array(rise_times) 
#sort into 43 channels with 8192 samples and time and voltage values for each sample
rise_times=rise_times.reshape(43,8192,2)
#format: rise_times[channels][sample][element (t="0",V="1")]

#quantify stability after 50ms; every sample is .00012207s
#tune/optimize stabilization time; <.05s is good
stabilization_time_s = .05
total_samples = len(rise_times[0,:])
time_per_sample_s = 1/total_samples
#50ms =409.01049 samples
stable_sample_index = int(stabilization_time_s/time_per_sample_s)

#for saving each figure lby the correct channel name, rather than just the index of each channel from rise_times
channel_number = [1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]

#initialize list of 
stability_per_channel = []
#initialize failure channels list that lists which channels failed to be 99% stable after *<50ms
failure_channels = []
#this number determines within what percent to allow the standard deviation to vary.
custom_stability = 1
for channel in range(len(rise_times)):
    stability_per_channel.append(truncate(statistics.stdev(rise_times[channel,stable_sample_index:total_samples,1])))
    plt.plot(rise_times[channel,:,0],rise_times[channel,:,1])
    plt.xlabel("samples (T=1s)")
    plt.ylabel("voltage response (V), autoranged")
    if stability_per_channel[channel]*100>custom_stability:
        failure_channels.append(channel_number[channel])
        plt.title("Channel " + str(channel_number[channel]) + "\n FAILURE: " + str(int(100-(stability_per_channel[channel]*100))) + "% stability after " + str(stabilization_time_s) + "s")
    else:
        plt.title("Channel " + str(channel_number[channel]) + "\n PASS: " + str(int(100-(stability_per_channel[channel]*100))) + "% stability after " + str(stabilization_time_s) + "s")
    plt.savefig(rise_time_graphs + "/channel " + str(channel_number[channel]) + " rise time")
    plt.close()

report_file = open(os.path.join(rise_time_graphs,"report.txt"),"w")
report = str("the following channels are < " + str(100-custom_stability) + " stable within " + str(stabilization_time_s) + " seconds: " + str(failure_channels))
n = report_file.write(report)
report_file.close()

