import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import StepStrideFrequencyModuleConstants
import sys

#load the global peaks
global_peaks = {}
with open("global_peaks.pkl","r") as f:
    global_peaks = pickle.load(f)

print global_peaks
sys.exit()
#load the frequency points
f = StepStrideFrequencyModuleConstants.F[5:]

#load the power spectrum file
with open("p_spectrum.pkl","r") as f:
    p_spectrum = pickle.load(f)

for key, val in p_spectrum.iteritems():
    print type(key[0])
    print key[1]
    break

#sys.exit()
#run on validation set1
set1 = pd.read_csv("validation-set1.csv")
for idx,row in set1.iterrows():
    str = row["healthCode_medTimepoint"]
    str = str.replace("(","")
    str = str.replace(")","")
    tokens = str.split(",")
    str1 = tokens[0].replace("\'","")
    str2 = tokens[1].replace("\"","")
    print type(str1)
    print str2
    t = (str1,str2)
    print p_spectrum.has_key(t)
    break
    #print key
    #plt.plot(f,p_spectrum[row[0]][5:])
    #plt.show()