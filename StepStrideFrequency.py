import Utilities as utilities
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import Constants
import StepStrideFrequencyModuleConstants
import peakutils
import sys
(X,Y) = utilities.load_data()
print "Computing Fourier transform and power spectrum..."
Z = []
for x in X:
    (f,a) = utilities.fourier_transform(x["z"])
    #a[0] = 0
    a = utilities.remove_noise(a, 0.5)
    #interpolate
    (f,a) = utilities.interpolate_data(f,a)
    #print len(a)
    #a = utilities.remove_noise(a,0.5)
    a = utilities.squared_signal(a)
    Z.append(a)

print "Averaging the power spectrum..."
power_spectrums_by_class_label = defaultdict(list)
for idx,y in enumerate(Y):
    power_spectrums_by_class_label[y].append(Z[idx])

avg_power_spectrum = {}
for key in power_spectrums_by_class_label.keys():
    power_spectrums_by_class_label[key] = np.array(power_spectrums_by_class_label[key])
    avg_power_spectrum[key] = np.mean(power_spectrums_by_class_label[key],axis=0)

#print avg_power_spectrum['Immediately before Parkinson medication'].shape
#print np.linspace(0,Constants.sampling_frequency,StepStrideFrequencyModuleConstants.sample_points)[5:]

global_peaks_grouped_by_key = {}
i = 1
for key in avg_power_spectrum.keys():
    indices = peakutils.indexes(avg_power_spectrum[key][5:],thres=0.3,min_dist=1)
    f = np.linspace(0,Constants.sampling_frequency/2,StepStrideFrequencyModuleConstants.sample_points)[5:]
    p = avg_power_spectrum[key][5:]
    #print np.array([ f[indices[0]], p[indices[0]], p[indices[0]]*p[indices[0]] ])
    #print np.array([f[indices[1]], p[indices[1]], p[indices[1]] * p[indices[1]]])
    global_peaks_grouped_by_key[key] = {"p1":np.array([f[indices[0]]]),
                                 "p2":np.array([f[indices[1]]])}
    #print f[indices]
    #plt.subplot(4,1,i)
    #plt.plot(np.linspace(0,Constants.sampling_frequency/2,StepStrideFrequencyModuleConstants.sample_points)[5:],avg_power_spectrum[key][5:])
    #plt.title(key)
    i += 1

#plt.show()
print 'Loading meta data...'
meta_data = utilities.get_meta_data()
meta_data = meta_data[["healthCode","medTimepoint","outbound_walk_json_file"]]
grouped = meta_data.groupby(by=["healthCode","medTimepoint"])
avg_pspectrum_by_healthcode_and_class_label = {}
alpha = 0.0
for name, group in grouped:
    series = group["outbound_walk_json_file"]
    Z = []
    for idx,value in series.iteritems():
        x = utilities.get_data(value)
        (f, a) = utilities.fourier_transform(x["z"])
        a = utilities.remove_noise(a, 0.5)
        #interpolate
        (f, a) = utilities.interpolate_data(f, a)
        a = utilities.squared_signal(a)
        Z.append(a)
        #plt.plot(f,a)
        #plt.show()
        #break
    Z = np.array(Z)
    avg_pspectrum_by_healthcode_and_class_label[name] = np.mean(Z,axis=0)
    print "================="
    print name
    indices = peakutils.indexes(avg_pspectrum_by_healthcode_and_class_label[name], thres=0.1, min_dist=1)
    #print indices
    #print len(indices)
    if len(indices) > 2:
        print np.linspace(0, Constants.sampling_frequency / 2, StepStrideFrequencyModuleConstants.sample_points)[indices]
        score = float("inf")
        (a,b) = (None,None)
        for i in range(len(indices)):
            f1 = np.linspace(0,Constants.sampling_frequency/2,StepStrideFrequencyModuleConstants.sample_points)[indices[i]]
            a1 = avg_pspectrum_by_healthcode_and_class_label[name][indices[i]]
            p1 = np.array([f1])
            p1_prime = global_peaks_grouped_by_key[name[1]]["p1"]
            #print "p1_prime: " + str(p1_prime)
            for j in range(i+1,len(indices)):
                f2 = np.linspace(0,Constants.sampling_frequency/2,StepStrideFrequencyModuleConstants.sample_points)[indices[j]]
                a2 = avg_pspectrum_by_healthcode_and_class_label[name][indices[i]]
                p2 = np.array([f2])
                p2_prime = global_peaks_grouped_by_key[name[1]]["p2"]
                #print "p2_prime" + str(p2_prime)
                #minimize this function to find p1 and p2
                new_score = np.linalg.norm(p1-p1_prime) + np.linalg.norm(p2-p2_prime) + alpha*np.linalg.norm(2*p1[0] - p2[0])
                #print str(new_score) + "," + str(p1) + "," + str(p2)
                if new_score < score:
                    score = new_score
                    (a,b) = (i,j)
        #print "a = " + str(a)
        #print "b = " + str(b)
        print np.linspace(0, Constants.sampling_frequency / 2, StepStrideFrequencyModuleConstants.sample_points)[list([indices[a],indices[b]])]
    else:
        print np.linspace(0,Constants.sampling_frequency/2,StepStrideFrequencyModuleConstants.sample_points)[indices]

    '''if name == ('02f42f94-f1bd-4a90-8b98-8fea8680a7d7', 'Another time'):
        plt.plot(np.linspace(0,Constants.sampling_frequency/2,StepStrideFrequencyModuleConstants.sample_points),
             avg_pspectrum_by_healthcode_and_class_label[name])
        plt.plot(np.linspace(0,Constants.sampling_frequency/2,StepStrideFrequencyModuleConstants.sample_points),avg_power_spectrum[name[1]],'--')
        plt.title(str(name))
        plt.show()
        raw_input()'''
    print "================="
    #break