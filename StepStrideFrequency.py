import Utilities as utilities
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import Constants
import StepStrideFrequencyModuleConstants
import peakutils
import math
import random
import pandas as pd
import pickle
import sys

def get_step_stride_frequency(global_peaks, p_spectrum_by_group_type, group_type):
    alpha = 1.0
    # peaks will be stored in this array
    indices = []
    t = 0.1
    f = StepStrideFrequencyModuleConstants.F
    p_spectrum = p_spectrum_by_group_type[group_type]
    #find the start-index
    start_index = next(x[0] for x in enumerate(f) if x[1] >= 0.5)
    #find the end-index
    end_index = next(x[0] for x in enumerate(f) if x[1] >= 2.5)
    while len(indices) < 2:
        #indices = peakutils.indexes(p_spectrum, thres=t, min_dist=1)
        indices = utilities.get_peaks_between(p_spectrum, start_index=start_index, end_index=end_index,thres=t,min_dist=1)
        t = t / 10

    #print f[indices]
    score = float("inf")
    (a, b) = (None, None)
    for i in range(len(indices)):
        f1 = f[indices[i]]
        a1 = p_spectrum[indices[i]]
        left_idx = indices[i-1] if i-1>=0 else 0
        right_idx = indices[i+1] if i+1<len(indices) else len(f)-1
        p1 = np.array([f1, math.sqrt(a1), utilities.get_half_energy(p_spectrum, f, indices[i],left_idx,right_idx)])
        #print "p1=" + str(p1)
        p1_prime = global_peaks[group_type[1]]["p1"]
        #print "========================"
        #print "p1_prime=" + str(p1_prime)
        # print "p1_prime: " + str(p1_prime)
        for j in range(i + 1, len(indices)):
            f2 = f[indices[j]]
            a2 = p_spectrum[indices[j]]
            left_idx = indices[j-1] if j-1>=0 else 0
            right_idx = indices[j+1] if j+1<len(indices) else len(f)-1
            p2 = np.array([f2, math.sqrt(a2), utilities.get_half_energy(p_spectrum, f, indices[j],left_idx,right_idx)])
            p2_prime = global_peaks[group_type[1]]["p2"]
            #print "p2 = " + str(p2)
            #print "p2_prime=" + str(p2_prime)
            #print "========================"
            # minimize this function to find p1 and p2
            new_score = np.linalg.norm(p1 - p1_prime) + np.linalg.norm(p2 - p2_prime) + alpha * np.linalg.norm(2 * p1[0] - p2[0])  # + p1[0]**2 + p2[0]**2
            if new_score < score:
                score = new_score
                (a, b) = (i, j)
    #print f[[indices[a], indices[b]]]
    #print "+++++++++++++++++++++++++++++++++++"
    return {"healthCode_medTimepoint":group_type, "all_peaks": f[indices], "selected_peaks":f[[indices[a], indices[b]]]}

def get_global_peaks():
    (X,Y) = utilities.load_data()
    print "Computing Fourier transform and power spectrum..."
    X1 = []
    Y1 = []
    Z1 = []
    for x in X:
        (fx,x1) = utilities.fourier_transform(x["x"])
        (fy,y1) = utilities.fourier_transform(x["y"])
        (fz,z1) = utilities.fourier_transform(x["z"])

        x1 = utilities.remove_noise(x1,0.5)
        y1 = utilities.remove_noise(y1,0.5)
        z1 = utilities.remove_noise(z1,0.5)
        #interpolate
        (fx,x1) = utilities.interpolate_data(fx,x1)
        (fy,y1) = utilities.interpolate_data(fy,y1)
        (fz,z1) = utilities.interpolate_data(fz,z1)
        x1 = utilities.squared_signal(x1)
        y1 = utilities.squared_signal(y1)
        z1 = utilities.squared_signal(z1)
        X1.append(x1)
        Y1.append(y1)
        Z1.append(z1)

    print "Averaging the power spectrum..."
    power_spectrums_by_class_label_x = defaultdict(list)
    power_spectrums_by_class_label_y = defaultdict(list)
    power_spectrums_by_class_label_z = defaultdict(list)
    for idx,y in enumerate(Y):
        power_spectrums_by_class_label_x[y].append(X1[idx])
        power_spectrums_by_class_label_y[y].append(Y1[idx])
        power_spectrums_by_class_label_z[y].append(Z1[idx])

    avg_power_spectrum_x = {}
    avg_power_spectrum_y = {}
    avg_power_spectrum_z = {}
    for key in power_spectrums_by_class_label_x.keys():
        power_spectrums_by_class_label_x[key] = np.array(power_spectrums_by_class_label_x[key])
        avg_power_spectrum_x[key] = np.mean(power_spectrums_by_class_label_x[key],axis=0)
        avg_power_spectrum_y[key] = np.mean(power_spectrums_by_class_label_y[key],axis=0)
        avg_power_spectrum_z[key] = np.mean(power_spectrums_by_class_label_z[key],axis=0)

    '''for key in avg_power_spectrum.keys():
        indices = peakutils.indexes(avg_power_spectrum[key][5:],thres=0.3,min_dist=1)
        f = np.linspace(0, Constants.sampling_frequency / 2, StepStrideFrequencyModuleConstants.sample_points)[5:]
        print f[indices]
        for index in indices:
            print key + " peaks..."
            print f[index]
            roots,area = utilities.get_half_energy(avg_power_spectrum[key][5:],f,index)
            print roots
            print area
        sys.exit()'''

    global_peaks_grouped_by_key_x = {}
    global_peaks_grouped_by_key_y = {}
    global_peaks_grouped_by_key_z = {}
    for key in avg_power_spectrum_x.keys():
        p = avg_power_spectrum_x[key]
        f = StepStrideFrequencyModuleConstants.F
        # find the start-index
        start_index = next(x[0] for x in enumerate(f) if x[1] >= 0.5)
        # find the end-index
        end_index = next(x[0] for x in enumerate(f) if x[1] >= 2.5)
        indices = []
        t = 0.3
        while len(indices) < 2:
            # indices = peakutils.indexes(p_spectrum, thres=t, min_dist=1)
            indices = utilities.get_peaks_between(p, start_index=start_index, end_index=end_index, thres=t,
                                                  min_dist=1)
            t = t / 10
        #indices = peakutils.indexes(p[start_index:end_index], thres=0.3, min_dist=1)
        print f[indices]

        left_idx = 0
        right_idx = indices[1]
        p1 = np.array([f[indices[0]],math.sqrt(p[indices[0]]),utilities.get_half_energy(p,f,indices[0],left_idx,right_idx)])

        left_idx = indices[0]
        right_idx = indices[2] if len(indices)>2 else Constants.sampling_frequency
        p2 = np.array([f[indices[1]],math.sqrt(p[indices[1]]),utilities.get_half_energy(p,f,indices[1],left_idx,right_idx)])

        global_peaks_grouped_by_key_x[key] = {"p1":p1,
                                    "p2":p2} #p is already a power spectrum
    print "=========================="

    for key in avg_power_spectrum_y.keys():
        p = avg_power_spectrum_y[key]
        f = StepStrideFrequencyModuleConstants.F
        # find the start-index
        start_index = next(x[0] for x in enumerate(f) if x[1] >= 0.5)
        # find the end-index
        end_index = next(x[0] for x in enumerate(f) if x[1] >= 2.5)
        indices = []
        t = 0.3
        while len(indices) < 2:
            indices = utilities.get_peaks_between(p, start_index=start_index, end_index=end_index, thres=t,
                                                  min_dist=1)
            t = t / 10
        print f[indices]
        left_idx = 0
        right_idx = indices[1]
        p1 = np.array([f[indices[0]],math.sqrt(p[indices[0]]),utilities.get_half_energy(p,f,indices[0],left_idx,right_idx)])

        left_idx = indices[0]
        right_idx = indices[2] if len(indices)>2 else Constants.sampling_frequency
        p2 = np.array([f[indices[1]],math.sqrt(p[indices[1]]),utilities.get_half_energy(p,f,indices[1],left_idx,right_idx)])

        global_peaks_grouped_by_key_y[key] = {"p1":p1,
                                    "p2":p2} #p is already a power spectrum
    print "=========================="

    for key in avg_power_spectrum_z.keys():
        p = avg_power_spectrum_z[key]
        f = StepStrideFrequencyModuleConstants.F
        # find the start-index
        start_index = next(x[0] for x in enumerate(f) if x[1] >= 0.5)
        # find the end-index
        end_index = next(x[0] for x in enumerate(f) if x[1] >= 2.5)
        indices = []
        t = 0.3
        while len(indices) < 2:
            indices = utilities.get_peaks_between(p, start_index=start_index, end_index=end_index, thres=t,
                                                  min_dist=1)
            t = t / 10
        print f[indices]
        left_idx = 0
        right_idx = indices[1]
        p1 = np.array([f[indices[0]],math.sqrt(p[indices[0]]),utilities.get_half_energy(p,f,indices[0],left_idx,right_idx)])

        left_idx = indices[0]
        right_idx = indices[2] if len(indices)>2 else Constants.sampling_frequency
        p2 = np.array([f[indices[1]],math.sqrt(p[indices[1]]),utilities.get_half_energy(p,f,indices[1],left_idx,right_idx)])

        global_peaks_grouped_by_key_z[key] = {"p1":p1,
                                    "p2":p2} #p is already a power spectrum

    print global_peaks_grouped_by_key_x
    print "=========================="
    print global_peaks_grouped_by_key_y
    print "=========================="
    print global_peaks_grouped_by_key_z

    with open("global_peaks_x.pkl","w") as f:
        pickle.dump(global_peaks_grouped_by_key_x,f)

    with open("global_peaks_y.pkl","w") as f:
        pickle.dump(global_peaks_grouped_by_key_y,f)

    with open("global_peaks_z.pkl","w") as f:
        pickle.dump(global_peaks_grouped_by_key_z,f)

    sys.exit()
    with open("global_peaks.pkl","r") as f:
        global_peaks_grouped_by_key_x = pickle.load(f)
    print "=================================================="
    '''print 'Loading meta data...'
    meta_data = utilities.get_meta_data()
    meta_data = meta_data[["healthCode","medTimepoint","outbound_walk_json_file"]]
    grouped = meta_data.groupby(by=["healthCode","medTimepoint"])
    avg_pspectrum_by_healthcode_and_class_label = {}
    rows = []
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
        Z = np.array(Z)
        avg_pspectrum_by_healthcode_and_class_label[name] = np.mean(Z,axis=0)
        rows.append(name)'''

    with open("p_spectrum.pkl", "r") as f:
        avg_pspectrum_by_healthcode_and_class_label = pickle.load(f)

    rows = list(avg_pspectrum_by_healthcode_and_class_label.keys())

    #print get_step_stride_frequency(global_peaks_grouped_by_key, avg_pspectrum_by_healthcode_and_class_label,('6b104542-0b0a-4421-a490-568f66c7ebc4', 'Immediately before Parkinson medication'))
    #create validation set
    for i in xrange(3):
        random_indices = random.sample(range(len(rows)), 500)
        df = pd.DataFrame(columns=["healthCode_medTimepoint","all_peaks","selected_peaks"])
        for idx in random_indices:
            result = get_step_stride_frequency(global_peaks_grouped_by_key_x,avg_pspectrum_by_healthcode_and_class_label,rows[idx])
            df = df.append([result])
        df.to_csv("validation-set" + str(i+1)+ ".csv",index=False)

#if __name__ == "__main__":
#    main()

#load the data
X,recordIds = utilities.load_full_data()

#load the global peaks
with open("global_peaks_x.pkl", "r") as f:
    global_peaks_grouped_by_key_x = pickle.load(f)

with open("global_peaks_y.pkl", "r") as f:
    global_peaks_grouped_by_key_y = pickle.load(f)

with open("global_peaks_z.pkl", "r") as f:
    global_peaks_grouped_by_key_z = pickle.load(f)

#load the arrays for estimating mean and sd
with open("half-energies_x.pkl","r") as f:
    half_energies_x = pickle.load(f)
    half_energies_x = np.array(half_energies_x)

half_energies_x_mu = np.mean(half_energies_x)
half_energies_x_sd = np.std(half_energies_x)

with open("half-energies_y.pkl","r") as f:
    half_energies_y = pickle.load(f)
    half_energies_y = np.array(half_energies_y)

half_energies_y_mu = np.mean(half_energies_y)
half_energies_y_sd = np.std(half_energies_y)

with open("half-energies_z.pkl","r") as f:
    half_energies_z = pickle.load(f)
    half_energies_z = np.array(half_energies_z)

half_energies_z_mu = np.mean(half_energies_z)
half_energies_z_sd = np.std(half_energies_z)

with open("amplitudes_x.pkl","r") as f:
    amplitude_x = pickle.load(f)
    amplitude_x = np.array(amplitude_x)

amplitude_x_mu = np.mean(amplitude_x)
amplitude_x_sd = np.std(amplitude_x)

with open("amplitudes_y.pkl","r") as f:
    amplitude_y = pickle.load(f)
    amplitude_y = np.array(amplitude_y)

amplitude_y_mu = np.mean(amplitude_y)
amplitude_y_sd = np.std(amplitude_y)

with open("amplitudes_z.pkl","r") as f:
    amplitude_z = pickle.load(f)
    amplitude_z = np.array(amplitude_z)

amplitude_z_mu = np.mean(amplitude_z)
amplitude_z_sd = np.std(amplitude_z)

with open("frequencies_x.pkl","r") as f:
    frequency_x = pickle.load(f)
    frequency_x = np.array(frequency_x)

frequency_x_mu = np.mean(frequency_x)
frequency_x_sd = np.std(frequency_x)

with open("frequencies_y.pkl","r") as f:
    frequency_y = pickle.load(f)
    frequency_y = np.array(frequency_y)

frequency_y_mu = np.mean(frequency_y)
frequency_y_sd = np.std(frequency_y)

with open("frequencies_z.pkl","r") as f:
    frequency_z = pickle.load(f)
    frequency_z = np.array(frequency_z)

frequency_z_mu = np.mean(frequency_z)
frequency_z_sd = np.std(frequency_z)

print "Running through each record id..."
assert len(X) == len(recordIds)
df = pd.DataFrame(columns=["recordId","axis","group_type","all_peaks","selected_peaks"])
i = 1
for idx,x in enumerate(X):
    (fx, x1) = utilities.fourier_transform(x["x"])
    (fy, y1) = utilities.fourier_transform(x["y"])
    (fz, z1) = utilities.fourier_transform(x["z"])

    x1 = utilities.remove_noise(x1, 0.5)
    y1 = utilities.remove_noise(y1, 0.5)
    z1 = utilities.remove_noise(z1, 0.5)
    # interpolate
    (fx, x1) = utilities.interpolate_data(fx, x1)
    (fy, y1) = utilities.interpolate_data(fy, y1)
    (fz, z1) = utilities.interpolate_data(fz, z1)
    x1 = utilities.squared_signal(x1)
    y1 = utilities.squared_signal(y1)
    z1 = utilities.squared_signal(z1)

    rows = []
    for key in global_peaks_grouped_by_key_x.keys():
        row = {}
        row["recordId"] = recordIds[idx]
        row["axis"] = "x"
        d = utilities.get_step_stride_frequency(global_peaks_grouped_by_key_x, x1, key,mu=np.array([frequency_x_mu,amplitude_x_mu,half_energies_x_mu]),
                                                sd=np.array([frequency_x_sd,amplitude_x_sd,half_energies_x_sd]))
        #d = utilities.get_step_stride_frequency(global_peaks_grouped_by_key_x, x1, key,mu=np.array([0,0,0]),sd=np.array([1,1,1]))
        row.update(d)
        rows.append(row)
    #print rows
    #break
    for key in global_peaks_grouped_by_key_y.keys():
        row = {}
        row["recordId"] = recordIds[idx]
        row["axis"] = "y"
        d = utilities.get_step_stride_frequency(global_peaks_grouped_by_key_y, y1, key, mu=np.array([frequency_y_mu,amplitude_y_mu,half_energies_y_mu]),
                                                sd=np.array([frequency_y_sd,amplitude_y_sd,half_energies_y_sd]))
        #d = utilities.get_step_stride_frequency(global_peaks_grouped_by_key_y, y1, key,mu=np.array([0, 0, 0]),sd=np.array([1, 1, 1]))
        row.update(d)
        rows.append(row)

    for key in global_peaks_grouped_by_key_z.keys():
        row = {}
        row["recordId"] = recordIds[idx]
        row["axis"] = "z"
        d = utilities.get_step_stride_frequency(global_peaks_grouped_by_key_z, z1, key, mu=np.array([frequency_z_mu,amplitude_z_mu,half_energies_z_mu]),
                                                sd=np.array([frequency_z_sd,amplitude_z_sd,half_energies_z_sd]))
        #d = utilities.get_step_stride_frequency(global_peaks_grouped_by_key_z, z1, key,mu=np.array([0,0,0]),sd=np.array([1,1,1]))
        row.update(d)
        rows.append(row)
    df = df.append(rows)
    i = i + 1
    if i > 100:
        break

df.to_csv("Peaks-With-Normalization.csv",index=False)