import Constants
import StepStrideFrequencyModuleConstants
import numpy as np
import pandas as pd
import os
from scipy.fftpack import fft
from collections import defaultdict
from scipy.interpolate import InterpolatedUnivariateSpline
import pickle
import math
import matplotlib.pyplot as plt
import peakutils
import sys

'''Finds peaks in y between start_index to end_index'''
def get_peaks_between(y, start_index, end_index, thres, min_dist):
    assert start_index < end_index
    indices = peakutils.indexes(y[start_index:end_index],thres=thres,min_dist=min_dist)
    indices = [start_index + index for index in indices]
    return indices

def fourier_transform(x):
    N = len(x)
    x = np.array(x)
    #x = x - np.mean(x)
    y = fft(x)
    a = np.abs(y[0:N//2])
    f = np.linspace(0,Constants.sampling_frequency/2,N//2)
    return (f,a)

def squared_signal(x):
    return x**2

def remove_noise(x,l):
    x_new = np.array([i-l if i > l else 0 for i in x])
    return x_new

def interpolate_data(f,a):
    x = np.linspace(0,Constants.sampling_frequency/2,StepStrideFrequencyModuleConstants.sample_points)
    ius = InterpolatedUnivariateSpline(f,a)
    return (x,ius(x))

def default_dict():
    return np.zeros(shape=(StepStrideFrequencyModuleConstants.sample_points,1),dtype=float)

def get_meta_data():
    return pd.read_csv(Constants.meta_data_location)

'''This method reads the meta-data file and retrieves maximum possible equal sized datapoints for
class label. If the data is pickled in X.pkl and Y.pkl, it will directly load from that file'''
def load_data():
    print "Loading the data..."
    X = None
    class_label = None
    if os.path.isfile('X.pkl'):
        with open('X.pkl','r') as f:
            X = pickle.load(f)
    if os.path.isfile('Y.pkl'):
        with open('Y.pkl','r') as f:
            class_label = pickle.load(f)

    if X is not None and class_label is not None:
        return (X,class_label)
    meta_data = pd.read_csv(Constants.meta_data_location)
    medTimepointToJsonFiles = defaultdict(list)
    labels = meta_data["medTimepoint"]
    file_location = meta_data["recordId"]
    for label, file_location in zip(labels, file_location):
        if type(label) is float:
            continue
        else:
            medTimepointToJsonFiles[label].append(file_location)

    (size, label) = min([(len(medTimepointToJsonFiles[key]), key) for key in medTimepointToJsonFiles.keys()])
    X = []
    class_label = []
    for key in medTimepointToJsonFiles.keys():
        for file_location in medTimepointToJsonFiles[key][:size]:
            #if file_location.endswith(".tmp"):
            if os.path.isfile(Constants.walking_outbound_path + "/" + file_location + "/deviceMotion_walking_outbound.json.items.csv"):
              x= pd.read_csv(Constants.walking_outbound_path + "/" + file_location + "/deviceMotion_walking_outbound.json.items.csv")
              X.append(x)
              class_label.append(key)
            '''else:
                print file_location'''

    '''meta_data = pd.read_csv(Constants.meta_data_location)
    class_label_map = {}
    for index,row in meta_data.iterrows():
        class_label_map[row["outbound_walk_json_file"]] = row["medTimepoint"]
    file_names = os.listdir(Constants.walking_outbound_path)
    for file_name in file_names:
        if file_name.endswith(".tmp"):
            x = pd.read_json(Constants.walking_outbound_path + "/" + file_name)
            X.append(x)
            class_label.append(class_label_map[file_name])'''
    with open('X.pkl','w') as f:
        pickle.dump(X,f)
    with open('Y.pkl','w') as f:
        pickle.dump(class_label,f)
    return X,class_label

'''This method reads the meta-data file and retrieves ALL the datapoints i.e . 
If the data is pickled in X_full.pkl, it will directly load from that file'''
def load_full_data():
    print "Loading the data..."
    X = None
    recordIds = None
    if os.path.isfile('X_full.pkl'):
        with open('X_full.pkl', 'r') as f:
            X = pickle.load(f)

    if os.path.isfile('recordIds_full.pkl'):
        with open('recordIds_full.pkl','r') as f:
            recordIds = pickle.load(f)
    if X is not None and recordIds is not None:
        return X, recordIds

    meta_data = pd.read_csv(Constants.meta_data_location)
    if X is None:
        X = []
        for file_location in meta_data["recordId"]:
            if os.path.isfile(Constants.walking_outbound_path + "/" + file_location + "/deviceMotion_walking_outbound.json.items.csv"):
                x = pd.read_csv(Constants.walking_outbound_path + "/" + file_location + "/deviceMotion_walking_outbound.json.items.csv")
                X.append(x)
        with open('X_full.pkl', 'w') as f:
            pickle.dump(X, f)

    if recordIds is None:
        recordIds = []
        for file_location in meta_data["recordId"]:
            if os.path.isfile(Constants.walking_outbound_path + "/" + file_location + "/deviceMotion_walking_outbound.json.items.csv"):
                recordIds.append(file_location)
        with open('recordIds_full.pkl', 'w') as f:
            pickle.dump(recordIds, f)

    return X, recordIds


'''@:param idx is the index for which to find half-energy
    prev_idx index of the previous peak
    next_idx index of the next peak'''
def get_half_energy(p_spectrum,f,idx,prev_idx,next_idx):
    a_spectrum = np.sqrt(p_spectrum)
    p_value = a_spectrum[idx]
    q_value = p_value/2
    ius = InterpolatedUnivariateSpline(f,a_spectrum-q_value)
    p_ius = InterpolatedUnivariateSpline(f,p_spectrum)
    roots = ius.roots()
    if len(roots) == 0:
        l_limit = find_minima(p_spectrum,f,prev_idx,idx,idx)
        r_limit = find_minima(p_spectrum,f,idx,next_idx,idx)
    elif len(roots) == 1:
        #check the root if it lies on the left side or the right side
        if roots[0] < f[prev_idx]:
            l_limit = find_minima(p_spectrum,f,prev_idx,idx,idx)
            r_limit = find_minima(p_spectrum,f,idx,next_idx,idx)
        elif roots[0] >= f[prev_idx] and roots[0] < f[idx]:
            l_limit = roots[0]
            #find minina on the right side
            r_limit = find_minima(p_spectrum,f,idx,next_idx,idx)
        elif roots[0] >= f[idx] and roots[0] <= f[next_idx]:
            r_limit = roots[0]
            l_limit = find_minima(p_spectrum,f,prev_idx,idx,idx)
        else:
            l_limit = find_minima(p_spectrum, f, prev_idx, idx, idx)
            r_limit = find_minima(p_spectrum, f, idx, next_idx, idx)
    elif len(roots) >= 2:
        #find left-closest root
        t = [(f[idx]-root,root) for root in roots if root < f[idx]]
        if len(t) == 0:
            l_limit = find_minima(p_spectrum, f, prev_idx, idx, idx)
        else:
            (d,l_limit) = min(t)
            if l_limit <= f[prev_idx]:
                l_limit = find_minima(p_spectrum,f,prev_idx,idx,idx)

        #find right-closest root
        t = [(root-f[idx],root) for root in roots if root > f[idx]]
        if len(t) == 0:
            r_limit = find_minima(p_spectrum, f, idx, next_idx, idx)
        else:
            d,r_limit = min(t)
            if r_limit >= f[next_idx]:
                r_limit = find_minima(p_spectrum,f,idx,next_idx,idx)
        '''diff = np.array([abs(root - f[idx]) for root in roots])
        sorted_indices = np.argsort(diff)
        i1,i2 = sorted_indices[0],sorted_indices[1]
        if f[i1] >= f[i2]:
            r_limit = roots[i1]
            l_limit = roots[i2]
        else:
            r_limit = roots[i2]
            l_limit = roots[i1]'''

    assert  r_limit > l_limit
    if r_limit is None:
        print "r_limit is None"
        sys.exit()
    if l_limit is None:
        print "l_limit is None"
        sys.exit()

    num = int((r_limit - l_limit)*StepStrideFrequencyModuleConstants.sample_points/(Constants.sampling_frequency/2))
    x = np.linspace(l_limit,r_limit,num=num)
    return np.trapz(p_ius(x),x)

#finds minima in y between a and b closest to point f[p]
def find_minima(y,f,a,b,p):
    assert a < b
    #find minima i.e multiply by -1 and find maxima
    indices = peakutils.indexes(-1*y[a:b+1])
    if len(indices) == 0:
        idx = np.argmin(y[a:b+1])
        return f[a+idx]
    elif len(indices) == 1:
        return f[a+indices[0]]
    else:
        (d,i) = min([(abs(f[a+idx]-f[p]),idx) for idx in indices])
        return f[a+i]

'''This method calculates the global peaks for the four labels and pickles the output'''
def get_global_peaks():
    (X,Y) = load_data()
    print "Computing Fourier transform and power spectrum..."
    X1 = []
    Y1 = []
    Z1 = []
    for x in X:
        (fx,x1) = fourier_transform(x["x"])
        (fy,y1) = fourier_transform(x["y"])
        (fz,z1) = fourier_transform(x["z"])
        x1 = remove_noise(x1,0.5)
        y1 = remove_noise(y1,0.5)
        z1 = remove_noise(z1,0.5)

        #interpolate
        (fx,x1) = interpolate_data(fx,x1)
        (fy,y1) = interpolate_data(fy,y1)
        (fz,z1) = interpolate_data(fz,z1)
        x1 = squared_signal(x1)
        y1 = squared_signal(y1)
        z1 = squared_signal(z1)
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
            indices = get_peaks_between(p, start_index=start_index, end_index=end_index, thres=t,
                                                  min_dist=1)
            t = t / 10

        print f[indices]

        left_idx = 0
        right_idx = indices[1]
        p1 = np.array([f[indices[0]],math.sqrt(p[indices[0]]),get_half_energy(p,f,indices[0],left_idx,right_idx)])

        left_idx = indices[0]
        right_idx = indices[2] if len(indices)>2 else Constants.sampling_frequency
        p2 = np.array([f[indices[1]],math.sqrt(p[indices[1]]),get_half_energy(p,f,indices[1],left_idx,right_idx)])

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
            indices = get_peaks_between(p, start_index=start_index, end_index=end_index, thres=t,
                                                  min_dist=1)
            t = t / 10
        print f[indices]
        left_idx = 0
        right_idx = indices[1]
        p1 = np.array([f[indices[0]],math.sqrt(p[indices[0]]),get_half_energy(p,f,indices[0],left_idx,right_idx)])

        left_idx = indices[0]
        right_idx = indices[2] if len(indices)>2 else Constants.sampling_frequency
        p2 = np.array([f[indices[1]],math.sqrt(p[indices[1]]),get_half_energy(p,f,indices[1],left_idx,right_idx)])

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
            indices = get_peaks_between(p, start_index=start_index, end_index=end_index, thres=t,
                                                  min_dist=1)
            t = t / 10
        print f[indices]
        left_idx = 0
        right_idx = indices[1]
        p1 = np.array([f[indices[0]],math.sqrt(p[indices[0]]),get_half_energy(p,f,indices[0],left_idx,right_idx)])

        left_idx = indices[0]
        right_idx = indices[2] if len(indices)>2 else Constants.sampling_frequency
        p2 = np.array([f[indices[1]],math.sqrt(p[indices[1]]),get_half_energy(p,f,indices[1],left_idx,right_idx)])

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
    #create validation set
    for i in xrange(3):
        random_indices = random.sample(range(len(rows)), 500)
        df = pd.DataFrame(columns=["healthCode_medTimepoint","all_peaks","selected_peaks"])
        for idx in random_indices:
            result = get_step_stride_frequency(global_peaks_grouped_by_key_x,avg_pspectrum_by_healthcode_and_class_label,rows[idx])
            df = df.append([result])
        df.to_csv("validation-set" + str(i+1)+ ".csv",index=False)

'''This method is used to find step and stride frequency for each p_spectrum'''
def get_step_stride_frequency(global_peaks, p_spectrum, group_type, mu, sd):
    alpha = 1.0
    #peaks will be stored in this array
    indices = []
    t = 0.1
    f = StepStrideFrequencyModuleConstants.F

    #find the start-index
    start_index = next(x[0] for x in enumerate(f) if x[1] >= 0.5)
    #find the end-index
    end_index = next(x[0] for x in enumerate(f) if x[1] >= 2.5)
    while len(indices) < 2:
        indices = get_peaks_between(p_spectrum, start_index=start_index, end_index=end_index,thres=t,min_dist=1)
        t = t / 10

    #print f[indices]
    score = float("inf")
    (a, b) = (None, None)
    for i in range(len(indices)):
        f1 = f[indices[i]]
        a1 = p_spectrum[indices[i]]
        left_idx = indices[i-1] if i-1>=0 else 0
        right_idx = indices[i+1] if i+1<len(indices) else len(f)-1
        p1 = np.array([f1, math.sqrt(a1), get_half_energy(p_spectrum, f, indices[i],left_idx,right_idx)])
        p1 = (p1 - mu)/sd
        p1_prime = global_peaks[group_type]["p1"]
        p1_prime = (p1_prime - mu)/sd
        for j in range(i + 1, len(indices)):
            f2 = f[indices[j]]
            a2 = p_spectrum[indices[j]]
            left_idx = indices[j-1] if j-1>=0 else 0
            right_idx = indices[j+1] if j+1<len(indices) else len(f)-1
            p2 = np.array([f2, math.sqrt(a2), get_half_energy(p_spectrum, f, indices[j],left_idx,right_idx)])
            p2 = (p2 - mu)/sd
            p2_prime = global_peaks[group_type]["p2"]
            p2_prime = (p2_prime - mu)/sd

            # minimize this function to find p1 and p2
            new_score = np.linalg.norm(p1 - p1_prime) + np.linalg.norm(p2 - p2_prime) + alpha * np.linalg.norm(2 * p1[0] - p2[0])  # + p1[0]**2 + p2[0]**2
            if new_score < score:
                score = new_score
                (a, b) = (i, j)
    return {"group_type":group_type, "all_peaks": f[indices], "selected_peaks":f[[indices[a], indices[b]]]}

'''This method is used to load the data with given filename'''
def get_data(filename):
    return pd.read_json(Constants.walking_outbound_path + "/" + filename)

if __name__ == '__main__':
    load_data()
