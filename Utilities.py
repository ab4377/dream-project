import Constants
import StepStrideFrequencyModuleConstants
import numpy as np
import pandas as pd
import os
from scipy.fftpack import fft
from collections import defaultdict
from scipy.interpolate import InterpolatedUnivariateSpline
import pickle

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
class label'''
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
    file_location = meta_data["outbound_walk_json_file"]
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
            x = pd.read_json(Constants.walking_outbound_path + "/" + file_location)
            X.append(x)
            class_label.append(key)
            '''else:
                print file_location'''
        #print len(medTimepointToJsonFiles[key][:size])
    #print len(class_label)
            #print len(medTimepointToJsonFiles[key][:size])
            #print len(medTimepointToJsonFiles)

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
    return (X,class_label)

'''This method is used to load the data with given filename'''
def get_data(filename):
    return pd.read_json(Constants.walking_outbound_path + "/" + filename)

if __name__ == '__main__':
    load_data()
