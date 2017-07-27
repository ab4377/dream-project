import pandas as pd
import os
import numpy as np
import math
from scipy.fftpack import fft
import mlpy
import json
import matplotlib.pyplot as plt

walking_outbound_path = "../dataset/accel_motion_walking_outbound"
meta_data_location = "../meta-data.csv"
sampling_frequency = 100

def load_data():
    meta_data = pd.read_csv(meta_data_location)
    class_label_map = {}
    for index,row in meta_data.iterrows():
        class_label_map[row["outbound_walk_json_file"]] = row["medTimepoint"]
    X = []
    class_label = []
    file_names = os.listdir(walking_outbound_path)
    for file_name in file_names:
        if file_name.endswith(".tmp"):
            x = pd.read_json(walking_outbound_path + "/" + file_name)
            X.append(x)
            class_label.append(class_label_map[file_name])
    return (X,class_label)

def get_accel_magnitude(x,y,z):
    return math.sqrt(x**2 + y**2 + z**2)

def get_frequency_components(x):
    N = len(x)
    y = fft(x)
    a = np.abs(y[0:N//2])
    f = np.linspace(0,sampling_frequency/2,N//2)
    return (f,a)

def dtw_distance(x,y):
    return mlpy.dtw_std(x,y,dist_only=True)

def to_json(distance_matrix,class_labels):
    minimum = distance_matrix.min(axis=0).min()
    maximum = distance_matrix.max(axis=1).max()

    #new range
    low = 1.0
    high = 10.0
    #group labels
    group_code = dict(zip(list(set(class_labels)),[1,2,3,4]))
    data = {}
    data["nodes"] = []
    data["links"] = []
    for label in class_labels:
        datum = {}
        datum["id"] = label
        datum["group"] = group_code[label]
        data["nodes"].append(datum)

    N = distance_matrix.shape[0]
    for i in xrange(N):
        for j in xrange(i+1,N):
            datum = {}
            datum["source"] = class_labels[i]
            datum["target"] = class_labels[j]
            datum["value"] = int(low + (distance_matrix[i][j] - minimum)*(high - low)/(maximum - minimum))
            data["links"].append(datum)
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)

def get_average_distance(distance_matrix,class_labels):
    group_code = dict(zip(list(set(class_labels)), [1, 2, 3, 4]))
    print group_code
    distance = np.zeros(shape=(len(group_code),len(group_code)),dtype=float)
    total_examples = np.zeros(shape=(len(group_code),len(group_code)))
    (M,N) = distance_matrix.shape
    for i in xrange(N):
        for j in xrange(i+1,N):
            distance[group_code[class_labels[i]]-1][group_code[class_labels[j]]-1] += distance_matrix[i][j]
            total_examples[group_code[class_labels[i]]-1][group_code[class_labels[j]]-1] += 1

    distance = distance/total_examples
    heatmap = plt.pcolor(distance)
    plt.colorbar(heatmap)
    plt.show()
    '''columns = ["row","column","value"]
    df = pd.DataFrame(columns=columns)
    data = []
    for i in xrange(distance.shape[0]):
        for j in xrange(distance.shape[1]):
            datum = {}
            datum["row"] = int(i+1)
            datum["column"] = int(j+1)
            datum["value"] = distance[i][j]
            data.append(datum)
    df = df.append(data)
    print df
    df.to_csv('data.tsv')
    with open('data.tsv','w') as f:
        f.write("row\tcolumn\tvalue\n")
        for key in distance.keys():
            f.write(group_code[key[0]] + "\t" + group_code[key[1]] + "\t" + str(distance[key]) + "\n")'''

if __name__ == '__main__':
    (X,class_label) = load_data()
    Z = []
    for x in X:
        accel = []
        for index,row in x.iterrows():
            accel.append(get_accel_magnitude(row["x"],row["y"],row["z"]))
        accel = np.array(accel)
        Z.append(accel)


    print 'Computing the DTW distances...'
    N = len(Z)
    distance_matrix = np.ndarray(shape=(N,N))
    for i in xrange(N):
        for j in xrange(i+1,N):
            distance_matrix[i][j] = dtw_distance(Z[i],Z[j])

    #distance_matrix = distance_matrix + distance_matrix.T
    print "Done!"
    print "Creating TSV data..."
    #to_json(distance_matrix,class_label)
    get_average_distance(distance_matrix,class_label)
