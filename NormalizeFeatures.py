import numpy as np
import argparse
import os
import pickle
from collections import defaultdict
import pandas as pd

'''This method is used to bring all the rows to the same size for a given (activity,category) 
i.e ({outbound,return,both},{0,1,2,3})'''
def normalize_features(feature_set):
    max_length = max([t[0] for t in feature_set.values()])
    new_feature_set = []
    feature_values = [] #used to get the mean value of features
    for idx,key in enumerate(feature_set.keys()):
        collapsed_seq = feature_set[key][1]
        if len(collapsed_seq) < max_length :
            curr_len = len(collapsed_seq)
            collapsed_seq = np.append(collapsed_seq, np.zeros(shape=(max_length-curr_len,), dtype=float), axis=0)
        assert len(collapsed_seq) == max_length
        fraction_time = feature_set[key][2]
        feature_values.append(np.append(collapsed_seq,fraction_time,axis=0))
        feature = [key]
        feature.extend(np.append(collapsed_seq,fraction_time,axis=0).tolist())
        new_feature_set.append(feature)
    #now calculate the mean of the feature values which is used for missing data
    feature_values = np.array(feature_values)
    return new_feature_set, np.mean(feature_values,axis=0)

def transform(x):
    if x == 0:
        return "recordId"
    return "Feature" + str(x)

if __name__ == '__main__':
    meta_data_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/meta-data.csv"
    meta_testdata_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/meta-data-testing.csv"
    meta_additional_data_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/meta-data-additional.csv"

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("activity", help="activity")
    arg_parser.add_argument("category_index", help="index")
    args = arg_parser.parse_args()

    feature_set = defaultdict(tuple)
    missing_records = []

    print "Loading the pickled files..."
    #look in training-folder
    for file in os.listdir("training-new/"):
        if "feature_set_training_{}_category{}".format(args.activity,args.category_index) in file:
            #print "Loading {} ...".format(file)
            with open("training-new/"+file, "r") as f:
                feature_set.update(pickle.load(f))
        elif "missing_records_training_{}_category{}".format(args.activity,args.category_index) in file:
            #print "Loading {} ...".format(file)
            with open("training-new/"+file,"r") as f:
                missing_records.extend(pickle.load(f))

    print "Loading the additional dataset..."
    #look in additional-folder
    for file in os.listdir("additional-new/"):
        if "feature_set_additional_{}_category{}".format(args.activity,args.category_index) in file:
            #print "Loading {} ...".format(file)
            with open("additional-new/"+file, "r") as f:
                feature_set.update(pickle.load(f))
        elif "missing_records_additional_{}_category{}".format(args.activity,args.category_index) in file:
            #print "Loading {} ...".format(file)
            with open("additional-new/"+file,"r") as f:
                missing_records.extend(pickle.load(f))

    print "Loading the testing dataset..."
    # look in testing-folder
    for file in os.listdir("testing-new/"):
        if "feature_set_testing_{}_category{}".format(args.activity, args.category_index) in file:
            #print "Loading {} ...".format(file)
            with open("testing-new/"+file, "r") as f:
                feature_set.update(pickle.load(f))
        elif "missing_records_testing_{}_category{}".format(args.activity, args.category_index) in file:
            #print "Loading {} ...".format(file)
            with open("testing-new/"+file, "r") as f:
                missing_records.extend(pickle.load(f))

    assert len(feature_set) > 0
    print "Normalizing the features..."
    (feature_set,mean_value) = normalize_features(feature_set)
    for record in missing_records:
        feature = [record]
        feature.extend(mean_value.tolist())
        feature_set.append(feature)
    feature_set = pd.DataFrame(data=feature_set)
    feature_set = feature_set.rename(columns=transform)

    #there may be few records missing. Check for them using the meta-data
    meta_data = pd.read_csv(meta_data_location)
    meta_data_test = pd.read_csv(meta_testdata_location)
    meta_data_additional = pd.read_csv(meta_additional_data_location)

    original_records = set(meta_data["recordId"].as_matrix()).union(set(meta_data_test["recordId"].as_matrix())).union(set(meta_data_additional["recordId"].as_matrix()))
    current_records = set(feature_set["recordId"].as_matrix())
    missing_records = original_records.difference(current_records)
    missing_records_matrix = []
    #use the mean-value for the missing records
    for record in missing_records:
        feature = [record]
        feature.extend(mean_value.tolist())
        missing_records_matrix.append(feature)
    missing_records_matrix = pd.DataFrame(data=missing_records_matrix)
    missing_records_matrix = missing_records_matrix.rename(columns=transform)
    feature_set = feature_set.append(missing_records_matrix,ignore_index=True)
    feature_set.to_csv("feature_set_{}_{}.csv".format(args.activity, args.category_index), index=False)
    print len(feature_set)