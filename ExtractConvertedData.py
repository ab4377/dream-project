import pandas as pd
import numpy as np
import Constants
import os
import sys
import argparse

'''Call this method to prepare data per key'''
def data_preparation(recordIds,start_index,end_index,filename):
    assert start_index < end_index
    if end_index > len(recordIds):
        end_index = len(recordIds) - 1
    recordIds = recordIds.iloc[start_index:end_index]
    accel_data = pd.DataFrame(columns=["x", "y", "z"])
    for idx,recordId in recordIds.iterrows():
        file_location = Constants.converted_data_location + recordId["recordId"] + "/"
        #print file_location
        if os.path.isfile(file_location + "outbound.csv"):
            print "Fetching outbound data for recordId " + recordId["recordId"]
            outbound_df = pd.read_csv(file_location + "outbound.csv")
            for idx, row in outbound_df.iterrows():
                data = [{"x":row["x"], "y":row["y"], "z":row["z"]}]
                accel_data = accel_data.append(data)
        if os.path.isfile(file_location + "return.csv"):
            print "Fetching return data for recordId " + recordId["recordId"]
            return_df = pd.read_csv(file_location + "return.csv")
            for idx, row in return_df.iterrows():
                data = [{"x":row["x"],"y":row["y"], "z":row["z"]}]
                accel_data = accel_data.append(data)
    if not accel_data.empty:
        accel_data.to_csv(filename,index=False)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("phone_type", help="phone-type")
    arg_parser.add_argument("label", help="label")
    arg_parser.add_argument("start_index", help="start-index")
    arg_parser.add_argument("end_index", help="end-index")
    args = arg_parser.parse_args()
    search_key = (args.phone_type, args.label)
    #print search_key
    df = pd.read_csv(Constants.meta_data_location)
    grouped = df.groupby(by=["phoneInfo","medTimepoint"])
    groups = {}
    for key,group in grouped:
        grouped_df = grouped.get_group(key)
        ll = []
        for idx,row in grouped_df.iterrows():
            ll.append(row["recordId"])
        groups[key] = ll


    phones = df["phoneInfo"].unique()
    phone_index = {}

    for idx,phone in enumerate(phones):
        if type(phone) is not float: #weird way of checking for nan
            phone_index[idx] = phone

    inv_phone_index = {v: k for k,v in phone_index.iteritems()}
    labels = df["medTimepoint"].unique()
    label_index = {}
    for idx,label in enumerate(labels):
        if type(label) is not float: #weird way of checking for nan
            label_index[idx] = label

    inv_labels_index = {v: k for k,v in label_index.iteritems()}

    filename = Constants.data_location + "phoneindex-" + str(inv_phone_index[args.phone_type]) + "_labelindex-" + str(inv_labels_index[args.label]) + ".csv"
    #print search_key
    #print grouped.get_group(search_key)
    print "FETCHING DATA for " + str(search_key) + "..."
    if groups.has_key(search_key):
        print "filename: " + filename
        data_preparation(grouped.get_group(search_key),start_index=int(args.start_index),end_index=int(args.end_index),filename=filename)
        print "Written for " + str(search_key)

