import os
import argparse
import pandas as pd
import sys

training_data_location = "/ifs/home/c2b2/ip_lab/shares/DATA/fwd_bwd_data/converted_fb_accel_data/"
additional_training_data_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/supp_fwd_bwd_data/"
test_data_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/test_fwd_bwd_data/"

meta_data_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/meta-data.csv"
meta_testdata_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/meta-data-testing.csv"
meta_additional_data_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/meta-data-additional.csv"

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset",help="dataset_type")
    arg_parser.add_argument("start_index", help="start_index")
    arg_parser.add_argument("end_index", help="end_index")
    args = arg_parser.parse_args()

    data_location = None
    meta_file = None
    if args.dataset == "training":
        data_location = training_data_location
        meta_file = meta_data_location
    elif args.dataset == "testing":
        data_location = test_data_location
        meta_file = meta_testdata_location
    else:
        data_location = additional_training_data_location
        meta_file = meta_additional_data_location

    files = os.listdir(data_location)
    outbound_files = []
    return_files = []
    #print "here"
    for file in files:
        if "outbound.csv" in os.listdir(data_location + file):
            outbound_files.append(file)
        if "return.csv" in os.listdir(data_location + file):
            return_files.append(file)

    print "hello"
    outbound_count = 0
    return_count = 0
    both_count = 0
    meta_data = pd.read_csv(meta_file)
    meta_data = meta_data[int(args.start_index):int(args.end_index)]
    for idx,row in meta_data.iterrows():
        if row["recordId"] in outbound_files:
            outbound_count += 1
        if row["recordId"] in return_files:
            return_count += 1
        if row["recordId"] in outbound_files and row["recordId"] in return_files:
            both_count += 1

    print "Outbound=" + str(outbound_count)
    print "Return=" + str(return_count)
    print "Both=" + str(both_count)

