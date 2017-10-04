import os
import pickle
import pandas as pd
import sys
from collections import Counter
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse

'''This method is used to fetch features from the data'''
def fetch_features(mdl,datum,**kwargs):
    viterbi_sequence = mdl.predict(datum, 0)[1]
    #collapse the sequence
    collapsed_sequence = []
    prev_state = viterbi_sequence[0]
    count = 1
    for idx, state in enumerate(viterbi_sequence[1:]):
        if state != prev_state:
            collapsed_sequence.append(count)
            prev_state = state
            count = 1
        else:
            count = count + 1
    collapsed_sequence.append(count)
    # get the time spent in each collapsed sequence
    total_time = kwargs["total_time"]
    assert len(viterbi_sequence) > 0
    time_in_collapsed_sequence = np.array([total_time * seq / len(viterbi_sequence) for seq in collapsed_sequence])
    fraction_time_in_states = np.zeros(shape=(25,), dtype=float)
    counts = Counter(viterbi_sequence)
    for key in counts.keys():
        fraction_time_in_states[key] = total_time * counts[key] / len(viterbi_sequence)
    return len(time_in_collapsed_sequence), time_in_collapsed_sequence, fraction_time_in_states

'''This method is used to bring all the rows to the same size for a given (activity,category) 
i.e ({outbound,return,both},{0,1,2,3})'''
def normalize_features(feature_set):
    max_length = max([t[0] for t in feature_set.values()])
    new_feature_set = []
    feature_values = []
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

#constants to be used throughout the program
model_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/models/"
training_data_location = "/ifs/home/c2b2/ip_lab/shares/DATA/fwd_bwd_data/converted_fb_accel_data/"
additional_training_data_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/supp_fwd_bwd_data/"
test_data_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/test_fwd_bwd_data/"
meta_data_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/meta-data.csv"
meta_testdata_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/meta-data-testing.csv"
meta_additional_data_location = "/ifs/home/c2b2/ip_lab/shares/DATA/dataset/meta-data-additional.csv"

#different types of models
both_mdls = {}
return_mdls = {}
outbound_mdls = {}

#different models based on the phones
#this data can be found in category.csv, but hardcoding it now, since it is a one-time thing
iphone_6 = ["category0.pyhsmm","category1.pyhsmm","category2.pyhsmm","category3.pyhsmm"]
iphone_6_plus = ["category4.pyhsmm","category5.pyhsmm","category6.pyhsmm","category7.pyhsmm"]
iphone_5s_gsm = ["category8.pyhsmm","category9.pyhsmm","category10.pyhsmm","category11.pyhsmm"]
others = ["category12.pyhsmm","category13.pyhsmm","category14.pyhsmm","category15.pyhsmm"]

def run_on_data(type,activity,category_index,start_index,end_index):

    data_location = None
    if type == "training":
        meta_data = pd.read_csv(meta_data_location)
        data_location = training_data_location
    elif type == "testing":
        meta_data = pd.read_csv(meta_testdata_location)
        data_location = test_data_location
    elif type == "additional":
        meta_data = pd.read_csv(meta_additional_data_location)
        data_location = additional_training_data_location
    else:
        print "Invalid dataset type {}".format(type)
        sys.exit()

    if activity == "both":
        print "Loading the outbound+return models..."
        for file in os.listdir(model_location + "both/"):
            full_path = model_location + "both/" + file
            with open(full_path,"r") as f:
                both_mdls[file] = pickle.load(f)
    elif activity == "outbound":
        print "Loading the outbound models..."
        for file in os.listdir(model_location + "outbound/"):
            full_path = model_location + "outbound/" + file
            with open(full_path, "r") as f:
                outbound_mdls[file] = pickle.load(f)
    elif activity == "return":
        print "Loading the return models..."
        for file in os.listdir(model_location + "return/"):
            full_path = model_location + "return/" + file
            with open(full_path, "r") as f:
                return_mdls[file] = pickle.load(f)
    else:
        print "Invalid activity {}".format(activity)
        sys.exit()

    print "Running on the data..."
    start_index = int(start_index)
    end_index = int(end_index)
    assert start_index < end_index
    meta_data = meta_data[start_index:end_index]
    feature_set = defaultdict(tuple)
    missing_records = []
    if activity == "outbound":
        for index,row in meta_data.iterrows():
            record_id = row["recordId"]
            phone_info = row["phoneInfo"]
            #load the data corresponding to this record_id
            record_id_location = data_location + record_id
            if os.path.isdir(record_id_location):
                outbound_path = record_id_location + "/" + "outbound.csv"
                if os.path.isfile(outbound_path):
                    outbound_df = pd.read_csv(outbound_path)
                    start_time = outbound_df.loc[0, "timestamp"]
                    end_time = outbound_df.loc[len(outbound_df) - 1, "timestamp"]
                    outbound_array = outbound_df[["x","y","z"]].as_matrix()
                    if phone_info == "iPhone 6":
                        model = iphone_6[category_index]
                    elif phone_info == "iPhone 6 Plus":
                        model = iphone_6_plus[category_index]
                    elif phone_info == "iPhone 5s (GSM)":
                        model = iphone_5s_gsm[category_index]
                    else:
                        model = others[category_index]
                    #for idx,model_name in enumerate(model):
                    if outbound_mdls.has_key(model):
                        mdl = outbound_mdls[model]
                        t = fetch_features(mdl, outbound_array, total_time=end_time - start_time)
                        feature_set[record_id] = t
                    else:
                        print "{} does not exist for {}".format(model,activity)

            else:
                missing_records.append(record_id)
                print "Outbound data for {} is missing".format(record_id)
    elif activity == "return":
        for index, row in meta_data.iterrows():
            record_id = row["recordId"]
            phone_info = row["phoneInfo"]
            # load the data corresponding to this record_id
            record_id_location = data_location + record_id
            if os.path.isdir(record_id_location):
                return_path = record_id_location + "/" + "return.csv"
                if os.path.isfile(return_path):
                    outbound_df = pd.read_csv(return_path)
                    start_time = outbound_df.loc[0, "timestamp"]
                    end_time = outbound_df.loc[len(outbound_df) - 1, "timestamp"]
                    return_array = outbound_df[["x", "y", "z"]].as_matrix()
                    if phone_info == "iPhone 6":
                        model = iphone_6[category_index]
                    elif phone_info == "iPhone 6 Plus":
                        model = iphone_6_plus[category_index]
                    elif phone_info == "iPhone 5s (GSM)":
                        model = iphone_5s_gsm[category_index]
                    else:
                        model = others[category_index]

                    if return_mdls.has_key(model):
                        mdl = return_mdls[model]
                        t = fetch_features(mdl, return_array, total_time=end_time - start_time)
                        feature_set[record_id] = t
                    else:
                        print "{} does not exist for {}".format(model, activity)
            else:
                missing_records.append(record_id)
                print "Return data for {} is missing".format(record_id)
    elif activity == "both":
        for index, row in meta_data.iterrows():
            record_id = row["recordId"]
            phone_info = row["phoneInfo"]
            # load the data corresponding to this record_id
            record_id_location = data_location + record_id
            if os.path.isdir(record_id_location):
                outbound_path = record_id_location + "/" + "outbound.csv"
                return_path = record_id_location + "/" + "return.csv"
                if os.path.isfile(outbound_path) and os.path.isfile(return_path):
                    outbound_df = pd.read_csv(outbound_path)
                    return_df = pd.read_csv(return_path)
                    outbound_time = outbound_df.loc[len(outbound_df) - 1, "timestamp"] - outbound_df.loc[0, "timestamp"]
                    return_time = return_df.loc[len(return_df) - 1, "timestamp"] - return_df.loc[0, "timestamp"]
                    total_time = outbound_time + return_time
                    both_df = outbound_df[["x", "y", "z"]]
                    both_array = both_df.append(return_df[["x", "y", "z"]]).as_matrix()
                    if phone_info == "iPhone 6":
                        model = iphone_6[category_index]
                    elif phone_info == "iPhone 6 Plus":
                        model = iphone_6_plus[category_index]
                    elif phone_info == "iPhone 5s (GSM)":
                        model = iphone_5s_gsm[category_index]
                    else:
                        model = others[category_index]
                    if both_mdls.has_key(model):
                        mdl = both_mdls[model]
                        t = fetch_features(mdl, both_array, total_time=total_time)
                        feature_set[record_id] = t
                    else:
                        print "{} does not exist for {}".format(model, activity)
            else:
                missing_records.append(record_id)
                print "Outbound+Return data for {} is missing".format(record_id)
    else:
        print "Invalid value {} for activity".format(activity)
        sys.exit()

    with open("{}/feature_set_{}_{}_category{}_start-index{}_end-index{}.pkl".format(type+"-new",type,activity,category_index,start_index,end_index),"w") as f:
        pickle.dump(feature_set,f)

    with open("{}/missing_records_{}_{}_category{}_start-index{}_end-index{}.pkl".format(type+"-new",type,activity,category_index,start_index,end_index),"w") as f:
        pickle.dump(missing_records,f)

    '''feature_set, mean_value = normalize_features(feature_set)
    for record in missing_records:
        feature_set.append([record, mean_value.tolist()])
    feature_set = pd.DataFrame(data=feature_set)
    feature_set.to_csv("feature_set_{}_{}.csv".format(activity,category_index), index=False)'''


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("type",help="type of dataset")
    arg_parser.add_argument("activity", help="activity")
    arg_parser.add_argument("category_index", help="index")
    arg_parser.add_argument("start_index",help="start_index")
    arg_parser.add_argument("end_index",help="end_index")
    args = arg_parser.parse_args()
    run_on_data(args.type,args.activity,int(args.category_index),args.start_index,args.end_index)

