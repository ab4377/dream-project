import os
import argparse

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset",help="dataset_type")
    arg_parser.add_argument("activity", help="activity")
    arg_parser.add_argument("category_index", help="index")
    args = arg_parser.parse_args()
    feature_count = 0
    missing_records_count = 0
    for file in os.listdir("{}-new/".format(args.dataset)):
        if "feature_set_{}_{}_category{}".format(args.dataset,args.activity,args.category_index) in file:
            feature_count += 1
        if "missing_records_{}_{}_category{}".format(args.dataset,args.activity,args.category_index) in file:
            missing_records_count += 1

    print feature_count
    print missing_records_count
