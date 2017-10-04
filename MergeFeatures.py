import pandas as pd
import argparse

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("left_df", help="left_df")
    arg_parser.add_argument("right_df", help="right_df")
    arg_parser.add_argument("outfile",help="outfile name")
    #arg_parser.add_argument("left_suffix", help="left_suffix")
    #arg_parser.add_argument("right_suffix", help="right_suffix")
    args = arg_parser.parse_args()
    left_df = pd.read_csv(args.left_df)
    right_df = pd.read_csv(args.right_df)
    merged_df = pd.merge(left_df,right_df,on=["recordId"],sort=False)
    merged_df.to_csv(args.outfile,index=False)