import pandas as pd
import numpy as np
import dream
import argparse
import os
import fasteners
import sys

def main():
    file_types = ['outbound', 'rest', 'return']
    parser = argparse.ArgumentParser(description='Find forward-backward direction.')
    parser.add_argument('-l', dest='fb_vect_list', required=True)
    parser.add_argument('-s', dest='start', required=True)
    parser.add_argument('-e', dest='end', required=True)
    parser.add_argument('-data', dest='data_dir', required=True)
    parser.add_argument('-dest', dest='dest_dir', required=True)
    args = parser.parse_args()
    fb_vect_list = pd.read_csv(args.fb_vect_list, skiprows=range(1, int(args.start) + 1),
                      nrows=int(args.end) - int(args.start) + 1)

    for i in fb_vect_list.size:
        path = args.data_dir + '/' + fb_vect_list.loc[i, 'id'] \
               + '/deviceMotion_walking_' + fb_vect_list.loc[i, 'type'] + '.json.items.csv'

        df = pd.read_csv(path, index_col=0)
        fb_vect = [float(x) for x in fb_vect_list.loc[i, 'vector'][-1:1].split(',')]
        converted_df = dream.convert_to_forward_backward_coordinates(df, fb_vect)
        converted_dir = args.dest_dir + '/' + fb_vect_list[i, 'id']

        try:
            if not os.path.exists(converted_dir):
                os.makedirs(converted_dir)
            converted_df.to_csv(converted_dir + '/' + fb_vect_list.loc[i, 'type'] + '.csv')
        except IOError as e:
            print >> sys.stderr, e


if __name__ == '__main__':
    main()