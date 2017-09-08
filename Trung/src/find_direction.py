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
    parser.add_argument('-l', dest='id_list', required=True)
    parser.add_argument('-s', dest='start', required=True)
    parser.add_argument('-e', dest='end', required=True)
    parser.add_argument('-data', dest='data_dir', required=True)
    parser.add_argument('-dest', dest='dest', required=True)
    args = parser.parse_args()
    ids = pd.read_csv(args.id_list, skiprows=int(args.start), nrows=int(args.end) - int(args.start) + 1)['recordId']

    for id in ids:
        for type in file_types:
            path = args.data_dir + '/' + id + '/deviceMotion_walking_' + type + '.json.items.csv'
            try:
                df = pd.read_csv(path, index_col=0)
                vect = dream.find_forward_backward_direction(motion_df=df, target_delta_theta=np.pi / 100)

                lock = fasteners.InterProcessLock(args.dest)
                lock.acquire()
                with open(args.dest, 'a') as f:
                    f.write(id + '\t' + type + '\t' + str(vect[0][0].tolist()) + '\n')
                lock.release()
            except IOError as e:
                print >> sys.stderr, e



if __name__ == '__main__':
    main()