import synapseclient
import os
import argparse
import dream
import pandas as pd
from multiprocessing import Pool

columns = ['deviceMotion_walking_outbound.json.items',
            'deviceMotion_walking_return.json.items',
            'deviceMotion_walking_rest.json.items']

def get_data(row, record_id, folder, syn):
    try:
        download_dir = folder + '/' + record_id
    except Exception as e:
        print e
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for column in columns:
        if os.path.exists(download_dir + '/' + column + '.csv'):
            continue
        try:
            print 'Downloading ' + column + ' for ' + record_id + '...'
            file_path = syn.downloadTableFile(table='syn10146553', column=column, downloadLocation=download_dir, rowId=row, versionNumber=1)
            dream.felative_to_absolute_coordinates_file(file_path, download_dir + '/' + column + '.csv')
            os.remove(file_path)
        except Exception as e:
            print e

def main():
    parser = argparse.ArgumentParser(description='Download and process data.')
    parser.add_argument('-d', dest='data_folder', required=True)
    parser.add_argument('-l', dest='id_list', required=True)
    parser.add_argument('record_ids', nargs='+')
    args = parser.parse_args()
    syn = synapseclient.login('ttv2107@columbia.edu', 'Txrxuxnxg123!')
    pool = Pool(8)

    print 'Saving to ' + args.data_folder
    try:
        ids = pd.read_csv(args.id_list, index_col=0)
    except Exception as e:
        print 'There was an exception in reading list of ids...'
        print ellipsis
        raise

    # read in the healthCodes of interest from demographics training table
    for row in args.record_ids:
        record_id = ids.iloc[int(row), 0]
        get_data(row, record_id, args.data_folder, syn)
        

if __name__ == '__main__':
    main()


