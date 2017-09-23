import pandas as pd
import numpy as np
import dream
import argparse
import os
import fasteners
import sys
import cPickle

def main():
	file_types = ['outbound', 'rest', 'return']
	parser = argparse.ArgumentParser(description='Find forward-backward direction.')
	parser.add_argument('-l', dest='id_list', required=True)
	parser.add_argument('-s', dest='start', required=True)
	parser.add_argument('-e', dest='end', required=True)
	parser.add_argument('-data', dest='data_dir', required=True)
	parser.add_argument('-dest', dest='dest', required=True)
	args = parser.parse_args()
	ids = pd.read_csv(args.id_list, skiprows=range(1, int(args.start) + 1),
			  nrows=int(args.end) - int(args.start) + 1)['recordId']

	for id in ids:
		for type in file_types:
			path = args.data_dir + '/' + id
			if not os.path.exists(path):
				os.makedirs(path)

			file = open(path + '/' + type + '.pyhsmm', 'wb')
			df = pd.read_csv(path, index_col=0)
			data = df[['x', 'y', 'z']]
			model = dream.fit_hmm(data)
			cPickle.dump(model, file)



if __name__ == '__main__':
	main()
