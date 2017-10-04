import pandas as pd
import os
import synapseclient

def main():
	syn = synapseclient.login(email='ab4377@columbia.edu', password='Shivbaba1234@', rememberMe=True)
	print "Querying the Walking training table..."
	#Query 'walking training table' for walk data recordIDs and healthCodes. 
	INPUT_WALKING_ACTIVITY_TABLE_SYNID = "syn10733842"
	actv_walking_syntable = syn.tableQuery(('SELECT * FROM {0}').format(INPUT_WALKING_ACTIVITY_TABLE_SYNID))
	actv_walking = actv_walking_syntable.asDataFrame()
	actv_walking[["recordId","phoneInfo"]].to_csv('meta-data-testing.csv',index=False,header=True)
	print "Done!"
	
if __name__ == '__main__':
	main()