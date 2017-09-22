# simplified_walk.py

# Walk feature extraction for training data: demographics:syn10146552 , table: syn10146553
import pandas as pd
import json
import numpy as np
import synapseclient
import sys
import shutil
import sys

syn = synapseclient.login() #use this if you've already set up your config file

prev_cache_loc = syn.cache.cache_root_dir
syn.cache.cache_root_dir = "./cache"
download_location = "./dataset/accel_motion_walking_outbound/"
demographics_location = "./cache/765/16714765/Job-42404212324897933952755971.csv"
meta_file_location = "./cache/771/16714771/Job-42404236483516541103356733.csv"
# read in the healthCodes of interest from demographics training table
print "Querying the Demographics table..."
demo_syntable = syn.tableQuery("SELECT * FROM syn10146552")
demo = demo_syntable.asDataFrame()
healthCodeList = ", ".join( repr(i) for i in demo["healthCode"]) 


print "Querying the Walking training table..."
#Query 'walking training table' for walk data recordIDs and healthCodes. 
INPUT_WALKING_ACTIVITY_TABLE_SYNID = "syn10146553"
actv_walking_syntable = syn.tableQuery(('SELECT * FROM {0} WHERE healthCode IN ({1}) AND "accel_walking_outbound.json.items" is not null').format(INPUT_WALKING_ACTIVITY_TABLE_SYNID, healthCodeList))
actv_walking = actv_walking_syntable.asDataFrame()
actv_walking['idx'] = actv_walking.index

######################
# Download JSON Files
######################
# bulk download outbound walk JSON files containing sensor data
walk_json_files = syn.downloadTableColumns(actv_walking_syntable, "accel_walking_outbound.json.items")
items = walk_json_files.items()

# create pandas dataframe of JSON filehandleIDs and filepaths
walk_json_files_temp = pd.DataFrame({"accel_walking_outbound.json.items": [i[0] for i in items], "outbound_walk_json_file": [i[1] for i in items]})

actv_walking["accel_walking_outbound.json.items"] = actv_walking["accel_walking_outbound.json.items"].astype(str)

actv_walk_temp = pd.merge(actv_walking, walk_json_files_temp, on="accel_walking_outbound.json.items")

for index,row in actv_walk_temp.iterrows():
	tokens = row["outbound_walk_json_file"].split("/")
	actv_walk_temp.loc[index,"outbound_walk_json_file"] = tokens[len(tokens)-1]
actv_walk_temp.to_csv('meta-data.csv')

print "Moving the files..."
for file_handle_id,path in items:
	shutil.move(path, download_location)

syn.cache.cache_root_dir = prev_cache_loc