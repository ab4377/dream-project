import Utilities as utilities
import numpy as np
import pandas as pd
import Constants

def project(direction, v):
    print np.dot(direction,v)/[(np.linalg.norm(direction))**2]

def load_forward_backward_data():
    df = pd.read_csv(Constants.forward_backward_path,delimiter="\t")
    direction_per_record_id = {}
    for idx,row in df.iterrows():
        vectors = row["vector"].replace("[","").replace("]","").split(",")
        direction_per_record_id[row["id"]] = np.array([float(vectors[0]),float(vectors[1])])
    return direction_per_record_id

direction_per_record_id = load_forward_backward_data()
(X,recordIds) = utilities.load_full_data()
for idx,x in enumerate(X):
    if direction_per_record_id.has_key(recordIds[idx]):
        direction = direction_per_record_id[recordIds[idx]]
        project(direction,np.array(x["x"],x["y"]))
    else:
        print "Key not found!"
    break
