import pandas as pd
import numpy as np
import os
import sys
from scipy.interpolate import InterpolatedUnivariateSpline
import argparse

N = 1600

def interpolate_data(f,a,length):
    x = np.linspace(0,length,N)
    ius = InterpolatedUnivariateSpline(f,a,k=1)
    return (x,ius(x))

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("start_index", help="start-index")
    arg_parser.add_argument("end_index", help="end-index")
    args = arg_parser.parse_args()

    if os.path.isfile("recordIdList.csv"):
        recordIds = pd.read_csv("recordIdList.csv")
    else:
        print "Record list file doesn't exist."
        sys.exit()

    base_path = "/ifs/home/c2b2/ip_lab/shares/DATA/fwd_bwd_data/converted_fb_accel_data"
    start_index = int(args.start_index)
    end_index = int(args.end_index)
    for i in range(start_index,end_index):
        recordId = recordIds.loc[i]["columns"]
        recordId_path = base_path + "/" + recordId
        if os.path.isdir(recordId_path):
            #check if outbound file exists
            outbound_path = recordId_path + "/" + "outbound.csv"
            df_new = pd.DataFrame(columns=["x","y","z"])
            cx = []
            cy = []
            cz = []
            if os.path.isfile(outbound_path):
                df = pd.read_csv(outbound_path)
                x = df["x"]
                y = df["y"]
                z = df["z"]
                start_time = df.iloc[0]["timestamp"]
                end_time = df.iloc[len(df)-1]["timestamp"]
                t = np.linspace(0,end_time-start_time,len(df))
                (tx,x_new) = interpolate_data(t,x,len(df))
                (ty,y_new) = interpolate_data(t,y,len(df))
                (tz,z_new) = interpolate_data(t,z,len(df))
                cx.extend(x_new)
                cy.extend(y_new)
                cz.extend(z_new)
            #check if return file exists
            return_path = recordId_path + "/" + "return.csv"
            if os.path.isfile(return_path):
                df = pd.read_csv(return_path)
                x = df["x"]
                y = df["y"]
                z = df["z"]
                start_time = df.iloc[0]["timestamp"]
                end_time = df.iloc[len(df) - 1]["timestamp"]
                t = np.linspace(0,end_time-start_time,len(df))
                (tx, x_new) = interpolate_data(t, x, len(df))
                (ty, y_new) = interpolate_data(t, y, len(df))
                (tz, z_new) = interpolate_data(t, z, len(df))
                cx.extend(x_new)
                cy.extend(y_new)
                cz.extend(z_new)
            if len(cx) > 0: #files were present
                sx = pd.Series(cx)
                sy = pd.Series(cy)
                sz = pd.Series(cz)
                df_new["x"] = sx
                df_new["y"] = sy
                df_new["z"] = sz
            if len(cx) > 0:
                df_new.to_csv("DATA/" + recordId + "-start_index-" + str(start_index) + "-end_index-" + str(end_index) + ".csv")

