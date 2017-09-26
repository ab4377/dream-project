import pandas as pd
import os

base_path = "/ifs/home/c2b2/ip_lab/shares/DATA/fwd_bwd_data/converted_fb_accel_data"
recordIds = os.listdir(base_path)
recordIds_df = pd.DataFrame(data=recordIds,columns=["columns"])
recordIds_df.to_csv("recordIdList.csv",index=True,header=True)