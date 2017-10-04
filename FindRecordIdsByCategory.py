import pandas as pd
import Constants

category_df = pd.read_csv("category.csv")
category = {}
for idx,row in category_df.iterrows():
    tokens = row["value"].split("-")
    category[(tokens[0],tokens[1])] = row["category"]

df = pd.read_csv(Constants.meta_data_location)
grouped = df.groupby(by=["phoneInfo","medTimepoint"])
groups = {}
for key,group in grouped:
    grouped_df = grouped.get_group(key)
    ll = []
    for idx,row in grouped_df.iterrows():
        ll.append(row["recordId"])
    groups[key] = ll

for key in groups.keys():
    data = pd.DataFrame(groups[key],columns=["recordIds"])
    data.to_csv(category[key] + ".csv",index=False,header=False)