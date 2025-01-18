import pandas as pd

df = pd.read_csv("QA_1217FULLTEXT3_all.csv")
print(df.head(5))

#df = df.drop(columns=["filename"])

current_context = df["summary"][0]
id = 1
df["id"] = 0

for index, row in df.iterrows():
    context = row["summary"]
    if context != current_context:
        current_context = context
        id += 1
        df.at[index, "id"] = id
    else:
        df.at[index, "id"] = id

print(df.head(5))

df.to_csv("QA_1217FULLTEXT3_all_modified.csv", index=False)