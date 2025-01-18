import pandas as pd
df = pd.read_csv("ragas_results.csv")
total_time_spent = sum(df["Total Time"])
print(total_time_spent)

#19.5
#9.5