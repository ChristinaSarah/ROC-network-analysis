import pandas as pd

# Load the data
df = pd.read_csv("/workspaces/ROC-network-analysis/input-files/roc_network_data.csv")

# Create sets of all students who are listed at least once as emotional or academic support
friend_support_set = set(pd.to_numeric(df["friend_1"].dropna(), downcast="integer")) \
                   | set(pd.to_numeric(df["friend_2"].dropna(), downcast="integer")) \
                   | set(pd.to_numeric(df["friend_3"].dropna(), downcast="integer"))
support_support_set = set(pd.to_numeric(df["support_1"].dropna(), downcast="integer")) \
                       | set(pd.to_numeric(df["support_2"].dropna(), downcast="integer")) \
                       | set(pd.to_numeric(df["support_3"].dropna(), downcast="integer"))

# Convert them to strings to handle potential mixed data types
# (This avoids issues if some IDs are numeric while others are strings.)
friend_support_set = set(map(str, friend_support_set))
support_support_set = set(map(str, support_support_set))

# Convert student_id to string to match sets above
df["student_id_str"] = df["student_id"].astype(str)

# 1) isolated_friend_in: 1 if not listed by anyone else as friend, 0 otherwise
df["isolated_friend_in"] = df["student_id_str"].apply(lambda x: 0 if x in friend_support_set else 1)

# 2) isolated_support_in: 1 if not listed by anyone else as emotional support, 0 otherwise
df["isolated_support_in"] = df["student_id_str"].apply(lambda x: 0 if x in support_support_set else 1)

# 3) isolated_friend_out: 1 if student does not list anyone as friend, 0 otherwise
df["isolated_friend_out"] = df[["friend_1", "friend_2", "friend_3"]]\
    .isnull()\
    .all(axis=1)\
    .astype(int)

# 4) isolated_support_out: 1 if student does not list anyone as support, 0 otherwise
df["isolated_support_out"] = df[["support_1", "support_2", "support_3"]]\
    .isnull()\
    .all(axis=1)\
    .astype(int)

# Clean up: drop the helper column
df.drop(columns=["student_id_str"], inplace=True)

# Export everything (including new indicators) to a CSV
df.to_csv("/workspaces/ROC-network-analysis/output-files/roc_isolatedness_endline.csv", index=False)