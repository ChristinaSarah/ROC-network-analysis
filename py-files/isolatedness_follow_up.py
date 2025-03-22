import pandas as pd

# Load the data
df = pd.read_csv("/workspaces/ROC-network-analysis/input-files/roc_network_data_follow_up.csv")

# Create sets of all students who are listed at least once as emotional or academic support
emot_support_set = set(df["emot_1"].dropna()) \
                   | set(df["emot_2"].dropna()) \
                   | set(df["emot_3"].dropna())
academic_support_set = set(df["academic_1"].dropna()) \
                       | set(df["academic_2"].dropna()) \
                       | set(df["academic_3"].dropna())

# Convert them to strings to handle potential mixed data types
# (This avoids issues if some IDs are numeric while others are strings.)
emot_support_set = set(map(str, emot_support_set))
academic_support_set = set(map(str, academic_support_set))

# Convert fs_student_id to string to match sets above
df["fs_student_id_str"] = df["fs_student_id"].astype(str)

# 1) isolated_emot_in: 1 if not listed by anyone else as emotional support, 0 otherwise
df["isolated_emot_in"] = df["fs_student_id_str"].apply(lambda x: 0 if x in emot_support_set else 1)

# 2) isolated_academic_in: 1 if not listed by anyone else as academic support, 0 otherwise
df["isolated_academic_in"] = df["fs_student_id_str"].apply(lambda x: 0 if x in academic_support_set else 1)

# 3) isolated_emot_out: 1 if student does not list anyone as emot_1/emot_2/emot_3, 0 otherwise
df["isolated_emot_out"] = df[["emot_1", "emot_2", "emot_3"]]\
    .isnull()\
    .all(axis=1)\
    .astype(int)

# 4) isolated_academic_out: 1 if student does not list anyone as academic_1/academic_2/academic_3, 0 otherwise
df["isolated_academic_out"] = df[["academic_1", "academic_2", "academic_3"]]\
    .isnull()\
    .all(axis=1)\
    .astype(int)

# Clean up: drop the helper column
df.drop(columns=["fs_student_id_str"], inplace=True)

# Export everything (including new indicators) to a CSV
df.to_csv("/workspaces/ROC-network-analysis/output-files/roc_isolatedness_followup.csv", index=False)