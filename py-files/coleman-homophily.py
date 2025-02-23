import pandas as pd
import numpy as np

# Input and output paths
INPUT_PATH = "/workspaces/ROC-network-analysis/input-files/roc_network_data_follow_up.csv"
OUTPUT_PATH = "/workspaces/ROC-network-analysis/output-files/coleman-homophily.csv"

# Load data
usecols = [
    "fs_classroom", "fs_student_id", "high_math",
    "academic_1", "academic_2", "academic_3",
    "emot_1", "emot_2", "emot_3"
]
df = pd.read_csv(INPUT_PATH, usecols=usecols)

# Convert 'yes'/'no' to binary (1 for high ability, 0 for low ability)
df["is_high"] = df["high_math"].map({"yes": 1, "no": 0})

# Create a dictionary to map student_id to ability level
df_unique = df[["fs_student_id", "is_high"]].drop_duplicates("fs_student_id")
map_ability = dict(zip(df_unique["fs_student_id"], df_unique["is_high"]))

def get_ability(student_id):
    """Return 0 (low), 1 (high), or NaN if unknown."""
    if pd.isna(student_id):
        return np.nan
    student_id = int(student_id)
    return map_ability.get(student_id, np.nan)

# Get the ability of each nominated friend
for i in [1, 2, 3]:
    df[f"acad_friend_ability_{i}"] = df[f"academic_{i}"].apply(get_ability)
    df[f"emot_friend_ability_{i}"] = df[f"emot_{i}"].apply(get_ability)

# Count number of low-low and high-high ties for each student
df["low_low_acad"] = sum((df["is_high"] == 0) & (df[f"acad_friend_ability_{i}"] == 0) for i in [1,2,3])
df["low_low_emot"] = sum((df["is_high"] == 0) & (df[f"emot_friend_ability_{i}"] == 0) for i in [1,2,3])
df["high_high_acad"] = sum((df["is_high"] == 1) & (df[f"acad_friend_ability_{i}"] == 1) for i in [1,2,3])
df["high_high_emot"] = sum((df["is_high"] == 1) & (df[f"emot_friend_ability_{i}"] == 1) for i in [1,2,3])

# Count total academic and emotional friendships
df["acad_friend_count"] = df[["academic_1", "academic_2", "academic_3"]].notna().sum(axis=1)
df["emot_friend_count"] = df[["emot_1", "emot_2", "emot_3"]].notna().sum(axis=1)

# Compute classroom-level aggregates
agg = df.groupby("fs_classroom").agg(
    total_student_number=("fs_student_id", "count"),
    low_math_student_number=("is_high", lambda x: (x == 0).sum()),
    high_math_student_number=("is_high", lambda x: (x == 1).sum()),
    low_low_acad_sum=("low_low_acad", "sum"),
    low_low_emot_sum=("low_low_emot", "sum"),
    high_high_acad_sum=("high_high_acad", "sum"),
    high_high_emot_sum=("high_high_emot", "sum"),
    acad_ties=("acad_friend_count", "sum"),
    emot_ties=("emot_friend_count", "sum")
).reset_index()

# Compute the shares
agg["low_math_share"] = agg["low_math_student_number"] / agg["total_student_number"]
agg["high_math_share"] = agg["high_math_student_number"] / agg["total_student_number"]

agg["low_low_share_acad"] = agg["low_low_acad_sum"] / agg["acad_ties"]
agg["low_low_share_emot"] = agg["low_low_emot_sum"] / agg["emot_ties"]
agg["high_high_share_acad"] = agg["high_high_acad_sum"] / agg["acad_ties"]
agg["high_high_share_emot"] = agg["high_high_emot_sum"] / agg["emot_ties"]

# Handle cases where the share is undefined (i.e., classrooms without both types of students)
agg.loc[agg["low_math_student_number"] == 0, ["low_low_share_acad", "low_low_share_emot"]] = np.nan
agg.loc[agg["high_math_student_number"] == 0, ["high_high_share_acad", "high_high_share_emot"]] = np.nan

# Compute homophily indices
agg["homophily_low_acad"] = (agg["low_low_share_acad"] - agg["low_math_share"]) / (1 - agg["low_math_share"])
agg["homophily_low_emot"] = (agg["low_low_share_emot"] - agg["low_math_share"]) / (1 - agg["low_math_share"])
agg["homophily_high_acad"] = (agg["high_high_share_acad"] - agg["high_math_share"]) / (1 - agg["high_math_share"])
agg["homophily_high_emot"] = (agg["high_high_share_emot"] - agg["high_math_share"]) / (1 - agg["high_math_share"])

# Export results
out_cols = [
    "fs_classroom", "total_student_number", "low_math_student_number", "high_math_student_number",
    "low_low_share_acad", "low_low_share_emot",
    "high_high_share_acad", "high_high_share_emot",
    "low_math_share", "high_math_share",
    "homophily_low_acad", "homophily_low_emot",
    "homophily_high_acad", "homophily_high_emot"
]
agg[out_cols].to_csv(OUTPUT_PATH, index=False)

print("Done. Output saved to:", OUTPUT_PATH)


