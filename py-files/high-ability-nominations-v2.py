import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv("/workspaces/ROC-network-analysis/input-files/roc_network_data_follow_up.csv")

# Map "yes"/"no" to boolean indicators (1=high, 0=low)
df["is_high"] = (df["high_math"] == "yes").astype(int)

# Prepare a lookup for ability by student
ability_dict = dict(zip(df["fs_student_id"], df["is_high"]))

# Function to build a list of edges (from_id -> to_id) for given columns
def build_edges(df, columns):
    edge_list = []
    for _, row in df.iterrows():
        from_id = row["fs_student_id"]
        from_ability = row["is_high"]
        for col in columns:
            to_id = row[col]
            if pd.notna(to_id):
                to_ability = ability_dict.get(to_id, np.nan)
                edge_list.append((from_id, from_ability, to_id, to_ability))
    return edge_list

# Build academic and emotional edges
acad_edges = build_edges(df, ["academic_1", "academic_2", "academic_3"])
emot_edges = build_edges(df, ["emot_1", "emot_2", "emot_3"])

# Convert edge lists to DataFrames
acad_df = pd.DataFrame(acad_edges, columns=["from_id","from_high","to_id","to_high"])
emot_df = pd.DataFrame(emot_edges, columns=["from_id","from_high","to_id","to_high"])

# Count nominations received by each student from low-ability and from all students
def summarize_nominations(edges_df):
    # Group by 'to_id' and compute how many come from low-ability (from_high=0) and total
    counts = edges_df.groupby("to_id").agg(
        from_low=("from_high", lambda x: sum(x==0)),
        total=("from_high", "count")
    ).reset_index()
    return counts

acad_counts = summarize_nominations(acad_df)
emot_counts = summarize_nominations(emot_df)

# Merge these counts back into the main df (on fs_student_id)
# We only care about these counts if the student is high-ability
# but we'll merge for everyone, then fill in 0 if not high.
merged = df[["fs_student_id", "s_merge_id", "high_math", "is_high"]].copy()

# Merge academic
merged = merged.merge(acad_counts, how="left", left_on="fs_student_id", right_on="to_id")
merged.rename(columns={"from_low":"high_nominated_acad_l_count","total":"acad_total"}, inplace=True)

# Merge emotional
merged = merged.merge(emot_counts, how="left", left_on="fs_student_id", right_on="to_id")
merged.rename(columns={"from_low":"high_nominated_emot_l_count","total":"emot_total"}, inplace=True)

# Fill missing counts with 0
for col in ["high_nominated_acad_l_count","acad_total","high_nominated_emot_l_count","emot_total"]:
    merged[col] = merged[col].fillna(0)

# Compute the requested variables:
#   high_nominated_acad_l: number of academic nominations from low-ability
#   high_nominated_acad_l_perc: % of all academic nominations received that come from low-ability
#   high_nominated_emot_l: number of emotional nominations from low-ability
#   high_nominated_emot_l_perc: % of all emotional nominations received that come from low-ability
# We only define these for high-ability students; for others, set them to 0.

def perc_or_zero(num, denom):
    return (num / denom * 100) if denom > 0 else 0

merged["high_nominated_acad_l"] = np.where(
    merged["is_high"]==1,
    merged["high_nominated_acad_l_count"],
    0
)
merged["high_nominated_acad_l_perc"] = np.where(
    merged["is_high"]==1,
    merged.apply(lambda x: perc_or_zero(x["high_nominated_acad_l_count"], x["acad_total"]), axis=1),
    0
)

merged["high_nominated_emot_l"] = np.where(
    merged["is_high"]==1,
    merged["high_nominated_emot_l_count"],
    0
)
merged["high_nominated_emot_l_perc"] = np.where(
    merged["is_high"]==1,
    merged.apply(lambda x: perc_or_zero(x["high_nominated_emot_l_count"], x["emot_total"]), axis=1),
    0
)

# Keep only the final columns needed (plus anything else you want to retain)
final_cols = [
    "fs_student_id",
    "s_merge_id",
    "high_math",
    "high_nominated_acad_l",
    "high_nominated_acad_l_perc",
    "high_nominated_emot_l",
    "high_nominated_emot_l_perc"
]
out_df = merged[final_cols].copy()

# Export
out_df.to_csv("/workspaces/ROC-network-analysis/output-files/high_nomination_counts-v2.csv", index=False)