import numpy as np
import math
import pandas as pd

def compute_cross_ability_ratio(input_csv, output_csv):
    # 1) Load columns
    df = pd.read_csv(input_csv, 
                     usecols=["fs_student_id","fs_classroom","high_math",
                              "emot_1","emot_2","emot_3"])
    
    # 2) Convert 'yes'/'no' to 1/0, if necessary
    map_bool = {"yes": 1, "no": 0}
    df["high_math"] = df["high_math"].map(map_bool)
    
    # 3) Build a lookup: fs_student_id -> high_math
    df_unique = df[["fs_student_id","high_math"]].drop_duplicates(subset="fs_student_id")
    map_high = dict(zip(df_unique["fs_student_id"], df_unique["high_math"]))

    # 4) Reshape friend columns into one 'friend_id'
    df_long = df.melt(
        id_vars=["fs_student_id","fs_classroom","high_math"],
        value_vars=["emot_1","emot_2","emot_3"],
        var_name="friend_rank",
        value_name="friend_id"
    ).dropna(subset=["friend_id"])

    df_long["friend_id"] = pd.to_numeric(df_long["friend_id"], errors="coerce").astype("Int64")

    # 5) Map friend_id -> high_math
    df_long["friend_high_math"] = df_long["friend_id"].map(map_high)

    # 6) Count cross-ability nominations
    #    A cross-ability nomination is one where nominator's high_math != friend's high_math
    #    We'll code cross_ability=1 if they differ, else 0
 
    mask = df_long["friend_high_math"].notna()

    #  On those rows only, set cross_ability to 1 if high_math != friend_high_math, else 0
    df_long.loc[mask, "cross_ability"] = (
        df_long.loc[mask, "high_math"] != df_long.loc[mask, "friend_high_math"]
    ).astype(int)

    # 7) Group by classroom (nominator’s classroom), compute:
    #      x+y = sum of cross_ability
    #      n   = total nominations in that classroom
    grouped = df_long.groupby("fs_classroom", dropna=False)
    cross_sum = grouped["cross_ability"].sum()   # (x + y)
    total_noms = grouped.size()                  # n

    # 8) Create results DataFrame with ratio
    results = pd.DataFrame({
        "fs_classroom": cross_sum.index, 
        "cross_ability_count": cross_sum.values,
        "total_nominations": total_noms.values
    })
    results["cross_ability_ratio"] = results["cross_ability_count"] / results["total_nominations"]

    # 9) Save to CSV (fs_classroom + ratio)
    results[["fs_classroom","cross_ability_ratio"]].to_csv(output_csv, index=False)
    print(f"Done. Results saved to {output_csv}")


# Example usage:
# compute_cross_ability_ratio("roc_network_data_follow_up.csv", "cross_ability_ratio_per_classroom.csv")

def compute_mu(n_r, n_h):

    p_r = compute_p(np.sum(n_h), np.sum(n_r))

    p_h = compute_p(np.sum(n_r), np.sum(n_h))

    num = compute_num(n_r, n_h, p_r, p_h)

    den = compute_den(n_r, n_h)

    return num / den

def compute_den(n_r, n_h):
    # np.arange(1,4) creates an array [1, 2, 3]
    # n_r + n_h returns the elementwise sum of the two vectors
    # np.arange(1, 4) * (n_r + n_h) computes the elementwise numltiplication of the two vectors
    # the np.sum sums all the entries in the resulting vector
    return np.sum(np.arange(1, 4) * (n_r + n_h))

def compute_p(n_successes, n_fails):
    p = np.zeros((3,3))
    for i in range(1, 4):
        for j in range(1, i+1):
            num_1 = math.comb(n_successes, j)
            num_2 = math.comb(n_fails - 1, i - j)
            den = math.comb(n_successes + n_fails - 1, i)
            if den > 0:  # Only execute if den > 0
                p[i-1, j-1] = (num_1 * num_2) / den
            else:
                p[i-1, j-1] = 0  # Set to 0 if denominator is zero to avoid division by zero

    return p






def compute_num(n_r, n_h, p_r, p_h):
    sum = 0.
    for x in range(1,4):
        for y in range(1,x+1):
            sum += n_r[x-1] * p_r[x-1,y-1] * y
            sum += n_h[x-1] * p_h[x-1,y-1] * y

    return sum

# 1) Load only columns needed
df = pd.read_csv("/workspaces/ROC-network-analysis/input-files/roc_network_data_follow_up.csv",
                 usecols=["fs_classroom","fs_student_id","high_math","emot_1","emot_2","emot_3"])

# 2) If 'high_math' is 'yes'/'no', convert to 1/0
map_bool = {"yes": 1, "no": 0}
df["high_math"] = df["high_math"].map(map_bool)

# 3) Count how many friends (0–3) each student nominated
df["n_friends"] = df[["emot_1","emot_2","emot_3"]].notna().sum(axis=1)

# 4) Create a boolean for high-ability
df["is_high"] = (df["high_math"] == 1)

# 5) Group by (fs_classroom, is_high, n_friends)
grouped = df.groupby(["fs_classroom","is_high","n_friends"]).size().reset_index(name="count")

# 6) Pivot into wide form
#    This yields columns with multi-index: (is_high, n_friends)
pivoted = grouped.pivot_table(index="fs_classroom",
                              columns=["is_high","n_friends"],
                              values="count",
                              fill_value=0)

# 7) Rename columns: (False,1)->'low_1', (True,3)->'high_3', etc.
pivoted.columns = [
    f"{'high' if is_high else 'low'}_{n}" 
    for (is_high, n) in pivoted.columns
]

# 8) Reset index so 'fs_classroom' is a normal column
pivoted = pivoted.reset_index()

# 9) Ensure all columns exist (some classes might not have any students with 1,2,3 friends)
for col in ["low_1","low_2","low_3","high_1","high_2","high_3"]:
    if col not in pivoted.columns:
        pivoted[col] = 0

# ----------------------------------------------------------
# EXTRACT THE ARRAYS FOR USE IN ANOTHER FUNCTION
# ----------------------------------------------------------

# 10) Convert to NumPy arrays, shape = (num_classrooms, 3)
low_arrays = pivoted[["low_1","low_2","low_3"]].to_numpy(dtype=int)
high_arrays = pivoted[["high_1","high_2","high_3"]].to_numpy(dtype=int)

# Now each row in 'low_arrays[i]' is [low_1_count, low_2_count, low_3_count] for classroom i
# Similarly, 'high_arrays[i]' is [high_1_count, high_2_count, high_3_count].

# Example usage in some function:
# my_function(low_arrays, high_arrays)  # hypothetical call

mu = np.zeros((np.shape(low_arrays)[0]))
for i in range(np.shape(low_arrays)[0]):
    print(low_arrays[i])
    print(high_arrays[i])
    if np.max(high_arrays[i]) == 0 or np.max(low_arrays[i]) == 0:
        mu[i] = 'NaN'
    else:
        mu[i] = compute_mu(low_arrays[i], high_arrays[i])

# ----------------------------------------------------------
# MERGE CLASSROOM IDS & ARRAYS INTO A SINGLE CSV
# ----------------------------------------------------------

# 11) If you want to store each array row in one CSV cell, convert them to Python lists
#     so each row is a list and can be converted to a string. 
pivoted["low_array"] = list(low_arrays.tolist())
pivoted["high_array"] = list(high_arrays.tolist())
pivoted["mu"] = list(mu.tolist())

# 12) Now pivoted has columns: fs_classroom, low_1, low_2, low_3, high_1, high_2, high_3, low_array, high_array
#     We'll keep just the columns we want in the final CSV:
final_df = pivoted[["fs_classroom","low_array","high_array","mu"]]

# 13) Write to CSV. The arrays will show up as string representations (e.g. "[1, 2, 0]")
final_df.to_csv("classroom_arrays.csv", index=False)

print("Done. 'classroom_arrays.csv' saved.")
print("Sample output:")
print(final_df.head())

compute_cross_ability_ratio("/workspaces/ROC-network-analysis/input-files/roc_network_data_follow_up.csv", "/workspaces/ROC-network-analysis/output-files/classroom_segregation_actual.csv")