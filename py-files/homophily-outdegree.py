import pandas as pd
import numpy as np

def compute_same_ability_homophily(
    input_csv: str,
    output_csv: str
):
    """
    Reads a CSV with:
      - fs_classroom, fs_student_id, s_merge_id
      - high_math ('yes' or 'no')
      - academic_1, academic_2, academic_3
      - emot_1, emot_2, emot_3

    Writes a CSV with columns including:
      low_low_acad_math, low_low_emot_math, high_high_acad_math, high_high_emot_math
      low_low_acad_math_b, low_low_emot_math_b, ...
      low_low_acad_math_perc, etc.
    """

    # 1) Load needed columns
    usecols = [
        "fs_classroom", "fs_student_id", "s_merge_id",
        "high_math",
        "academic_1","academic_2","academic_3",
        "emot_1","emot_2","emot_3"
    ]
    df = pd.read_csv(input_csv, usecols=usecols)

    # 2) Convert high_math from 'yes'/'no' to numeric (1=high, 0=low)
    map_yes_no = {"yes": 1, "no": 0}
    df["is_high"] = df["high_math"].map(map_yes_no)

    # Build a dict: fs_student_id -> is_high
    df_unique = df[["fs_student_id","is_high"]].drop_duplicates("fs_student_id")
    map_ability = dict(zip(df_unique["fs_student_id"], df_unique["is_high"]))

    # Helper: get ability of a friend_id
    def get_ability(friend_id):
        if pd.isna(friend_id):
            return np.nan
        friend_id = int(friend_id)
        return map_ability.get(friend_id, np.nan)

    # 3) For each friend slot, figure out that friend's ability
    for i in [1,2,3]:
        df[f"acad_friend_ability_{i}"] = df[f"academic_{i}"].apply(get_ability)
        df[f"emot_friend_ability_{i}"]  = df[f"emot_{i}"].apply(get_ability)

    # 4) Count how many same-ability ties in academic/emotional for each row
    #    - low_low_acad_math: how many academic ties are low→low
    #    - high_high_acad_math: how many academic ties are high→high
    df["low_low_acad_math"]  = 0
    df["low_low_emot_math"]  = 0
    df["high_high_acad_math"] = 0
    df["high_high_emot_math"] = 0

    for i in [1,2,3]:
        # low->low academic
        df["low_low_acad_math"] += (
            (df["is_high"] == 0) &
            (df[f"acad_friend_ability_{i}"] == 0)
        ).astype(int)

        # low->low emotional
        df["low_low_emot_math"] += (
            (df["is_high"] == 0) &
            (df[f"emot_friend_ability_{i}"] == 0)
        ).astype(int)

        # high->high academic
        df["high_high_acad_math"] += (
            (df["is_high"] == 1) &
            (df[f"acad_friend_ability_{i}"] == 1)
        ).astype(int)

        # high->high emotional
        df["high_high_emot_math"] += (
            (df["is_high"] == 1) &
            (df[f"emot_friend_ability_{i}"] == 1)
        ).astype(int)

    # 5) Build binary indicators:
    #    1 if a student has at least 1 same-ability friend (in that domain), else 0
    df["low_low_acad_math_b"] = np.where(
        (df["is_high"] == 0) & (df["low_low_acad_math"] > 0), 1, 0
    )
    df["low_low_emot_math_b"] = np.where(
        (df["is_high"] == 0) & (df["low_low_emot_math"] > 0), 1, 0
    )
    df["high_high_acad_math_b"] = np.where(
        (df["is_high"] == 1) & (df["high_high_acad_math"] > 0), 1, 0
    )
    df["high_high_emot_math_b"] = np.where(
        (df["is_high"] == 1) & (df["high_high_emot_math"] > 0), 1, 0
    )

    # 6) Compute how many friend slots have *known* ability (ignore missing ability)
    df["valid_acad_friend_count"] = 0
    df["valid_emot_friend_count"] = 0
    for i in [1,2,3]:
        df["valid_acad_friend_count"] += df[f"acad_friend_ability_{i}"].notna().astype(int)
        df["valid_emot_friend_count"] += df[f"emot_friend_ability_{i}"].notna().astype(int)

    # 7) Compute fractions: share of nominated friends who match the student's own ability
    #    For a low-ability student:
    df["low_low_acad_math_perc"] = np.where(
        df["is_high"] == 0,
        df["low_low_acad_math"] / df["valid_acad_friend_count"],
        np.nan
    )
    df["low_low_emot_math_perc"] = np.where(
        df["is_high"] == 0,
        df["low_low_emot_math"] / df["valid_emot_friend_count"],
        np.nan
    )

    #    For a high-ability student:
    df["high_high_acad_math_perc"] = np.where(
        df["is_high"] == 1,
        df["high_high_acad_math"] / df["valid_acad_friend_count"],
        np.nan
    )
    df["high_high_emot_math_perc"] = np.where(
        df["is_high"] == 1,
        df["high_high_emot_math"] / df["valid_emot_friend_count"],
        np.nan
    )

    # 8) Prepare the final DataFrame
    out_cols = [
        "fs_classroom","fs_student_id","s_merge_id",
        "low_low_acad_math","low_low_emot_math",
        "high_high_acad_math","high_high_emot_math",
        "low_low_acad_math_b","low_low_emot_math_b",
        "high_high_acad_math_b","high_high_emot_math_b",
        "low_low_acad_math_perc","low_low_emot_math_perc",
        "high_high_acad_math_perc","high_high_emot_math_perc"
    ]
    df_final = df[out_cols].copy()

    # 9) Save to CSV
    df_final.to_csv(output_csv, index=False)
    print(f"Done. Results saved to {output_csv}")


# Example usage:
compute_same_ability_homophily(
    input_csv="/workspaces/ROC-network-analysis/input-files/roc_network_data_follow_up.csv",
    output_csv="/workspaces/ROC-network-analysis/output-files/homophily-outdegree.csv"
)

