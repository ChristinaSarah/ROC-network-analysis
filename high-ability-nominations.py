import pandas as pd

def compute_high_nomination_counts(input_csv: str, output_csv: str) -> None:
    """
    Reads 'input_csv' containing:
      - fs_student_id
      - high_math (coded as 'yes' or 'no')
      - academic_1, academic_2, academic_3
      - emot_1, emot_2, emot_3
    Outputs 'output_csv' with both academic and emot nomination counts:
      - high_nominated_acad, high_nominated_acad_h, high_nominated_acad_l
      - high_nominated_emot, high_nominated_emot_h, high_nominated_emot_l
    """

    ####################################
    # 1) Load the minimal columns needed
    ####################################
    usecols = [
        "fs_student_id", "high_math",
        "academic_1","academic_2","academic_3",
        "emot_1","emot_2","emot_3"
    ]
    df = pd.read_csv(input_csv, usecols=usecols)
    df1 = df.copy()
    df1["_temp_id_"] = range(len(df1))
    df = pd.read_csv(input_csv, usecols=usecols).drop(columns=["s_merge_id"], errors="ignore")


    # Convert 'yes'/'no' to 1/0
    map_bool = {"yes": 1, "no": 0}
    df["high_math"] = df["high_math"].map(map_bool)

    # Build dictionary for quick lookup: fs_student_id -> high_math
    df_unique = df[["fs_student_id","high_math"]].drop_duplicates(subset="fs_student_id")
    map_high_math = dict(zip(df_unique["fs_student_id"], df_unique["high_math"]))


    ##########################################################
    # 2) Helper function to get "high-nominated" counts
    ##########################################################
    def count_high_nominations(df_in, friend_cols, prefix):
        """
        df_in: DataFrame with columns [fs_student_id, high_math] + friend_cols
        friend_cols: list of columns to melt (e.g. ["academic_1","academic_2","academic_3"])
        prefix: string prefix for result columns (e.g. "acad" or "emot")

        Returns a DataFrame keyed by fs_student_id with:
          high_nominated_{prefix}
          high_nominated_{prefix}_h
          high_nominated_{prefix}_l
        """
        # Melt
        df_long = df_in.melt(
            id_vars=["fs_student_id","high_math"],
            value_vars=friend_cols,
            var_name=f"{prefix}_rank",
            value_name=f"{prefix}_friend_id"
        ).dropna(subset=[f"{prefix}_friend_id"])

        # Convert friend_id to numeric
        df_long[f"{prefix}_friend_id"] = pd.to_numeric(df_long[f"{prefix}_friend_id"], errors="coerce").astype("Int64")

        # Map friend_id -> high_math
        df_long[f"friend_is_high"] = df_long[f"{prefix}_friend_id"].map(map_high_math)

        # Keep rows where the friend is high
        df_long_high = df_long[df_long["friend_is_high"] == 1]

        # Group by the friend ID (the high student)
        grouped = df_long_high.groupby(f"{prefix}_friend_id")
        total_noms = grouped.size()
        high_noms = grouped["high_math"].sum()  # # of nominators who are also high
        low_noms = total_noms - high_noms

        # Build results DataFrame: fs_student_id vs counts
        unique_ids = df_unique["fs_student_id"].unique()
        out = pd.DataFrame({"fs_student_id": unique_ids})

        out[f"high_nominated_{prefix}"]   = out["fs_student_id"].map(total_noms).fillna(0)
        out[f"high_nominated_{prefix}_h"] = out["fs_student_id"].map(high_noms).fillna(0)
        out[f"high_nominated_{prefix}_l"] = out["fs_student_id"].map(low_noms).fillna(0)

        # Convert to int
        for c in [f"high_nominated_{prefix}", f"high_nominated_{prefix}_h", f"high_nominated_{prefix}_l"]:
            out[c] = out[c].astype(int)

        return out

    ####################################
    # 3) Compute academic & emot counts
    ####################################
    # For academic friend columns
    acad_results = count_high_nominations(df, ["academic_1","academic_2","academic_3"], prefix="acad")

    # For emot friend columns
    emot_results = count_high_nominations(df, ["emot_1","emot_2","emot_3"], prefix="emot")

    ####################################
    # 4) Merge academic + emot results
    ####################################
    results = pd.merge(acad_results, emot_results, on="fs_student_id", how="outer")

    ####################################
    # 5) Merge with original data if needed
    ####################################
    df_final = pd.read_csv(input_csv)
    df_final["high_math"] = df_final["high_math"].map(map_bool)
    df_final = df_final.merge(results, on="fs_student_id", how="left")

    df2 = df_final.copy()
    df2["_temp_id_"] = range(len(df2))
    df_merged = pd.merge(df1, df2, on="_temp_id_", how="inner", validate="1:1")

# Optionally drop the temporary column
    df_merged.drop(columns=["_temp_id_"], inplace=True)
    df_final=df_merged 


    # 6) Save
    df_final.to_csv(output_csv, index=False)
    print(f"Done. Results saved to {output_csv}")

# --------------------
# Example usage:
compute_high_nomination_counts("/workspaces/network-analysis/actual-code/input-files/roc_network_data_follow_up.csv", "/workspaces/network-analysis/actual-code/output-files/high_nomination_counts.csv")
