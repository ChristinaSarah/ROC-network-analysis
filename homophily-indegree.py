import pandas as pd
import numpy as np

def compute_in_degree_homophily(input_csv: str, output_csv: str):
    """
    Reads 'input_csv' with columns:
      - fs_classroom, fs_student_id, s_merge_id
      - high_math ('yes' or 'no')
      - academic_1, academic_2, academic_3
      - emot_1, emot_2, emot_3
    and produces an output CSV with columns:
      - in_low_low_acad_math, in_low_low_acad_math_b, ...
      - in_high_high_acad_math, in_high_high_acad_math_b, ...
      - in_low_low_acad_math_perc, in_high_high_acad_math_perc, ...
      - etc. (same for emot), but from the perspective of who *gets* nominated.
    """

    # 1) Load minimal columns
    usecols = [
        "fs_classroom", "fs_student_id", "s_merge_id",
        "high_math",
        "academic_1","academic_2","academic_3",
        "emot_1","emot_2","emot_3"
    ]
    df = pd.read_csv(input_csv, usecols=usecols)

    # 2) Convert 'yes'/'no' -> 1 (high) / 0 (low)
    map_yes_no = {"yes": 1, "no": 0}
    df["is_high"] = df["high_math"].map(map_yes_no)

    # Build lookup: fs_student_id -> (is_high, s_merge_id, fs_classroom) if needed
    # so we can map from "friend ID" back to that friend's ability
    df_unique = df[["fs_student_id","is_high","s_merge_id","fs_classroom"]].drop_duplicates("fs_student_id")
    ability_map = dict(zip(df_unique["fs_student_id"], df_unique["is_high"]))
    merge_map   = dict(zip(df_unique["fs_student_id"], df_unique["s_merge_id"]))
    class_map   = dict(zip(df_unique["fs_student_id"], df_unique["fs_classroom"]))

    def get_ability(stud_id):
        if pd.isna(stud_id):
            return np.nan
        return ability_map.get(int(stud_id), np.nan)

    # -------------------------------------------------------------------------
    # 3) Build a "long" DataFrame of nominations for academic & emot separately
    #    Each row is: (nominator, nominee, nominator_is_high, domain="acad"/"emot")
    # -------------------------------------------------------------------------
    def melt_friends(df_in, friend_cols, domain_name):
        """
        df_in: the main DataFrame with columns [fs_student_id, is_high] + friend_cols
        domain_name: "acad" or "emot"
        
        Returns a DataFrame with columns:
          - nominator_id
          - nominator_is_high
          - nominee_id
          - nominee_is_high (if known)
          - domain ("acad" or "emot")
        ignoring rows where nominee_id is NaN
        """
        df_m = df_in.melt(
            id_vars=["fs_student_id","is_high"],
            value_vars=friend_cols,
            var_name=f"{domain_name}_rank",
            value_name="nominee_id"
        ).dropna(subset=["nominee_id"])

        df_m = df_m.rename(columns={
            "fs_student_id": "nominator_id",
            "is_high": "nominator_is_high"
        })
        df_m["domain"] = domain_name
        # Map the nominee's ability
        df_m["nominee_is_high"] = df_m["nominee_id"].apply(get_ability)
        return df_m[["nominator_id","nominator_is_high","nominee_id","nominee_is_high","domain"]]

    # Melt academic
    df_acad = melt_friends(df, ["academic_1","academic_2","academic_3"], domain_name="acad")
    # Melt emot
    df_emot = melt_friends(df, ["emot_1","emot_2","emot_3"], domain_name="emot")

    # Combine
    df_long = pd.concat([df_acad, df_emot], ignore_index=True)

    # -------------------------------------------------------------------------
    # 4) For each nominee, count how many times they are nominated "low->low", "high->high", ignoring missing ability
    #    We'll group by nominee_id
    # -------------------------------------------------------------------------
    # Create columns for "low->low" or "high->high" flags
    df_long["low_low_flag"] = (
        (df_long["nominee_is_high"] == 0) & 
        (df_long["nominator_is_high"] == 0)
    ).astype(int)

    df_long["high_high_flag"] = (
        (df_long["nominee_is_high"] == 1) &
        (df_long["nominator_is_high"] == 1)
    ).astype(int)

    # We'll also track whether the nominator's ability is known for each row
    df_long["nominator_known"] = df_long["nominator_is_high"].notna().astype(int)

    # Now group by (nominee_id, domain) so we can separately handle "acad" or "emot"
    grouped = df_long.groupby(["nominee_id","domain"], dropna=False)

    # For each group, sum up how many "low->low", "high->high", and how many nominators are known
    agg_df = grouped.agg({
        "low_low_flag": "sum",
        "high_high_flag": "sum",
        "nominator_known": "sum"
    }).reset_index()

    # We'll pivot to get columns for "acad"/"emot" in separate columns
    pivoted = agg_df.pivot_table(
        index="nominee_id",
        columns="domain",
        values=["low_low_flag","high_high_flag","nominator_known"],
        fill_value=0
    )

    # Flatten MultiIndex columns
    pivoted.columns = [
        f"{colname}_{domain}" for colname, domain in pivoted.columns
    ]
    pivoted = pivoted.reset_index().rename(columns={"nominee_id":"fs_student_id"})

    # This pivoted now has columns like:
    # low_low_flag_acad, low_low_flag_emot, high_high_flag_acad, high_high_flag_emot, 
    # nominator_known_acad, nominator_known_emot
    # all keyed by fs_student_id (the nominee).

    # -------------------------------------------------------------------------
    # 5) Merge with original to get the nominee's ability so we can compute "in_low_low_acad_math" etc.
    # -------------------------------------------------------------------------
    # We'll keep them in the same DataFrame for clarity.
    # If a student never appeared as a nominee, they'll be missing from pivoted. We'll do an outer merge.
    df_nominees = pd.merge(
        df_unique, pivoted,
        on="fs_student_id", how="left"
    )

    # Fill zeros for missing columns if they never got nominated
    for c in ["low_low_flag_acad","low_low_flag_emot","high_high_flag_acad","high_high_flag_emot",
              "nominator_known_acad","nominator_known_emot"]:
        if c not in df_nominees.columns:
            df_nominees[c] = 0
    df_nominees.fillna(value={
        "low_low_flag_acad":0, "low_low_flag_emot":0,
        "high_high_flag_acad":0,"high_high_flag_emot":0,
        "nominator_known_acad":0,"nominator_known_emot":0
    }, inplace=True)

    # Now rename to the final column names requested:
    # "in_low_low_acad_math", "in_high_high_acad_math" etc.
    df_nominees["in_low_low_acad_math"]   = df_nominees["low_low_flag_acad"]
    df_nominees["in_low_low_emot_math"]   = df_nominees["low_low_flag_emot"]
    df_nominees["in_high_high_acad_math"] = df_nominees["high_high_flag_acad"]
    df_nominees["in_high_high_emot_math"] = df_nominees["high_high_flag_emot"]

    # Binary indicators: 1 if count > 0
    df_nominees["in_low_low_acad_math_b"] = np.where(
        (df_nominees["is_high"] == 0) & (df_nominees["in_low_low_acad_math"] > 0),
        1, 0
    )
    df_nominees["in_low_low_emot_math_b"] = np.where(
        (df_nominees["is_high"] == 0) & (df_nominees["in_low_low_emot_math"] > 0),
        1, 0
    )
    df_nominees["in_high_high_acad_math_b"] = np.where(
        (df_nominees["is_high"] == 1) & (df_nominees["in_high_high_acad_math"] > 0),
        1, 0
    )
    df_nominees["in_high_high_emot_math_b"] = np.where(
        (df_nominees["is_high"] == 1) & (df_nominees["in_high_high_emot_math"] > 0),
        1, 0
    )

    # Denominator: how many nominators have known ability?
    # We'll do separate columns for acad vs emot:
    df_nominees["in_valid_acad_nominators"] = df_nominees["nominator_known_acad"]
    df_nominees["in_valid_emot_nominators"] = df_nominees["nominator_known_emot"]

    # Fractions:
    # "in_low_low_acad_math_perc" = (# low->low) / (# valid nominators) if the nominee is low, else NaN
    df_nominees["in_low_low_acad_math_perc"] = np.where(
        df_nominees["is_high"] == 0,
        df_nominees["in_low_low_acad_math"] / df_nominees["in_valid_acad_nominators"],
        np.nan
    )
    df_nominees["in_low_low_emot_math_perc"] = np.where(
        df_nominees["is_high"] == 0,
        df_nominees["in_low_low_emot_math"] / df_nominees["in_valid_emot_nominators"],
        np.nan
    )

    # Similarly for high
    df_nominees["in_high_high_acad_math_perc"] = np.where(
        df_nominees["is_high"] == 1,
        df_nominees["in_high_high_acad_math"] / df_nominees["in_valid_acad_nominators"],
        np.nan
    )
    df_nominees["in_high_high_emot_math_perc"] = np.where(
        df_nominees["is_high"] == 1,
        df_nominees["in_high_high_emot_math"] / df_nominees["in_valid_emot_nominators"],
        np.nan
    )

    # Merge back s_merge_id, fs_classroom from df_unique if needed
    # (We already have them in df_nominees via merge with df_unique)
    # But let's ensure we have everything consistent:

    # -------------------------------------------------------------------------
    # 6) Final output columns
    # -------------------------------------------------------------------------
    out_cols = [
        "fs_student_id", "s_merge_id", "fs_classroom", 
        # Basic counts
        "in_low_low_acad_math","in_low_low_emot_math",
        "in_high_high_acad_math","in_high_high_emot_math",
        # Binary
        "in_low_low_acad_math_b","in_low_low_emot_math_b",
        "in_high_high_acad_math_b","in_high_high_emot_math_b",
        # Fractions
        "in_low_low_acad_math_perc","in_low_low_emot_math_perc",
        "in_high_high_acad_math_perc","in_high_high_emot_math_perc"
    ]
    # Some might not exist if the columns never got created in pivot. Fill in with 0 or NaN:
    for col in out_cols:
        if col not in df_nominees.columns:
            df_nominees[col] = np.nan

    df_final = df_nominees[out_cols].copy()

    # 7) Write to CSV
    df_final.to_csv(output_csv, index=False)
    print(f"Done. Results saved to {output_csv}")

# Example usage:
compute_in_degree_homophily(
    input_csv="/workspaces/network-analysis/actual-code/input-files/roc_network_data_follow_up.csv",
    output_csv="/workspaces/network-analysis/actual-code/output-files/homophily-indegree.csv"
)

