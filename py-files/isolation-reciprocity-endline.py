import pandas as pd

# Load the data
df = pd.read_csv("/workspaces/ROC-network-analysis/input-files/roc_network_data_endline_high_ability.csv")

# We'll compute classroom-level network indicators by grouping on 'classroom_id'.
# For each classroom, we only consider edges (nominations) where both the nominator and
# the nominee are in the same classroom.

results = []

for classroom, group in df.groupby("classroom_id"):
    # Gather the set of student IDs in this classroom
    students_in_class = set(group["student_id"].unique())
   
    # Build directed edges for academic support and emotional support
    friend_edges = set()
    support_edges = set()
   
    for _, row in group.iterrows():
        from_id = row["student_id"]
       
        # Academic edges
        for col in ["friend_1","friend_2","friend_3"]:
            to_id = row[col]
            if pd.notna(to_id) and to_id in students_in_class:
                friend_edges.add((from_id, to_id))
       
        # Emotional edges
        for col in ["support_1","support_2","support_3"]:
            to_id = row[col]
            if pd.notna(to_id) and to_id in students_in_class:
                support_edges.add((from_id, to_id))
   
    # 1) isolate_in_f: share of students who are NOT nominated as friend by anyone
    if len(students_in_class) > 0:
        friend_nominees = {to_id for (_, to_id) in friend_edges}
        not_nominated_friend = students_in_class - friend_nominees
        isolate_in_f = len(not_nominated_friend) / len(students_in_class)
    else:
        isolate_in_f = 0
   
    # 2) isolate_in_s: share of students who are NOT nominated as support by anyone
    if len(students_in_class) > 0:
        support_nominees = {to_id for (_, to_id) in support_edges}
        not_nominated_support = students_in_class - support_nominees
        isolate_in_s = len(not_nominated_support) / len(students_in_class)
    else:
        isolate_in_s = 0
   
    # 3) reciprocity_share_f: share of friendship ties that are reciprocal
    total_friend_edges = len(friend_edges)
    if total_friend_edges > 0:
        # Count how many edges are reciprocated
        # i.e. for each (i->j), check if (j->i) is also in the set
        reciprocated_friend = sum(1 for (i, j) in friend_edges if (j, i) in friend_edges)
        reciprocity_share_f = reciprocated_friend / total_friend_edges
    else:
        reciprocity_share_f = 0
   
    # 4) reciprocity_share_s: share of support ties that are reciprocal
    total_support_edges = len(support_edges)
    if total_support_edges > 0:
        reciprocated_support = sum(1 for (i, j) in support_edges if (j, i) in support_edges)
        reciprocity_share_s = reciprocated_support / total_support_edges
    else:
        reciprocity_share_s = 0
   
    # Store the results
    results.append({
        "classroom_id": classroom,
        "isolate_in_f": isolate_in_f,
        "isolate_in_s": isolate_in_s,
        "reciprocity_share_f": reciprocity_share_f,
        "reciprocity_share_s": reciprocity_share_s
    })

# Convert to a DataFrame
out_df = pd.DataFrame(results)

# Export to CSV
out_df.to_csv("/workspaces/ROC-network-analysis/output-files/roc_isolation_reciprocity_endline_high_ability.csv", index=False)
print("Done. 'roc_isolation_reciprocity_endline.csv' saved.")