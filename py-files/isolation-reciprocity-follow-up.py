import pandas as pd

# Load the data
df = pd.read_csv("/workspaces/ROC-network-analysis/input-files/roc_network_data_follow_up.csv")

# We'll compute classroom-level network indicators by grouping on 'fs_classroom'.
# For each classroom, we only consider edges (nominations) where both the nominator and
# the nominee are in the same classroom.

results = []

for classroom, group in df.groupby("fs_classroom"):
    # Gather the set of student IDs in this classroom
    students_in_class = set(group["fs_student_id"].unique())
   
    # Build directed edges for academic support and emotional support
    academic_edges = set()
    emotional_edges = set()
   
    for _, row in group.iterrows():
        from_id = row["fs_student_id"]
       
        # Academic edges
        for col in ["academic_1","academic_2","academic_3"]:
            to_id = row[col]
            if pd.notna(to_id) and to_id in students_in_class:
                academic_edges.add((from_id, to_id))
       
        # Emotional edges
        for col in ["emot_1","emot_2","emot_3"]:
            to_id = row[col]
            if pd.notna(to_id) and to_id in students_in_class:
                emotional_edges.add((from_id, to_id))
   
    # 1) isolate_in_a: share of students who are NOT nominated academically by anyone
    if len(students_in_class) > 0:
        academic_nominees = {to_id for (_, to_id) in academic_edges}
        not_nominated_academic = students_in_class - academic_nominees
        isolate_in_a = len(not_nominated_academic) / len(students_in_class)
    else:
        isolate_in_a = 0
   
    # 2) isolate_in_e: share of students who are NOT nominated emotionally by anyone
    if len(students_in_class) > 0:
        emotional_nominees = {to_id for (_, to_id) in emotional_edges}
        not_nominated_emotional = students_in_class - emotional_nominees
        isolate_in_e = len(not_nominated_emotional) / len(students_in_class)
    else:
        isolate_in_e = 0
   
    # 3) reciprocity_share_a: share of academic ties that are reciprocal
    total_acad_edges = len(academic_edges)
    if total_acad_edges > 0:
        # Count how many edges are reciprocated
        # i.e. for each (i->j), check if (j->i) is also in the set
        reciprocated_acad = sum(1 for (i, j) in academic_edges if (j, i) in academic_edges)
        reciprocity_share_a = reciprocated_acad / total_acad_edges
    else:
        reciprocity_share_a = 0
   
    # 4) reciprocity_share_e: share of emotional ties that are reciprocal
    total_emot_edges = len(emotional_edges)
    if total_emot_edges > 0:
        reciprocated_emot = sum(1 for (i, j) in emotional_edges if (j, i) in emotional_edges)
        reciprocity_share_e = reciprocated_emot / total_emot_edges
    else:
        reciprocity_share_e = 0
   
    # Store the results
    results.append({
        "fs_classroom": classroom,
        "isolate_in_a": isolate_in_a,
        "isolate_in_e": isolate_in_e,
        "reciprocity_share_a": reciprocity_share_a,
        "reciprocity_share_e": reciprocity_share_e
    })

# Convert to a DataFrame
out_df = pd.DataFrame(results)

# Export to CSV
out_df.to_csv("/workspaces/ROC-network-analysis/output-files/roc_isolation_reciprocity_follow_up.csv", index=False)
print("Done. 'roc_isolation_reciprocity_follow_up.csv' saved.")