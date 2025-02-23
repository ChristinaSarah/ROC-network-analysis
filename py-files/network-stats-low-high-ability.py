
import pandas as pd
import numpy as np



# Step 1: Load dataset
df = pd.read_csv("/workspaces/ROC-network-analysis/input-files/roc_network_data_follow_up.csv")  # Change "data.csv" to your actual file name


# Step 2: Create dictionaries mapping student IDs to ability levels
ability_math = df.set_index("fs_student_id")["high_math"].to_dict()
ability_raven = df.set_index("fs_student_id")["high_raven"].to_dict()
ability_bangla = df.set_index("fs_student_id")["high_bangla"].to_dict()
ability_eyes = df.set_index("fs_student_id")["high_eyes"].to_dict()

# Step 3: Function to compute indicators with missing data handling
def lowhigh_ability_metrics(student_id, friends, ability_dict):
    """Returns (binary indicator, percentage of friends with high ability), or NaN if no valid friends"""
    
    # If student_id is missing, return NaN
    if pd.isna(student_id):
        return (np.nan, np.nan)

    student_ability = ability_dict.get(student_id, None)  # Get student i's ability
    
    # If student ability is missing or if the student is not low ability, return (0, 0)
    if student_ability is None or student_ability == "yes":
        return (0, 0)

    # Filter out missing friends
    valid_friends = [friend_id for friend_id in friends if not pd.isna(friend_id) and friend_id in ability_dict]

    # If no valid friends, return NaN
    if not valid_friends:
        return (np.nan, np.nan)

    # Count high-ability friends
    high_count = sum(1 for friend_id in valid_friends if ability_dict[friend_id] == "yes")

    # Compute percentage (only based on valid friends)
    high_percentage = high_count / len(valid_friends) if valid_friends else np.nan

    return (1 if high_count > 0 else 0, high_percentage)

# Step 4: Compute new indicators for all ability types
ability_types = {
    "math": ability_math,
    "raven": ability_raven,
    "bangla": ability_bangla,
    "eyes": ability_eyes
}

friend_types = {
    "emot": ["emot_1", "emot_2", "emot_3"],
    "acad": ["academic_1", "academic_2", "academic_3"]
}

# Loop through ability types and friend types to compute all indicators
for ability_name, ability_dict in ability_types.items():
    for friend_type, friend_cols in friend_types.items():
        indicator_name = f"lowhigh_inter_{friend_type}_{ability_name}"
        percentage_name = f"lowhigh_inter_{friend_type}_{ability_name}_perc"

        df[[indicator_name, percentage_name]] = df.apply(
            lambda row: lowhigh_ability_metrics(
                row["fs_student_id"], 
                [row[col] for col in friend_cols], 
                ability_dict
            ), 
            axis=1, result_type="expand"
        )


# Step 5: Export the updated dataset
df.to_csv("/workspaces/ROC-network-analysis/output-files/follow_up_inter_ability.csv", index=False)  # Saves the new dataset
print("Updated dataset saved as follow_up_inter_ability.csv")
