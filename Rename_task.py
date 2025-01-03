import pandas as pd
import re

# Read Excel file and sheet
input_file = 'process_state_log.xlsx'
sheet_name = 'flowchart_process_states_log'
df = pd.read_excel(input_file, sheet_name=sheet_name)

# Ensure 'block' column exists
if 'block' not in df.columns:
    raise KeyError("The 'block' column was not found in the sheet.")

# Prompt user for the task swap order
while True:
    try:
        user_input = input("Enter the new task order (comma-separated list of 30 integers, e.g., 1,2,3,...,30): ")
        swap_order = [int(x) for x in user_input.split(',')]

        # Validate input
        if sorted(swap_order) != list(range(1, 31)):
            raise ValueError("Invalid input. The list must contain all numbers from 1 to 30 exactly once.")
        break
    except ValueError as e:
        print(e)
        continue

# Create task mappings
old_tasks = [f'Task{str(i).zfill(2)}' for i in range(1, 31)]
new_tasks = [f'Task{str(i).zfill(2)}' for i in swap_order]
task_mapping = dict(zip(old_tasks, new_tasks))

# Use temporary placeholders to avoid overwriting issues
temp_mapping = {task: f'temp_{task}' for task in old_tasks}

# Step 1: Replace all tasks with temp values
df['block'] = df['block'].replace(temp_mapping, regex=True)

# Step 2: Replace temp values with final swapped tasks
temp_to_final_mapping = {f'temp_{k}': v for k, v in task_mapping.items()}
df['block'] = df['block'].replace(temp_to_final_mapping, regex=True)

# Save to a new Excel file
output_file = 'process_state_log_readytorebalance.xlsx'
df.to_excel(output_file, sheet_name=sheet_name, index=False)

print(f"Tasks have been successfully updated and saved to {output_file}")
