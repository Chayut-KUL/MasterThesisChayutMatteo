import pandas as pd
import random
import numpy as np
from collections import deque

task_to_operator = ["Worker2"]
task8 = [10.033, 9.585, 8.901, 6.799, 3.728, 3.508, 4.58, 3.917, 3.928, 3.392, 3.824, 3.598, 3.838, 3.116, 3.701, 6.865, 8.101, 4.323, 3.584, 4.067, 3.133, 5.765, 4.227, 2.676, 7.638, 5.76, 3.521, 4.395, 3.645, 3.252, 4.544, 0.805, 2.159, 2.257, 2.617, 4.6, 5.492, 3.752, 1.958, 2.362, 8.048, 2.566, 3.602]
task24 = [3.107, 4.904, 3.643, 3.007, 4.612, 3.657, 4.064, 3.212, 9.626, 3.122, 7.854, 3.746, 4.538, 4.15, 3.627, 4.419, 3.713, 3.028, 2.87, 3.492, 3.645, 3.33, 3.179, 5.093, 3.266, 3.851, 2.885, 5.044, 4.637, 2.484, 6.133, 5.568, 3.342, 4.357, 3.489, 6.23, 6.46, 3.871, 4.113, 4.08, 3.902, 3.557, 4.469]
task29 = [6.343, 6.222, 6.553, 6.22, 11.065, 5.152, 5.658, 6.634, 5.372, 6.179, 4.746, 5.445, 6.005, 10.601, 5.733, 6.731, 6.94, 6.342, 4.392, 4.627, 10.484, 3.858, 8.986, 3.754, 6.353, 6.158, 5.279, 5.628, 3.697, 4.446, 7.197, 5.253, 3.513, 4.532, 20.713, 7.302, 5.209, 7.964, 5.929, 5.662, 5.525, 6.47, 7.399]
task30 = [3.81, 4.909, 2.087, 10.389, 4.013, 3.989, 4.58, 3.4, 3.812, 5.053, 5.652, 8.554, 4.117, 5.715, 7.133, 3.693, 3.548, 4.436, 6.372, 4.194, 3.676, 11.876, 5.571, 5.031, 3.765, 4.844, 13.956, 4.414, 3.244, 5.302, 7.091, 5.834, 4.737, 3.927, 3.33, 3.415, 3.96, 8.031, 3.748, 6.166, 3.993, 4.584, 3.116]

tasks = [0]*30
tasks[7] = task8
tasks[23] = task24
tasks[28] = task29
tasks[29] = task30

learning_rates = [0.838, 0.99, 0.931, 0.982, 1, 1, 0.891, 1, 0.955, 0.959, 1, 0.979, 0.996, 0.999, 0.922, 1, 1, 1, 1, 1, 0.984, 0.999, 1, 1, 0.975, 1, 1, 1, 0.983, 1]
adjusted_learning_rates = [
    (0.838 + 0.8) / 2,  # 0.819
    (0.99 + 0.8) / 2,   # 0.895
    (0.931 + 0.8) / 2,  # 0.8655
    (0.982 + 0.8) / 2,  # 0.891
    (1 + 0.8) / 2,      # 0.9
    (1 + 0.8) / 2,      # 0.9
    (0.891 + 0.8) / 2,  # 0.8455
    (1 + 0.8) / 2,      # 0.9
    (0.955 + 0.8) / 2,  # 0.8775
    (0.959 + 0.8) / 2,  # 0.8795
    (1 + 0.8) / 2,      # 0.9
    (0.979 + 0.8) / 2,  # 0.8895
    (0.996 + 0.8) / 2,  # 0.898
    (0.999 + 0.8) / 2,  # 0.8995
    (0.922 + 0.8) / 2,  # 0.861
    (1 + 0.8) / 2,      # 0.9
    (1 + 0.8) / 2,      # 0.9
    (1 + 0.8) / 2,      # 0.9
    (1 + 0.8) / 2,      # 0.9
    (1 + 0.8) / 2,      # 0.9
    (0.984 + 0.8) / 2,  # 0.892
    (0.999 + 0.8) / 2,  # 0.8995
    (1 + 0.8) / 2,      # 0.9
    (1 + 0.8) / 2,      # 0.9
    (0.975 + 0.8) / 2,  # 0.8875
    (1 + 0.8) / 2,      # 0.9
    (1 + 0.8) / 2,      # 0.9
    (1 + 0.8) / 2,      # 0.9
    (0.983 + 0.8) / 2,  # 0.8915
    (1 + 0.8) / 2       # 0.9
]

class LineBalancing:
    def __init__(self):
        # Initialize the lists as empty
        self.alpha_l = []
        self.beta_l = []
        self.min_time_l = []
       
    def learning_rate(self, number):
        rate = learning_rates[number-1]
        return rate
        
    def load_excel(self, file_path, sheet_name):
        print(f"Loading Excel file: {file_path}, Sheet: {sheet_name}")
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        self.alpha_l = data['alpha'].tolist()
        self.beta_l = data['beta'].tolist()
        self.min_time_l = data['min_time'].tolist()
        print("Loaded alpha:", self.alpha_l)
        print("Loaded beta:", self.beta_l)
        print("Loaded min_time:", self.min_time_l)
        return "Excel file loaded successfully."

    def balance_line(self, list_name, number):
        # Ensure the index is within bounds
        if number - 1 < 0 or number - 1 >= len(list_name):
            return "Invalid task number."
        
        newDelayTime = list_name[number - 1]
        return newDelayTime

    def get_task_operator(self, task_index):
        """Return the operator assigned to a specific task."""
        global task_to_operator
        if 0 <= task_index < len(task_to_operator):
            return task_to_operator[task_index]
        else:
            return "Invalid task index"
    
    def load_mapping_from_excel(self, file_path):
    # Load the Excel file into a DataFrame
        df = pd.read_excel(file_path)
        
        # Convert the DataFrame into a dictionary
        return dict(zip(df['ServiceBlockName'], df['WorkerPool']))
    
    def get_pool_for_service_block(self, service_block_name):
        # Load the mapping from Excel
        mapping = load_mapping_from_excel("mapping.xlsx")
        
        # Return the appropriate worker pool
        return mapping.get(service_block_name, None)
    
    def random_dist(self, number):
        newDistribution = random.choice(tasks[number-1])
        return newDistribution
    

print(tasks)
file_path = "mapping.xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path)
mapping = dict(zip(df['ServiceBlockName'], df['WorkerPool']))
# Print the DataFrame to check the content
print(mapping)
    
    #def get_pool_for_service_block(self, service_block_name):
        #mapping = {
            #"Task01": "Worker1",
            #"Task02": "Worker2",
            #"Task03": "Worker3",
        #}
        #return mapping.get(service_block_name, None)
# Task-to-Operator list as a global variable

print(task_to_operator[0])
def initialize_task_assignment():
    global task_to_operator
    # Load task times from Excel
    file_path = "ListOperatorTimes.xlsx"  # Replace with your actual file path
    sheet_name = "list"       # Replace with the correct sheet name
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # Extract operator times
    operator1_times = data['operator1'].tolist()
    operator2_times = data['operator2'].tolist()
    operator3_times = data['operator3'].tolist()

    operator_times = [operator1_times, operator2_times, operator3_times]
    num_tasks = len(operator1_times)
    num_workstations = 3

    # Assign tasks
    def assign_tasks(operator_times, num_tasks, num_workstations):
        task_to_operator_local = [""] * num_tasks
        # Initialize task assignments and logic (see previous script for full details)
        # Assign logic dynamically based on rules

        # Example assignment logic (replace with full implementation)
        for i in range(num_tasks):
            task_to_operator_local[i] = f"worker{(i % 3) + 1}"  # Example worker assignment
        return task_to_operator_local

    task_to_operator = assign_tasks(operator_times, num_tasks, num_workstations)



