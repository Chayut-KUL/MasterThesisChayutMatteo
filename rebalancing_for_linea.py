import pandas as pd
import numpy as np
from collections import defaultdict, deque
import json
from scipy.optimize import curve_fit
import pulp  # Make sure PuLP is installed

def create_precedence_graph():
    precedence = {i: [i + 1] for i in range(1, 30)}
    precedence[30] = []  # Task 30 has no successors
    return precedence

def topological_sort(precedence):
    """
    We keep this function in case you want to reference topological order, 
    though it's not strictly needed for the ILP approach.
    """
    from collections import defaultdict, deque
    in_degree = defaultdict(int)
    for tasks in precedence.values():
        for task in tasks:
            in_degree[task] += 1

    for task in precedence.keys():
        if task not in in_degree:
            in_degree[task] = 0

    queue = deque([task for task in in_degree if in_degree[task] == 0])
    sorted_tasks = []

    while queue:
        current = queue.popleft()
        sorted_tasks.append(current)
        for neighbor in precedence.get(current, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return sorted_tasks

def ilp_balance_workstations(precedence, task_times_dict, num_workstations):
    """
    Use Integer Linear Programming (via PuLP) to assign tasks to workstations
    in a way that minimizes the max station load (cycle time),
    subject to precedence constraints.

    :param precedence: dict of {task: [successors]}
    :param task_times_dict: dict of {task: time_in_seconds}
    :param num_workstations: number of workstations
    :return: (new_workstations, new_workstation_times)
             new_workstations = list of lists of tasks
             new_workstation_times = list of total times for each workstation (seconds)
    """
    tasks = list(task_times_dict.keys())

    # Create the PuLP problem
    problem = pulp.LpProblem("LineBalancing", pulp.LpMinimize)

    # Decision variables: x[i, w] = 1 if task i is assigned to workstation w
    x = pulp.LpVariable.dicts(
        "x",
        (tasks, range(num_workstations)),
        cat=pulp.LpBinary
    )

    # T is the cycle time (in seconds) we want to minimize
    T = pulp.LpVariable("T", lowBound=0, cat=pulp.LpContinuous)

    # Objective: Minimize T
    problem += T, "Minimize_cycle_time"

    # Constraint: each task is assigned exactly once
    for i in tasks:
        problem += pulp.lpSum([x[i][w] for w in range(num_workstations)]) == 1

    # Workstation load cannot exceed T
    for w in range(num_workstations):
        problem += pulp.lpSum([task_times_dict[i] * x[i][w] for i in tasks]) <= T

    # Precedence constraints: if i -> j, then station(j) >= station(i).
    station_index = {}
    for i in tasks:
        # station_index[i] = sum_{w} (w * x[i][w])
        station_index[i] = pulp.lpSum([w * x[i][w] for w in range(num_workstations)])

    for i, successors in precedence.items():
        for j in successors:
            problem += station_index[j] >= station_index[i]

    # Solve the ILP
    problem.solve(pulp.PULP_CBC_CMD(msg=0))  # msg=0 hides solver output

    # Extract the solution
    assignment = {i: None for i in tasks}
    for i in tasks:
        for w in range(num_workstations):
            if pulp.value(x[i][w]) == 1:
                assignment[i] = w
                break

    # Build new_workstations: list of lists
    new_workstations = [[] for _ in range(num_workstations)]
    for i in tasks:
        w_assigned = assignment[i]
        new_workstations[w_assigned].append(i)

    # Compute total time (seconds) for each workstation
    new_workstation_times = []
    for w in range(num_workstations):
        ws_time = sum(task_times_dict[i] for i in new_workstations[w])
        new_workstation_times.append(ws_time)

    return new_workstations, new_workstation_times

def calculate_average_task_times(file_path):
    """
    Reads Excel, filters to only tasks Task01..Task30 with 'activity type' == 'WORK',
    then for each task, takes the latest 100% (all) data points (based on start date descending),
    and returns the average duration in SECONDS.
    """
    df = pd.read_excel(file_path)
    df['start date'] = pd.to_datetime(df['start date'])
    df['stop date'] = pd.to_datetime(df['stop date'])
    df['duration (s)'] = (df['stop date'] - df['start date']).dt.total_seconds()

    # Only keep rows with block in Task01..Task30 AND activity type == WORK
    task_filter = [f'Task{i:02d}' for i in range(1, 31)]
    df = df[(df['block'].isin(task_filter)) & (df['activity type'] == 'WORK')]

    average_times = []

    for task in task_filter:
        # Sort by start date descending and take 20% of data points
        task_data = df[df['block'] == task].sort_values(by='start date', ascending=False)
        num_records = len(task_data)
        subset_size = int(np.ceil(0.2 * num_records))  # 20%
        task_data = task_data.head(subset_size)

        if len(task_data) > 0:
            avg_duration = task_data['duration (s)'].mean()  # in seconds
        else:
            avg_duration = 0

        average_times.append(avg_duration)

    return average_times  # all in seconds

def calculate_learning_rates(file_path):
    """
    Reads Excel, keeps only tasks that start with 'Task' and 'activity type' == 'WORK',
    fits a power curve f(x) = a * x^b to each task's durations,
    returns a dict { 'Task01': b1, 'Task02': b2, ... } for the exponent b.
    """
    df = pd.read_excel(file_path, sheet_name="flowchart_process_states_log")
    df['start date'] = pd.to_datetime(df['start date'])
    df['stop date'] = pd.to_datetime(df['stop date'])
    df['duration (s)'] = (df['stop date'] - df['start date']).dt.total_seconds()

    # Only tasks named "Task..." with WORK
    df = df[df['block'].str.startswith('Task') & (df['activity type'] == 'WORK')]
    df = df.sort_values(by=['block', 'start date']).reset_index(drop=True)

    tasks = df['block'].unique()
    learning_rates = {}

    def power_curve(x, a, b):
        return a * np.power(x, b)

    for task in tasks:
        task_data = df[df['block'] == task].reset_index(drop=True)
        x = np.arange(1, len(task_data) + 1)
        y = task_data['duration (s)']

        if len(x) > 1:
            # Fit curve a*x^b
            params, _ = curve_fit(power_curve, x, y, maxfev=5000)
            b = params[1]  # exponent
            learning_rates[task] = b
        else:
            learning_rates[task] = 0  # no learning rate if insufficient data

    return learning_rates

def compare_scenarios(current_task_times, old_workstations, new_workstations, learning_rates, t_1):
    """
    Compares old vs. new scenarios using the same 'current_task_times' (in seconds).
    t_1 is also in seconds. Setup cost is 60 seconds per task moved (since we do * 60 in code).

    Returns a dict with cycle times, penalty, and time to net benefit.
    """
    def calculate_cycle_time(workstations, task_times):
        return [
            sum(task_times[task - 1] for task in workstation)
            for workstation in workstations
        ]

    old_cycle_times = calculate_cycle_time(old_workstations, current_task_times)
    new_cycle_times = calculate_cycle_time(new_workstations, current_task_times)

    old_total_time = max(old_cycle_times)
    new_total_time = max(new_cycle_times)

    old_task_to_workstation = {
        task: i + 1 for i, workstation in enumerate(old_workstations) for task in workstation
    }
    new_task_to_workstation = {
        task: i + 1 for i, workstation in enumerate(new_workstations) for task in workstation
    }

    task_changes = {i + 1: {"removed": [], "added": []} for i in range(len(old_workstations))}
    for task, new_station in new_task_to_workstation.items():
        old_station = old_task_to_workstation.get(task, None)
        if old_station is not None and old_station != new_station:
            task_changes[old_station]["removed"].append(task)
            task_changes[new_station]["added"].append(task)

    time_saved_per_cycle = old_total_time - new_total_time
    cycles_per_hour = 3600 / new_total_time if new_total_time > 0 else 0
    time_saved_per_hour = time_saved_per_cycle * cycles_per_hour  # seconds/hour

    # Setup cost: 60 seconds per moved task here
    total_task_movements = sum(
        len(changes['removed']) + len(changes['added'])
        for changes in task_changes.values()
    ) // 2
    setup_time_cost = total_task_movements * 60  # 60 seconds each

    # Learning penalty: sum of t1*(i^b) for i=1..10, for each moved task
    learning_penalty = 0
    for task, new_station in new_task_to_workstation.items():
        old_station = old_task_to_workstation.get(task, None)
        if old_station is not None and old_station != new_station:
            task_key = f'Task{task:02d}'
            b = learning_rates.get(task_key, 0)
            t1 = t_1[task - 1]  # in seconds
            learning_penalty += sum(t1 * (i ** b) for i in range(1, 11))

    total_penalty = setup_time_cost + learning_penalty

    if time_saved_per_hour > 0:
        time_to_net_benefit = total_penalty / time_saved_per_hour
    else:
        time_to_net_benefit = float('inf')

    return {
        "old_cycle_times": old_cycle_times,
        "new_cycle_times": new_cycle_times,
        "old_total_time": old_total_time,
        "new_total_time": new_total_time,
        "task_changes": task_changes,
        "time_saved_per_hour": time_saved_per_hour,
        "setup_time_cost": setup_time_cost,
        "learning_penalty": learning_penalty,
        "time_to_net_benefit": time_to_net_benefit
    }

def save_state_to_file(state, filename="previous_state.json"):
    with open(filename, "w") as f:
        json.dump(state, f, indent=4)

def load_state_from_file(filename="previous_state.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("No previous state file found. Starting fresh.")
        return None

def main():
    precedence = create_precedence_graph()

    file_path = "process_state_log.xlsx"

    # t_1 is in SECONDS
    t_1 = [
        2.33, 2.38, 4.47, 4.69, 4.83, 7.76, 9.27, 4.47, 4.69, 10.22,
        9.07, 9.83, 9.89, 5.54, 5.60, 3.46, 17.72, 13.83, 14.05, 11.97,
        14.40, 6.80, 5.21, 4.22, 9.39, 14.62, 19.71, 7.06, 6.47, 5.19
    ]

    print("Calculating average task times from ALL available data points (in seconds)...")
    avg_times_list = calculate_average_task_times(file_path)

    # -----------------------------------------------------------------
    # NEW: Print each task's calculated average time
    # -----------------------------------------------------------------
    print("\n--- Average Times per Task (Seconds) ---")
    for i, avg_time in enumerate(avg_times_list, start=1):
        print(f"Task{i:02d}: {avg_time:.2f} sec")

    print("\nCalculating learning rates (exponent b) for each TaskXX...")
    learning_rates = calculate_learning_rates(file_path)

    task_times_dict = {}
    for i, val in enumerate(avg_times_list, start=1):
        task_times_dict[i] = val  # keep in seconds

    num_workstations = 3

    print("\nLoading previous state (if any)...")
    previous_state = load_state_from_file()

    print("Performing ILP-based balancing (all times in seconds)...")
    new_workstations, new_workstation_times = ilp_balance_workstations(
        precedence,
        task_times_dict,
        num_workstations
    )

    # We'll pass the same list of times (seconds) for comparison
    current_task_times = avg_times_list

    if previous_state:
        print("Comparing current balancing scenarios...")
        comparison = compare_scenarios(
            current_task_times,
            previous_state['workstations'],
            new_workstations,
            learning_rates,
            t_1
        )

        print("\n--- Comparison Results ---")
        print("Old Cycle Times (by workstation) [seconds]:",
              [float(round(x, 2)) for x in comparison['old_cycle_times']])
        print("New Cycle Times (by workstation) [seconds]:",
              [float(round(x, 2)) for x in comparison['new_cycle_times']])
        print(f"Old Total Time (Bottleneck) [seconds]: {comparison['old_total_time']:.2f}")
        print(f"New Total Time (Bottleneck) [seconds]: {comparison['new_total_time']:.2f}")

        print("\n--- Task Changes ---")
        any_changes = False
        for ws, changes in comparison['task_changes'].items():
            if changes['removed'] or changes['added']:
                any_changes = True
                print(f"Workstation {ws}:")
                if changes['removed']:
                    print("  Removed tasks:", changes['removed'])
                if changes['added']:
                    print("  Added tasks:", changes['added'])
        if not any_changes:
            print("No tasks moved between workstations.")

        print("\n--- Metrics ---")
        print(f"Time Saved per Hour [seconds/hour]: {comparison['time_saved_per_hour']:.2f}")
        print(f"Setup Time Cost [seconds]: {comparison['setup_time_cost']:.2f}")
        print(f"Learning Penalty [seconds]: {comparison['learning_penalty']:.2f}")

        if np.isinf(comparison['time_to_net_benefit']):
            print("Time to Net Benefit: Never (no net benefit)")
        else:
            print(f"Time to Net Benefit [hours]: {comparison['time_to_net_benefit']:.2f}")

        print("\n--- New Scenario Workstation Allocation ---")
        for i, ws in enumerate(new_workstations, start=1):
            print(f"  Workstation {i}: {ws}")

        print("\nSaving the new scenario as the current baseline...")
        save_state_to_file({"workstations": new_workstations})

    else:
        # No previous scenario, so we just print the new scenario details
        print("No previous scenario found. Using the new scenario as the baseline.")

        new_cycle_times = [float(round(x, 2)) for x in new_workstation_times]
        new_total_time = max(new_cycle_times) if new_cycle_times else 0

        print("\n--- New Scenario (Baseline) ---")
        print("New Workstations Configuration:")
        for i, ws in enumerate(new_workstations, start=1):
            print(f"  Workstation {i}: {ws}")
        print("New Cycle Times (by workstation) [seconds]:", new_cycle_times)
        print(f"New Total Time (Bottleneck) [seconds]: {new_total_time:.2f}")

        print("\nSaving the new scenario as the current baseline...")
        save_state_to_file({"workstations": new_workstations})


if __name__ == "__main__":
    main()

