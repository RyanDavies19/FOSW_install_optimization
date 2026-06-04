# Optimal Scheduling of Offshore Wind Installation: A Multi-Objective Approach
# Updated May 21, 2026

####################################
######### Initialize tools #########
####################################
using JuMP
using Cbc
using PyFormattedStrings # to get python style f-string formatting
using Plots


"""
    optimizer(w1, w2)

Arguments:
    - `w1`: (Float) Time objective weight. 
    Must be 0 < w < 1. w2 = 1 - w1

Returns:
    - `ObjValue`: (Float) Optimal objective value
    - `x_return`: (Array) Optimal schedule
    - `Time`: (Array) The planning horizon
"""

###############################
######### Define Sets #########
###############################

# Set of turbines
maxTurbines = 4 # start with 2x2 array of turbines
Turbines = collect(1:maxTurbines)
nTurbines = length(Turbines)

# Set of timesteps
TimeStart = 1
TimeEnd = 30
Time = collect(TimeStart:TimeEnd)
nTime = length(Time)

# Set of tasks
Tasks = [1, 2, 3] # 1: cable installation (disabled for now), 2: mooring installation, 3: platform hook-up
nTasks = length(Tasks)

###################################################
############ Define Parameters and Data ###########
###################################################

# Task durations: how many timesteps each task takes at each turbine
taskDurations = [3, 4, 5] # example durations for each task [units of timesteps]

# Task costs: how much does it cost to do each task
taskCosts = [10, 50, 20]  # [$]

# Check arrays and sizes
@assert length(taskDurations) == nTasks || throw("ERROR: Durations and Task must be same size")


"""
    optimizer(w1)

Arguments:
    - `w1`: (Float) Weight on the time objective. Must satisfy 0 <= w1 <= 1.
      The cost objective weight is set to (1 - w1).

Returns:
    - `ObjValue`: (Float) Optimal objective value
    - `x_return`: (Array) Optimal schedule
"""
function optimizer(w1) # crudely put the whole thing in a function for looping

    ####################################
    ########## Declare model  ##########
    ####################################

    m = Model(Cbc.Optimizer) 
    # Note: if complaining of infeasibility, try increasing time

    # Set the number of threads to use for running in Parallel (10 cores on Ryans Mac)
    set_optimizer_attribute(m, "threads", 10)

    ####################################
    ######## Decision variables ########
    ####################################

    # x_i,k,t: A binary variable that equals 1 if task k at turbine i is active during timestep t, and 0 otherwise.
    @variable(m, x[1:nTurbines, 1:nTasks, 1:nTime], Bin)

    ######################################
    ######## Objective Function ##########
    ######################################
    # Weighted sum method based on Section 15.3.1 in Algorithms for Optimization - Kochenderfer and Wheeler

    # Check valid weight
    @assert (w1 <= 1 && w1 >= 0) || throw("ERROR: weight must be between 0 and 1")

    # Minimize total installation time using a time penalty approach
    @objective(m, Min, w1 * sum(x[i,k,t]*t for i = 1:nTurbines, k = 1:nTasks, t = 1:nTime) + (1-w1) * sum(x[i,k,t]*taskCosts[k] for i = 1:nTurbines, k = 1:nTasks, t = 1:nTime))

    ######################################
    ############# Constraints ############
    ######################################
    # Some of these are inspired by Housner et al. (2026), a forthcoming paper

    # Task Completion: sum of xi,k,t over all timesteps t must equal the required duration for task k. This also ensures each task is only completed once.
    @constraint(m, [i=1:nTurbines, k=1:nTasks], sum(x[i,k,t] for t = 1:nTime) == taskDurations[k])

    # Continuous tasks: Once a task stops, it cannot start again. tt is the time steps after t.
    @constraint(m, [i=1:nTurbines, k=1:nTasks, t=2:nTime, tt=t+1:nTime], x[i,k,tt] <= x[i,k,t] - x[i,k,t-1] + 1)

    # Dependency: Task k+1 cannot begin until task k is fully completed at turbine i. If task k+1 is active at t, then the sum of task k times to t-1 needs to be taskDurations.
    @constraint(m, [i=1:nTurbines, k=1:nTasks-1,  t=1:nTime], x[i,k+1,t] * taskDurations[k] <= sum(x[i,k,tt] for tt=1:t-1))

    # Resource Limit: sum of active tasks cannot exceed your available vessels (e.g., you only have 1 cable-laying vessel, so only one turbine can have Task 1 active per timestep).
    @constraint(m, [t=1:nTime, k=1:nTasks],sum(x[i,k,t] for i = 1:nTurbines) <= 1)

    ######################################
    ########### Print and solve ##########
    ######################################

    # print(m)

    optimize!(m)

    ObjValue = objective_value(m);

    x_return = value(x)

    return ObjValue, x_return
end

# Run function for time optimization
objective, x = optimizer(1)

# Make a nice looking Gantt chart w/ f-strings
taskNames = ["Mooring Install", "Platform Hook-up", "Cable Install", ] # strings must be <16 char
println("--- Gantt Chart ---")
header = "Turbine |       Task       |" * join([f" {t:2d}" for t in Time], "")
println(header)
println("-" ^ length(header))

for i in 1:nTurbines
    for k in 1:nTasks
        print(f"   {i:2d}   | {taskNames[k]:16s} |")
        for t in 1:nTime
            val = round(Int, x[i, k, t])
            cell = val == 1 ? "  X" : "  ."
            print(cell)
        end
        print("\n")
    end
    println("-" ^ length(header))
end

# Solve for 10 weights, to get Pareto front
nWeights = 10
time_objectives = []
cost_objectives = []
for i in 1:nWeights
    println(f"Optimizing for weight index {i}")
    objective, x = optimizer(i / nWeights)
    # Compute individual objectives
    time_obj = sum(x[i,k,t]*t for i = 1:nTurbines, k = 1:nTasks, t = 1:nTime)
    cost_obj = sum(x[i,k,t]*taskCosts[k] for i = 1:nTurbines, k = 1:nTasks, t = 1:nTime)
    push!(time_objectives, time_obj)
    push!(cost_objectives, cost_obj)
end

# Plot a Pareto front of cost vs duration
scatter(time_objectives, cost_objectives,
    xlabel = "Time Objective",
    ylabel = "Cost Objective (\$)",
    title  = "Pareto Front: Installation Time vs. Cost",
    legend = false,
    marker = :circle,
    dpi = 300
)

savefig("pareto.png")

