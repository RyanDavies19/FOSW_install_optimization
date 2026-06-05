# Optimal Scheduling of Offshore Wind Installation: A Multi-Objective Approach
# Updated May 21, 2026

"""
Priority of work:
- [] 1. Set of weather conditions
- [] 2. More realistic task lengths, and more vessel types
- [] 3. Cost function for install
- [] 4. Sets of turbines in strings
- [] 5. Time permitting: emissions
"""

####################################
######### Initialize tools #########
####################################
using CSV
using DataFrames
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

# TODO: Set of weather conditions
weather_df = CSV.read("gulf_of_maine_weather_states_daily.csv", DataFrame)
weatherSequence = weather_df.weather_state[1:nTime]

# TODO: Sets of turbines in strings

# TODO: Set of vessels (with backup vessels that can handle worse conditions but are more expensive)

# Set of tasks
Tasks = [1, 2, 3] # 1: cable installation (disabled for now), 2: mooring installation, 3: platform hook-up
nTasks = length(Tasks)

# Set of vessels
vessels_df = CSV.read("vessel_data.csv", DataFrame)
Vessels  = vessels_df.vesselType
nVessels = length(Vessels)

###################################################
############ Define Parameters and Data ###########
###################################################

# Integer representation of inifintely high value. Useful for big M method
int_inf = typemax(Int)

# Task durations by vessel: how many timesteps each task takes at each turbine (0 disables the vessel for that task)
# TODO: make these realistic for the difference vessel types based on table 5.9
aDurations = [3, 0, 0] # durations for each task [units of timesteps]
bDurations = [0, 4, 0] # durations for each task [units of timesteps]
cDurations = [0, 0, 5] # durations for each task [units of timesteps]
taskDurations = [aDurations, bDurations, cDurations]

# Task costs: how much does it cost to do each task
aCosts = [     10, int_inf, int_inf]  # Vessel A costs per task [$]
bCosts = [int_inf,      50, int_inf]  # Vessel B costs per task [$]
cCosts = [int_inf, int_inf,      20]  # Vessel C costs per task [$]
taskCosts = [aCosts, bCosts, cCosts]

# TODO: Add in weather constraints for each vessel
# Vessel parameters
dayRates = vessels_df.dayRate
maxWind  = vessels_df.maxWind
maxWave  = vessels_df.maxWave

# Task capability matrix — 1 if vessel can do that task
taskCapable = Matrix(vessels_df[:, [:mooreInstall, :platformHookup, :cableInstall]])

# Weather state thresholds — from GEE analysis
stateWorstWind = [8.0,  16.0, Inf]  # upper bound wind for each state
stateWorstWave = [2.0,   3.0, Inf]  # upper bound wave for each state

# vesselOperability[v, s] = 1 if vessel v can operate in ALL conditions in state s
vesselOperability = zeros(Int, nVessels, 3)
for v in 1:nVessels
    for s in 1:3
        if maxWind[v] >= stateWorstWind[s] && maxWave[v] >= stateWorstWave[s]
            vesselOperability[v, s] = 1
        end
    end
end

# Check arrays and sizes
@assert (length(aDurations) == nTasks && 
        length(bDurations) == nTasks && 
        length(cDurations) == nTasks) || "ERROR: Vessel task durations and tasks must be same size"
@assert (length(aCosts) == nTasks && 
        length(bCosts) == nTasks && 
        length(cCosts) == nTasks) || "ERROR: Vessel task durations and tasks must be same size"
@assert (length(taskDurations) == nVessels &&
        length(taskCosts) == nVessels) || throw("ERROR: Costs, durations, and vessles must be same size")

# """
#     optimizer(w1)

# Arguments:
#     - `w1`: (Float) Weight on the time objective. Must satisfy 0 <= w1 <= 1.
#       The cost objective weight is set to (1 - w1).

# Returns:
#     - `ObjValue`: (Float) Optimal objective value
#     - `x_return`: (Array) Optimal schedule
# """
# function optimizer(w1) # crudely put the whole thing in a function for looping

#     ####################################
#     ########## Declare model  ##########
#     ####################################

#     m = Model(Cbc.Optimizer) 
#     # Note: if complaining of infeasibility, try increasing time

#     # Set the number of threads to use for running in Parallel (10 cores on Ryans Mac)
#     set_optimizer_attribute(m, "threads", 10)

#     ####################################
#     ######## Decision variables ########
#     ####################################

#     # x_i,k,v,t: A binary variable that equals 1 if task k at turbine i is being completed by vessel v during timestep t, and 0 otherwise.
#     @variable(m, x[1:nTurbines, 1:nTasks, 1:nVessels, 1:nTime], Bin)

#     ######################################
#     ######## Objective Function ##########
#     ######################################
#     # Weighted sum method based on Section 15.3.1 in Algorithms for Optimization - Kochenderfer and Wheeler

#     # Check valid weight
#     @assert (w1 <= 1 && w1 >= 0) || throw("ERROR: weight must be between 0 and 1")

#     # Minimize total installation time using a time penalty and cost penalty approach
#     @objective(m, Min, w1 * sum(x[i,k,v,t]*t for i = 1:nTurbines, k = 1:nTasks, v = 1:nVessels, t = 1:nTime) + 
#                 (1-w1) * sum(x[i,k,v,t]*taskCosts[v][k] for i = 1:nTurbines, k = 1:nTasks, v = 1:nVessels, t = 1:nTime))

#     ######################################
#     ############# Constraints ############
#     ######################################
#     # Some of these are inspired by Housner et al. (2026), a forthcoming paper

#     # Task Completion: sum of xi,k,t over all timesteps t must equal the required duration for task k. This also ensures each task is only completed once.
#     @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels,], sum(x[i,k,v,t] for t = 1:nTime) == taskDurations[v][k])

#     # Continuous tasks: Once a task stops, it cannot start again. tt is the time steps after t.
#     @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=2:nTime, tt=t+1:nTime], x[i,k,v,tt] <= x[i,k, v, t] - x[i,k,v,t-1] + 1)

#     # Dependency: Task k+1 cannot begin until task k is fully completed at turbine i. If task k+1 is active at t, then the sum of task k times to t-1 needs to be taskDurations.
#     @constraint(m, [i=1:nTurbines, k=1:nTasks-1, v=1:nVessels, t=1:nTime], x[i,k+1,v,t] * taskDurations[v][k] <= sum(x[i,k,v,tt] for tt=1:t-1))

#     # Resource Limit: sum of active tasks (across all turbines) cannot exceed your available vessels (e.g., you only have 1 cable-laying vessel, so only one turbine can have Task 1 active per timestep).
#     @constraint(m, [t=1:nTime, v=1:nVessels], sum(x[i,k,v,t] for i = 1:nTurbines, k = 1:nTasks) <= 1)

#     ######################################
#     ########### Print and solve ##########
#     ######################################

#     # print(m)

#     optimize!(m)

#     ObjValue = objective_value(m);

#     x_return = value(x)

#     return ObjValue, x_return
# end

# # Run function for time optimization
# objective, x = optimizer(1)

# # Make a nice looking Gantt chart w/ f-strings
# taskNames = ["Mooring Install", "Platform Hook-up", "Cable Install", ] # strings must be <16 char
# println("--- Gantt Chart ---")
# header = "Turbine |       Task       | Vessel |" * join([f" {t:2d}" for t in Time], "")
# println(header)
# println("-" ^ length(header))

# for i in 1:nTurbines
#     for k in 1:nTasks
#         # Find which vessel is assigned to this (turbine, task) pair
#         assigned_vessel = "-"
#         for v in 1:nVessels
#             if sum(round(Int, x[i, k, v, t]) for t in 1:nTime) > 0 # round is needed because value(x) returns float
#                 assigned_vessel = Vessels[v]
#                 break
#             end
#         end
#         print(f"   {i:2d}   | {taskNames[k]:16s} |   {assigned_vessel:1s}    |")
#         for t in 1:nTime
#             val = sum(round(Int, x[i, k, v, t]) for v in 1:nVessels)
#             cell = val == 1 ? "  X" : "  ."
#             print(cell)
#         end
#         print("\n")
#     end
#     println("-" ^ length(header))
# end

# # Solve for 10 weights, to get Pareto front
# nWeights = 10
# time_objectives = []
# cost_objectives = []
# for i in 1:nWeights
#     println(f"Optimizing for weight index {i}")
#     objective, x = optimizer(i / nWeights)
#     # Compute individual objectives
#     time_obj = sum(x[i,k,t]*t for i = 1:nTurbines, k = 1:nTasks, t = 1:nTime)
#     cost_obj = sum(x[i,k,t]*taskCosts[k] for i = 1:nTurbines, k = 1:nTasks, t = 1:nTime)
#     push!(time_objectives, time_obj)
#     push!(cost_objectives, cost_obj)
# end

# # Plot a Pareto front of cost vs duration
# scatter(time_objectives, cost_objectives,
#     xlabel = "Time Objective",
#     ylabel = "Cost Objective (\$)",
#     title  = "Pareto Front: Installation Time vs. Cost",
#     legend = false,
#     marker = :circle,
#     dpi = 300
# )

# savefig("pareto.png")

