# Optimal Scheduling of Offshore Wind Installation: A Multi-Objective Approach
# Updated May 21, 2026

"""
Priority of work:
- [X] 1. Set of weather conditions
- [X] 2. More realistic task lengths and vessel costs
- [X] 3. Cost function for install (right now are fixed because they are tied to tasks and all vessels are the same for the tasks)
- [X] 4. Sets of turbines in strings
- [ ] 5. Distance between turbines and port in the cost (this will make a more meaningful patero for the costs)
- [ ] 6. Time permitting: emissions
"""

####################################
######### Initialize tools #########
####################################
using CSV
using DataFrames
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

# Set of strings, containing turbine indicies. 
# RD: once a string is completed, it adds negative cost. This encourages the model to install the turbines in a string together.
Str = [[1,3],[2,4]]
nStrs = length(Str)
@assert all(i in vcat(Str...) for i in Turbines) || throw("ERROR: all turbines must belong to a string")

# Set of timesteps
TimeStart = 200 # RD: 0-200 has a lot of infeasible seastates
TimeEnd = 215
Time = collect(TimeStart:TimeEnd)
nTime = length(Time)

# Set of weather conditions (12-hour timesteps, based on GEE analysis of Gulf of Maine data)
weather_df = CSV.read("gulf_of_maine_weather_states_12hourly.csv", DataFrame)
weatherSequence = weather_df.weather_state[TimeStart:TimeEnd]

# Set of tasks
Tasks = [1, 2, 3] # 1: cable installation, 2: mooring installation, 3: platform hook-up 
# TODO: error check to make sure data columns match tasks. Cable install should happen first.
nTasks = length(Tasks)

# Set of vessel types
VesselTypes = CSV.read("vessel_data.csv", DataFrame)
nVesselTypes = nrow(VesselTypes)

# Set of vessels (repeats each VesselType row numVessel times)
rows = Int[]
for type in 1:nVesselTypes
    for num in 1:VesselTypes.numVessel[type]
        push!(rows, type)
    end
end
Vessels = VesselTypes[rows, :]
nVessels = nrow(Vessels)

# TODO: Add backup vessels that can handle worse conditions but are more expensive

###################################################
############ Define Parameters and Data ###########
###################################################

# Integer representation of inifintely high value. Useful for big M method
int_inf = typemax(Int)

# TODO: locations of the turbines (in x and y) and the location of the port

# KH NOTE: Updated task durations & task remove hardcoding & pull in CSV vessel data.

# Task durations by vessel type: how many timesteps each task takes at each turbine (0 disables the vessel for that task)
# TODO: make these realistic for the difference vessel types based on table 5.9
baseDurations = [2, 2, 1]  # [moorInstall, platformHookup, cableInstall]

# taskDurations[v][k] = duration for vessel v doing task k
# If vessel can't do the task (taskCapable == 0), duration is 0
taskCapable = Matrix(Vessels[:, [:moorInstall, :platformHookup, :cableInstall]])
taskDurations = [taskCapable[v, :] .* baseDurations for v in 1:nVessels]

# Task costs: how much does it cost to do each task
# taskCosts[v][k] = cost for vessel v doing task k per timestep
# If vessel can't do the task, cost is int_inf
# Divide day rate by 2 because timesteps are 12-hour intervals, not full days
dayRates = Vessels.dayRate
taskCosts = [
    [taskCapable[v, k] == 1 ? dayRates[v] / 2 : int_inf 
     for k in 1:nTasks]
    for v in 1:nVessels
]

# Revenue per turbine once online
revenue = 25000 # [$/time step] # TODO: make this a more real number

# Add in weather constraints for each vessel type
maxWind  = Vessels.maxWind
maxWave  = Vessels.maxWave

# Weather state thresholds (three weather states) — from GEE analysis
stateWorstWind = [10.0, 20.0, Inf]  # upper bound wind for each state
stateWorstWave = [ 2.0,  3.0, Inf]  # upper bound wave for each state

# vesselOperability[v, s] = 1 if vessel v can operate in ALL conditions in state s
vesselOperability = zeros(Int, nVessels, 3)
for v in 1:nVessels
    for s in 1:3
        if maxWind[v] >= stateWorstWind[s] && maxWave[v] >= stateWorstWave[s]
            vesselOperability[v, s] = 1
        end
    end
end

# Create a weather operability matrix for each vessel and time step, based on the weather sequence and vessel operability by state
weather = [vesselOperability[v, weatherSequence[t]] for v in 1:nVessels, t in 1:nTime]

# Check arrays and sizes
@assert length(taskDurations) == nVessels || throw("ERROR: taskDurations must have one entry per vessel")
@assert length(taskCosts) == nVessels || throw("ERROR: taskCosts must have one entry per vessel")
@assert all(length(taskDurations[v]) == nTasks for v in 1:nVessels) || throw("ERROR: each vessel must have one duration per task")
@assert all(length(taskCosts[v]) == nTasks for v in 1:nVessels) || throw("ERROR: each vessel must have one cost per task")

"""
    optimizer(w1)

Arguments:
    - `w1`: (Float) Weight on the time objective. Must satisfy 0 <= w1 <= 1.
      The cost objective weight is set to (1 - w1).

Returns:
    - `ObjValue`: (Float) Optimal objective value
    - `x_return`: (Array) Optimal schedule
"""
function optimizer(w1; threads = 10, logLevel = 1) # put the whole thing in a function for looping

    ####################################
    ########## Declare model  ##########
    ####################################

    m = Model(Cbc.Optimizer) 
    # Note: if complaining of infeasibility, try increasing time

    # Set the number of threads to use for running in Parallel (10 cores on Ryans Mac)
    set_optimizer_attribute(m, "threads", threads)

    # Set the optimizer output verbosity
    set_optimizer_attribute(m, "logLevel", logLevel)

    ####################################
    ######## Decision variables ########
    ####################################

    # x_i,k,v,t: A binary variable that equals 1 if task k at turbine i is being completed by vessel v during timestep t, and 0 otherwise.
    @variable(m, x[1:nTurbines, 1:nTasks, 1:nVessels, 1:nTime], Bin) # TODO: maybe there is a way to reduce the size of X if Y has the other information

    # t_end: timestep when the last task finishes
    @variable(m, t_end >= 0)

    # y_i,k,v: A binary variable that tracks if vessel v is assigned to task k at turbine i
    @variable(m, y[1:nTurbines, 1:nTasks, 1:nVessels], Bin)

    # z: A binary variable that tracks if a string is complete at time t
    @variable(m, z[1:nStrs, 1:nTime], Bin)

    ######################################
    ######## Objective Function ##########
    ######################################
    # Weighted sum method based on Section 15.3.1 in Algorithms for Optimization - Kochenderfer and Wheeler

    # Check valid weight
    @assert (w1 <= 1 && w1 >= 0) || throw("ERROR: weight must be between 0 and 1")

    # Minimize total installation time using a time penalty and cost penalty approach
    # where there is a cost benefit to completing strings earlier (from eqn 1a in Amorosi et al.)
    @objective(m, Min, w1 * t_end + (1-w1) *
        (sum(x[i,k,v,t]*taskCosts[v][k] for i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=1:nTime))
         - sum(revenue * sum(z[s,t] * length(Str[s]) for t=1:nTime) for s=1:nStrs))

    # TODO: add a positive cost for when a strings of turbines is completed (Amorosi et al.). Maybe use @NLexpression?
    
    # TODO: transit distances as part of the time/costs

    ######################################
    ############# Constraints ############
    ######################################
    # Some of these are inspired by Housner et al. (2026), a forthcoming paper

    # Task completion: The total duration spent on a task at a turbine across all vessels must be the duration of the task. This also ensures each task is only completed once.
    @constraint(m, [i=1:nTurbines, k=1:nTasks], sum(x[i,k,v,t] for v=1:nVessels, t=1:nTime) == baseDurations[k])

    # Only valid vessels: Incapable vessels do no work on tasks they cannot perform (necessary for the time objective, because cost is ignored)
    @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels], sum(x[i,k,v,t] for t=1:nTime) <= taskCapable[v,k] * baseDurations[k])

    # Continuous tasks: Once a task stops, it cannot start again. tt is the time steps after t.
    @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=2:nTime, tt=t+1:nTime], x[i,k,v,tt] <= x[i,k, v, t] - x[i,k,v,t-1] + 1)

    # Consistent tasks: exactly one vessel is assigned to each (turbine, task) pair
    @constraint(m, [i=1:nTurbines, k=1:nTasks], sum(y[i,k,v] for v=1:nVessels) == 1)

    # Vessel usage: A vessel can only do task k at turbine i if it is assigned to it
    @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels], sum(x[i,k,v,t] for t=1:nTime) <= baseDurations[k] * y[i,k,v])

    # Dependency: Task k+1 cannot begin until task k is fully completed at turbine i. If task k+1 is active at t, then the sum of task k times to t-1 across all vessels needs to be baseDurations (i.e. any vessel doing task k needs to be finished).
    @constraint(m, [i=1:nTurbines, k=1:nTasks-1, v=1:nVessels, t=1:nTime], x[i,k+1,v,t] * baseDurations[k] <= sum(x[i,k,vv,tt] for vv=1:nVessels, tt=1:t-1))

    # Resource limit: sum of active tasks (across all turbines) cannot exceed your available vessels (e.g., you only have 1 cable-laying vessel, so only one turbine can have Task 1 active per timestep).
    @constraint(m, [t=1:nTime, v=1:nVessels], sum(x[i,k,v,t] for i = 1:nTurbines, k = 1:nTasks) <= 1)

    # Weather constraints: if weather conditions at time t prevent vessel v from operating, then x[i,k,v,t] must be 0 for all i and k. We can use the weather operability matrix we created above to enforce this.
    @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=1:nTime], x[i,k,v,t] <= weather[v,t])
    
    # Set ending time: t_end must be >= the last timestep any task is active. This sets t_end as the time the last task is completed.
    @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=1:nTime], t_end >= t * x[i,k,v,t])

    # String completeness: z[s,t] = 1 if all turbines in string s are complete by time t: 
    #    Z * (time to complete all tasks in the string) <= (time taken before t for all tasks at the turbines in the string), and objective pushes z to 1 early
    @constraint(m, [s=1:nStrs, t=1:nTime], z[s,t] * sum(baseDurations) * length(Str[s]) <= sum(x[i,k,v,tt] for i in Str[s], k=1:nTasks, v=1:nVessels, tt=1:t))
    
    # String completeness: Once a string is complete, it stays complete
    @constraint(m, [s=1:nStrs, t=2:nTime], z[s,t] >= z[s,t-1])

    ######################################
    ########### Print and solve ##########
    ######################################

    # print(m)
    set_optimizer_attribute(m, "seconds", 120)

    optimize!(m)

    ObjValue = objective_value(m);

    x_return = value(x)
    t_end_return = value(t_end)

    return ObjValue, x_return, t_end_return
end

# Run function for time optimization
objective, x, t_end = optimizer(1)

# Make a nice looking Gantt chart w/ f-strings
taskNames = ["Mooring Install", "Platform Hook-up", "Cable Install", ] # strings must be <16 char
println(f"--- Gantt Chart (start time: {TimeStart}) ---")
header = "Turbine |       Task       | Vessel |" * join([f" {t:2d}" for t in 1:nTime], "") # columns are the time steps from start
println(header)
println("-" ^ length(header))
print("        |                  |        |")
for t in 1:nTime
    print(f"  {weatherSequence[t]:1d}") # 2nd row of table is weather conditions
end
print("\n")
println("-" ^ length(header))

for i in 1:nTurbines
    for k in 1:nTasks
        # Find which vessel is assigned to this (turbine, task) pair
        assigned_vessel = "-"
        for v in 1:nVessels
            if sum(round(Int, x[i,k,v,t]) for t in 1:nTime) > 0 # round is needed because optimizer returns float
                assigned_vessel = v
                break
            end
        end
        print(f"   {i:2d}   | {taskNames[k]:16s} |   {assigned_vessel:1s}    |")
        for t in 1:nTime
            cell = if round(Int, x[i,k,assigned_vessel,t]) == 1
                "  O"
            elseif assigned_vessel != "-" && weather[assigned_vessel,t] == 0
                "  X"
            else
                "  ."
            end
            print(cell)
        end
        print("\n")
    end
    println("-" ^ length(header))
end
println("Legend:")
println("X - Task is inactive due to weather")
println("O - Task is active")
println(". - Task could be active but is not")
println("-----------------------------------")


# Solve for 10 weights, to get Pareto front
nWeights = 10
time_objectives = []
cost_objectives = []
for i in 1:nWeights
    println(f"Optimizing for weight index {i}")
    local objective, x, t_end = optimizer(i / nWeights; logLevel=0)
    cost_obj = sum(x[ii,k,v,t]*taskCosts[v][k] for ii=1:nTurbines, k=1:nTasks, v=1:nVessels, t=1:nTime)
    push!(time_objectives, t_end)
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