"""
Optimal Scheduling of Offshore Wind Installation: A Multi-Objective Approach
Updated June 8th, 2026

References:
- Mykel J. Kochenderfer and Tim Allan Wheeler. Algorithms for Optimization. The MIT press, Cambridge, 2019. ISBN 978-0-262-03942-0.
- Forthcoming: S. Housner, R. Davies, M. Hall, Y. R. Alkarem, F. Moreno, and E. Baca, “AN OPEN-SOURCE SCHEDULING OPTIMIZATION TOOL FOR MARINE OPERATIONS,” in Proceedings of the ASME 2026 45th International Conference on Ocean, Offshore and Arctic Engineering, Tokyo, Japan, June 7-12.
- L. Amorosi, M. Fischetti, R. Paradiso, and R. Roberti, “Optimization models for the installation planning of offshore wind farms,” European Journal of Operational Research, vol. 315, no. 3, pp. 1182–1196, Jun. 2024, doi: 10.1016/j.ejor.2024.01.011.

Priority of work:
- [X] 1. Set of weather conditions
- [X] 2. More realistic task lengths and vessel costs
- [X] 3. Cost function for install (right now are fixed because they are tied to tasks and all vessels are the same for the tasks)
- [X] 4. Sets of turbines in strings
- [ ] 5. Distance between turbines and port in the cost (this will make a more meaningful patero for the costs)
- [ ] 6. Time permitting: emissions

@author: ryandavies19, kelsey-hartfelder
"""


####################################
######### Initialize tools #########
####################################

using CSV
using DataFrames
using LinearAlgebra
using JuMP
using Cbc
using PyFormattedStrings # to get python style f-string formatting
using Plots


###############################
######### Define Sets #########
###############################

# Set of turbines
maxTurbines = 4 # start with 2x2 array of turbines
Turbines = collect(1:maxTurbines)
nTurbines = length(Turbines)

# Set of turbine strings, containing turbine indicies. 
Str = [[1,3],[2,4]]
nStrs = length(Str)
@assert all(i in vcat(Str...) for i in Turbines) || throw("ERROR: all turbines must belong to a string")

# Set of timesteps.
TimeStart = 200 # RD: 0-200 has a lot of infeasible sea states
TimeEnd = 225
Time = collect(TimeStart:TimeEnd)
nTime = length(Time)

# Set of weather conditions (12-hour timesteps, based on GEE analysis of Gulf of Maine data)
weather_df = CSV.read("gulf_of_maine_weather_states_12hourly.csv", DataFrame)
weatherSequence = weather_df.weather_state[TimeStart:TimeEnd]

# Set of tasks
Tasks = ["moorInstall", "platformTow", "platformHookup", "cableInstall"] # names to match vessel csv columns
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


###################################################
############ Define Parameters and Data ###########
###################################################

# Integer representation of inifintely high value. Useful for big M method
int_inf = typemax(Int)

# # Turbine locations [m]
# TurbLoc = [[-500,500], [-500,1000], [500,500], [500,1000]] # assume turbines are 1 km apart in a grid
# @assert length(TurbLoc) == nTurbines || throw("ERROR: Mismatch in nTurbines and num Turbine Locations")

# # Port location [m]
# PortLoc = [0,0]

# # Turbine distance to port [m] - shape: (nTurbines)
# PortDist = [norm(TurbLoc[i] .- PortLoc, 2) for i=1:nTurbines] # L2 norm

# # Time to port by vessel [12hr periods] - shape: nVessels array of (nTurbines, nTurbines) objects to match TurbTime, nonzero only on diagonal
# PortTime = [diagm(ceil.(Int, PortDist / Vessels.transitSpeed[v] / 3600 / 12)) for v=1:nVessels] # Units: m * (m/s)^-1 * 1hr / 3600s * 1 time period / 12hr
# for v in eachindex(PortTime)
#     PortTime[v] .= 0
# end

# # Turbine distance to eachother [m] - shape: (nTurbines, nTurbines)
# TurbDist = [norm(TurbLoc[i] .- TurbLoc[j]) for i in 1:nTurbines, j in 1:nTurbines]

# # Turbine time to eachother [12 hr periods] - shape: nVessels array of (nTurbines, nTurbines) objects
# TurbTime = [[ceil(Int, TurbDist[i,j] ./ Vessels.transitSpeed[v] / 3600 / 12) for i=1:nTurbines, j=1:nTurbines] for v=1:nVessels]
# for v in eachindex(PortTime)
#     TurbTime[v] .= 0
# end

# Baseline task duration (at the turbine)
# TODO: make these realistic for the difference vessel types based on table 5.9
baseDurations = [2, 2, 1, 1]  # [moorInstall, platformTow, platformHookup, cableInstall]. FIXME: set platformTow = 2 for now, but should be 0 when/if distances are completed

# # taskDurations[k] = duration for task k to be completed, including transit times
# taskDurations = [] # Shape: nTasks array of nVessels arrays of (nTurbines, nTurbines) objects
# for (k, task) in enumerate(Tasks)
#     # Two types of tasks: 
#     # 1) go to and from port every time
#     # 2) stay at sea for the whole project
#     if task == "moorInstall"
#         push!(taskDurations, [baseDurations[k] .+ 2 .* PortTime[v] for v in 1:nVessels])
#     elseif task == "platformTow"
#         push!(taskDurations, [baseDurations[k] .+ 2 .* PortTime[v] for v in 1:nVessels])
#     elseif task == "platformHookup"
#         push!(taskDurations, [baseDurations[k] .+ TurbTime[v] for v in 1:nVessels])
#     elseif task == "cableInstall"
#         push!(taskDurations, [baseDurations[k] .+ 2 .* PortTime[v] for v in 1:nVessels])
#     else
#         throw("ERROR: Invalid task")
#     end
# end

# Matrix of capable vessels for tasks: If vessel can't do the task, taskCapable == 0
taskCapable = Matrix(Vessels[:, [:moorInstall, :platformTow, :platformHookup, :cableInstall]])

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

# Calculate emissions per vessel per timestep
ef_co2 = 679.47     # g/kWh, constant across tiers
V_working = 1.5     # knots, assumed on-site speed
lf_floor = 0.02     # 2% minimum 
timestep_hr = 12    # 12-hour timesteps

# Create an empty array to hold the emissions for each vessel
vesselEmissions = zeros(Float64, nVessels)

for v in 1:nVessels
    # Propulsive load factor via Propeller Law
    lf_prop = max((V_working / Vessels.serviceSpeedKN[v])^3, lf_floor)
    power_prop = Vessels.propPowerKW[v] * lf_prop

    # Auxiliary power 
    power_aux = Vessels.auxPowerKW[v] * Vessels.auxLF[v]

    # Total power
    power_total = power_prop + power_aux

    # CO2 emissions per 12-hr timestep [kg]
    vesselEmissions[v] = timestep_hr * power_total * ef_co2 / 1000
end

# Check arrays and sizes (TODO expand me)
@assert length(taskCosts) == nVessels || throw("ERROR: taskCosts must have one entry per vessel")
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
function optimizer(w1, w3; threads = 10, logLevel = 1) # put the whole thing in a function for looping

    ####################################
    ########## Declare model  ##########
    ####################################

    m = Model(Cbc.Optimizer) 
    # Note: if complaining of infeasibility, try increasing time window

    # Set the number of threads to use for running in parallel
    set_optimizer_attribute(m, "threads", threads)

    # Set the optimizer output verbosity
    set_optimizer_attribute(m, "logLevel", logLevel)


    ####################################
    ######## Decision variables ########
    ####################################

    # x_i,k,v,t: A binary variable that equals 1 if task k at turbine i is being completed by vessel v during timestep t, and 0 otherwise.
    @variable(m, x[1:nTurbines, 1:nTasks, 1:nVessels, 1:nTime], Bin) # TODO: maybe there is a way to reduce the size of X if Y has the other information

    # t_end: timestep when the last task finishes
    @variable(m, t_end >= 0, Int)

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

    # # Minimize total installation time using a time penalty and cost penalty approach
    # # where there is a cost benefit to completing strings earlier (from eqn 1a in Amorosi et al.)
    # @objective(m, Min, w1 * t_end + (1-w1) *
    #     (sum(x[i,k,v,t]*taskCosts[v][k] for i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=1:nTime))
    #      - sum(revenue * sum(z[s,t] * length(Str[s]) for t=1:nTime) for s=1:nStrs))

    
    # w2 is implicitly 1 - w1 - w3
    @assert w1 + w3 <= 1.0 "ERROR: w1 + w3 must be <= 1"
    w2 = 1 - w1 - w3

    @objective(m, Min, 
        w1 * t_end + 
        w2 * (sum(x[i,k,v,t]*taskCosts[v][k] for i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=1:nTime)
             - sum(revenue * sum(z[s,t] * length(Str[s]) for t=1:nTime) for s=1:nStrs)) +
        w3 * sum(x[i,k,v,t]*vesselEmissions[v] for i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=1:nTime))


    ######################################
    ############# Constraints ############
    ######################################

    # Some of these are inspired by Housner et al. (2026), a forthcoming paper

    # Shapes and types:
    # - x:             (nTurbines, nTasks, nVessels, nTime), binary
    # - y:             (nTurbines, nTasks, nVessels), binary
    # - z:             (nStrs, nTime), binary
    # - t_end:         scalar, Int
    # - baseDurations: nTasks array, Int
    # - taskDurations: nTasks array of nVessels arrays of (nTurbines, nTurbines) objects, Int
    # - taskCosts:     nVessels array of nTasks arrays of (scalar per vessel/task) objects, Int
    # - weather:       (nVessels, nTime), binary

    # TODO: this is complaining of infeasibility when using taskDurations[k][v][i,i], which should return just the base durations. 
    # Need some form of tracking that says which turbine was visited last, for the (i,i) indexing. Need to figure out why it is 
    # saying infeasible.

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
    
    # Set ending time: t_end must be >= the last time step any task is active. This sets t_end as the time the last task is completed.
    @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=1:nTime], t_end >= t * x[i,k,v,t])

    # String completeness: z[s,t] = 0 if all turbines in string s have not been completed: 
    #    Z * (theoretical time to complete all tasks in the string) <= (actual time taken before t for all tasks at the turbines in the string), and objective pushes z to 1 early
    @constraint(m, [s=1:nStrs, t=1:nTime], z[s,t] * sum(baseDurations) * length(Str[s]) <= sum(x[i,k,v,tt] for i in Str[s], k=1:nTasks, v=1:nVessels, tt=1:t))
    
    # String completeness: Once a string is complete, it stays complete
    @constraint(m, [s=1:nStrs, t=2:nTime], z[s,t] >= z[s,t-1])

    
    ######################################
    ########### Print and solve ##########
    ######################################

    # print(m)
    set_optimizer_attribute(m, "seconds", 120) # solver time limit

    optimize!(m)

    ObjValue = objective_value(m);

    x_return = value(x)
    z_return = value(z)
    t_end_return = value(t_end)

    emissions_return = sum(value(x[i,k,v,t])*vesselEmissions[v] for i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=1:nTime)
    return ObjValue, x_return, z_return, t_end_return, emissions_return
end

# Run function for time optimization
objective, x, z, t_end, emissions = optimizer(1, 0)      # pure time
objective, x, z, t_end, emissions = optimizer(0, 1)      # pure emissions
objective, x, z, t_end, emissions = optimizer(0.5, 0.5)  # equal weight

# Make a nice looking Gantt chart w/ f-strings
println(f"--- Gantt Chart (start time: {TimeStart}) ---")
header = "Turbine |       Task       | Vessel |" * join([f" {t:2d}" for t in 1:nTime], "") # columns are the time steps from start
println(header)
println("-" ^ length(header))
print("        |                  |        |")
for t in 1:nTime
    print(f"  {weatherSequence[t]:1d}") # 2nd row of table is weather conditions (1, 2, or 3) per time step 
end
print("\n")
println("-" ^ length(header))

for i in 1:nTurbines
    for k in 1:nTasks
        # Find which vessel is assigned to this (turbine, task) pair
        assigned_vessel = "-"
        for v in 1:nVessels
            if sum(round(Int, x[i,k,v,t]) for t in 1:nTime) > 0 # round is needed because optimizer returns float, > 0 because some tasks take longer than 1 time block
                assigned_vessel = v
                break
            end
        end
        print(f"   {i:2d}   | {Tasks[k]:16s} |   {assigned_vessel:1d}    |")
        for t in 1:nTime # at each time step:
            if round(Int, x[i,k,assigned_vessel,t]) == 1
                cell = "  O" # assigned_vessel is completing task k at turbine i
            elseif assigned_vessel != "-" && weather[assigned_vessel,t] == 0
                cell = "  X" # assigned_vessel cannot be active because of weather
            else
                cell = "  ." # assigned_vessel could be active but is not
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

# TODO: save Gantt chart to txt file, useful for when looking at Pareto fronts

# TODO: plot of cost over time

# Solve for 10 weights, to get Pareto front
nWeights = 10
w3_values = [0.0, 0.25, 0.5]
time_objectives = []
cost_objectives = []
emissions_objectives = []
w3_labels = []

for w3 in w3_values
    for ii in 0:nWeights
        w1 = (ii / nWeights) * (1 - w3)
        local objective, x, z, t_end, emissions = optimizer(w1, w3; logLevel=0)
        cost_obj = sum(x[i,k,v,t]*taskCosts[v][k] for i=1:nTurbines,
                      k=1:nTasks, v=1:nVessels, t=1:nTime) -
                   sum(revenue * sum(z[s,t] * length(Str[s]) for t=1:nTime)
                      for s=1:nStrs)
        push!(time_objectives, t_end)
        push!(cost_objectives, cost_obj)
        push!(emissions_objectives, emissions)
        push!(w3_labels, w3)
    end
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

scatter(time_objectives, cost_objectives, emissions_objectives,
        xlabel="Time", ylabel="Cost (\$)", zlabel="CO2 (kg)",
        title="3D Pareto Front",
        marker_z=float.(w3_labels),
        colorbar_title="w3",
        dpi=300)

savefig("pareto_3d.png")