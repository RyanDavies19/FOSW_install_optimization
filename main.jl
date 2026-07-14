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
- [ ] 5. Distance between turbines and port in the cost (this will make a more meaningful pareto for the costs)
- [X] 6. Time permitting: emissions

@author: ryandavies19, kelsey-hartfelder, Claude (Sonnet 5)
"""


####################################
######### Initialize tools #########
####################################

using CSV
using DataFrames
using LinearAlgebra
using JuMP
using HiGHS
using PyFormattedStrings # to get python style f-string formatting
using Plots


###############################
######### Define Sets #########
###############################

# Set of turbines
maxTurbines = 8 # start with 2x2 array of turbines
Turbines = collect(1:maxTurbines)
nTurbines = length(Turbines)

# Set of turbine strings, containing turbine indicies
Str = [[1,3,5,7],[2,4,6,8]]
nStrs = length(Str)
@assert sort(vcat(Str...)) == Turbines || throw("ERROR: turbine strings must partition the turbine set")

# Set of timesteps.
TimeStart = 200 # RD: 0-200 has a lot of infeasible sea states
TimeEnd = 230
Time = collect(TimeStart:TimeEnd)
nTime = length(Time)

# Set of weather conditions (12-hour timesteps, based on GEE analysis of Gulf of Maine data)
weather_df = CSV.read("Data/gulf_of_maine_weather_states_12hourly.csv", DataFrame)
weatherSequence = weather_df.weather_state[TimeStart:TimeEnd]

# Set of tasks
Tasks = ["moorInstall", "platformTow", "platformHookup", "cableInstall"] # names to match vessel csv columns
nTasks = length(Tasks)

# Set of vessel types
VesselTypes = CSV.read("Data/vessel_data.csv", DataFrame)
nVesselTypes = nrow(VesselTypes)

# Set of vessels (repeats each VesselType row numVessel times)
rows = Int[]
for type in 1:nVesselTypes
    for num in 1:VesselTypes.numVessel[type]
        push!(rows, type)
    end
end
Vessels = VesselTypes[rows, :] # one row per vessel instance (may have duplicate types)
nVessels = nrow(Vessels)


###################################################
############ Define Parameters and Data ###########
###################################################

# Integer representation of inifintely high value. Useful for big M method
int_inf = typemax(Int)

## UNFINISHED: Port and Turbine Locations
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
baseDurations = [2, 2, 2, 1]  # [moorInstall, platformTow, platformHookup, cableInstall]. FIXME: set platformTow = 2 for now, but should be 0 when/if distances are completed

## UNFINISHED: Overall Task Durations
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
# If vessel can't do the task, set cost to 0 (decision var is forced 0 by the "Only valid vessels" 
# constraint; no need for a Big-M placeholder - Int_inf triggered HiGHS numerical warnings)
# Divide day rate by 2 because timesteps are 12-hour intervals, not full days
dayRates = Vessels.dayRate
taskCosts = [
    [taskCapable[v, k] == 1 ? dayRates[v] / 2 : 0
     for k in 1:nTasks]
    for v in 1:nVessels
]

# Revenue per turbine once online 
revenue = 9859.5 * 10^3 / 132 / 365 / 2 * 22.62 # (AEP in GWh) * (GWh / MWh) / (132 turbines) / (365 days/yr) / (2 periods / day) * (22.62 $/MWh) = $2314.47 per timestep

# Add in weather constraints for each vessel type
maxWind  = Vessels.maxWind
maxWave  = Vessels.maxWave

# Weather state thresholds (three weather states) — from GEE analysis
stateWorstWind = [10.0, 20.0, Inf]  # upper bound wind for each category [m/s]
stateWorstWave = [ 2.0,  3.0, Inf]  # upper bound wave for each category [m]

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
ef_co2 = 679.47     # g/kWh
V_working = 1.5     # knots
lf_floor = 0.02     # 2% min
timestep_hr = 12    # 12-hour timesteps

# Create empty array to hold the emissions for each vessel
vesselEmissions = zeros(Float64, nVessels)

for v in 1:nVessels
    # Propulsive power consumption
    lf_prop = max((V_working / Vessels.serviceSpeedKN[v])^3, lf_floor)
    power_prop = Vessels.propPowerKW[v] * lf_prop

    # Auxiliary power consumption
    power_aux = Vessels.auxPowerKW[v] * Vessels.auxLF[v]

    # Total power
    power_total = power_prop + power_aux

    # CO2 emissions per 12-hr timestep [kg]
    vesselEmissions[v] = timestep_hr * power_total * ef_co2 / 1000
end

# Check arrays and sizes
@assert length(taskCosts) == nVessels || throw("ERROR: taskCosts must have one entry per vessel")
@assert all(length(taskCosts[v]) == nTasks for v in 1:nVessels) || throw("ERROR: each vessel must have one cost per task")

    
"""
    optimizer(w1, w3; threads=10, logLevel=1)

Parameters:
-----------
    - `w1`: (Float) Weight on the time objective. 
    - `w3`: (Float) Weight on the emissions objective.
      The cost objective weight `w2` is set to `1 - w1 - w3`.
    - `threads`: (Int) Number of threads for the solver
    - `logLevel`: (Int) Verbosity of solver output.

Returns:
--------
    - `ObjValue`: (Float) Optimal objective value
    - `x_return`: (Array) Optimal schedule
    - `z_return`: (Array) String completion status
    - `t_end_return`: (Int) Timestep at which the last task is completed
    - `emissions_return`:(Float) Total CO2 emissions [kg] across all vessels and timesteps
"""
function optimizer(w1, w3; threads = 10, logLevel = 1) # put the whole thing in a function for looping

    ####################################
    ########## Declare model  ##########
    ####################################

    m = Model(HiGHS.Optimizer) 
    # Note: if complaining of infeasibility, try increasing time window

    # Set the number of threads to use for running in parallel
    set_optimizer_attribute(m, "threads", threads)
    set_optimizer_attribute(m, "parallel", "on")

    # Set the relative optimality gap to 5%
    set_optimizer_attribute(m, "mip_rel_gap", 0.05)

    # Set the optimizer output verbosity
    # set_optimizer_attribute(m, "log_to_console", false)

    # Save solver logs
    set_optimizer_attribute(m, "log_file", "solver_output.log")

    ####################################
    ######## Decision variables ########
    ####################################

    # x_i,k,v,t: A binary variable that equals 1 if task k at turbine i is being completed by vessel v during timestep t, and 0 otherwise.
    @variable(m, x[1:nTurbines, 1:nTasks, 1:nVessels, 1:nTime], Bin) # TODO: maybe there is a way to reduce the size of X if Y has the other information

    # t_end: timestep when the last task finishes
    @variable(m, t_end >= 0, Int)

    # y_i,k,v: A binary variable that tracks if vessel v is assigned to task k at turbine i
    @variable(m, y[1:nTurbines, 1:nTasks, 1:nVessels], Bin)

    # z_s,t: A binary variable that tracks if a string is complete at time t
    @variable(m, z[1:nStrs, 1:nTime], Bin)

    # start_i,k,v,t: A continuous [0,1] variable that tracks if vessel v begins task k at turbine i at timestep t; does not
    # need to be binary because the constraints below force it to behave correctly given binary x.
    # NEW SINCE REPORT.
    @variable(m, 0 <= start[1:nTurbines, 1:nTasks, 1:nVessels, 1:nTime] <= 1)


    ######################################
    ######## Objective Function ##########
    ######################################

    # Weighted sum method based on Section 15.3.1 in Algorithms for Optimization - Kochenderfer and Wheeler

    # Check valid weight
    @assert (w1 <= 1 && w1 >= 0) || throw("ERROR: weight must be between 0 and 1")
    
    # w2 is implicitly 1 - w1 - w3
    @assert w1 + w3 <= 1.0 "ERROR: w1 + w3 must be <= 1"
    w2 = 1 - w1 - w3

    # Minimize total installation time using a time penalty, cost penalty, and emissions penalty approach
    # where there is a cost benefit to completing strings earlier (from eqn 1a in Amorosi et al.)
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
    # - start:         (nTurbines, nTasks, nVessels, nTime), continuous [0,1]

    # 1) Task completion: The total duration spent on a task at a turbine across all vessels must be the duration of the task. This also ensures each task is only completed once.
    @constraint(m, [i=1:nTurbines, k=1:nTasks], sum(x[i,k,v,t] for v=1:nVessels, t=1:nTime) == baseDurations[k])

    # 2) Only valid vessels: Incapable vessels do no work on tasks they cannot perform (necessary for the time objective, because cost is ignored)
    @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels], sum(x[i,k,v,t] for t=1:nTime) <= taskCapable[v,k] * baseDurations[k])

    # 3) Continuous tasks: Once a task stops, it cannot start again. This ensures no tasks are repeated.
    # start[i,k,v,t] must be >= 1 at every timestep where x[i,k,v,t] rises from 0 to 1, and max 1 rise is allowed per (i,k,v).
    # NEW SINCE REPORT, replaces previous continuous tasks: O(nTime^2) --> O(nTime). Old approach made the MIP effectively unsolvable at scale.
    # This more closely follows Housner et al. 2026 constraints 12, 14-16
    @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels], start[i,k,v,1] >= x[i,k,v,1])
    @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=2:nTime], start[i,k,v,t] >= x[i,k,v,t] - x[i,k,v,t-1])
    @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels], sum(start[i,k,v,t] for t=1:nTime) <= 1)

    # 4) Single vessel per task: exactly one vessel is assigned to each (turbine, task) pair
    @constraint(m, [i=1:nTurbines, k=1:nTasks], sum(y[i,k,v] for v=1:nVessels) == 1)

    # 5) Vessel usage: A vessel can only do task k at turbine i if it is assigned to it
    @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels], sum(x[i,k,v,t] for t=1:nTime) <= baseDurations[k] * y[i,k,v])

    # 6) Task Dependency: Task k+1 cannot begin until task k is fully completed at turbine i. If task k+1 is active at t, then the sum of task k times to t-1 across all vessels 
    # needs to be baseDurations (i.e. any vessel doing task k needs to be finished). Modeled off Housner et al. eqn. 19
    @constraint(m, [i=1:nTurbines, k=1:nTasks-1, v=1:nVessels, t=1:nTime], x[i,k+1,v,t] * baseDurations[k] <= sum(x[i,k,vv,tt] for vv=1:nVessels, tt=1:t-1))

    # 7) Resource limit:  Each vessel v can only be assigned to one turbine-task pair in timestep t. This ensures that vessels are not assigned to simultaneous tasks.
    @constraint(m, [t=1:nTime, v=1:nVessels], sum(x[i,k,v,t] for i = 1:nTurbines, k = 1:nTasks) <= 1)

    # 8) Weather constraints: if weather conditions at time t prevent vessel v from operating, then x[i,k,v,t] must be 0 for all i and k. We can use the weather operability matrix we created above to enforce this.
    @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=1:nTime], x[i,k,v,t] <= weather[v,t])
    
    # 9) Set ending time: t_end must be >= the last time step any task is active. This sets t_end as the time the last task is completed.
    @constraint(m, [i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=1:nTime], t_end >= t * x[i,k,v,t])

    # 10) String completeness: z[s,t] = 0 if all turbines in string s have not been completed: 
    #    Z * (theoretical time to complete all tasks in the string) <= (actual time taken before t for all tasks at the turbines in the string), and objective pushes z to 1 early
    @constraint(m, [s=1:nStrs, t=1:nTime], z[s,t] * sum(baseDurations) * length(Str[s]) <= sum(x[i,k,v,tt] for i in Str[s], k=1:nTasks, v=1:nVessels, tt=1:t))

    # 11) String completeness: Once a string is complete, it stays complete
    @constraint(m, [s=1:nStrs, t=2:nTime], z[s,t] >= z[s,t-1])


    ######################################
    ########### Print and solve ##########
    ######################################

    # print(m)
    # set_optimizer_attribute(m, "seconds", 600) # solver time limit

    optimize!(m)

    ObjValue = objective_value(m);

    x_return = value(x)
    z_return = value(z)
    t_end_return = value(t_end)

    emissions_return = sum(value(x[i,k,v,t])*vesselEmissions[v] for i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=1:nTime)
    return ObjValue, x_return, z_return, t_end_return, emissions_return
end

# Run function for weight scenarios + store results
obj_time,  x_time,  z_time,  t_end_time,  em_time  = optimizer(1.0, 0.0)   # pure time
obj_emit,  x_emit,  z_emit,  t_end_emit,  em_emit  = optimizer(0.0, 1.0)   # pure emissions
w1 = 0.33
w3 = 0.34
obj_equal, x_equal, z_equal, t_end_equal, em_equal = optimizer(w1,  w3)    # equal weight

# Use equal-weight results for  Gantt chart
objective, x, z, t_end, emissions = obj_equal, x_equal, z_equal, t_end_equal, em_equal

# Build header string (used by Gantt chart file below)
header = "Turbine |       Task       | Vessel |" * join([f" {t:2d}" for t in 1:nTime], "")

# Save Gantt chart to txt file
open("gantt_chart.txt", "w") do io
    println(io, f"--- Gantt Chart (start time: {TimeStart}, w1={w1}, w3={w3}) ---")
    println(io, header)
    println(io, "-" ^ length(header))
    print(io, "        |                  |        |")
    for t in 1:nTime
        print(io, f"  {weatherSequence[t]:1d}")
    end
    print(io, "\n")
    println(io, "-" ^ length(header))

    for i in 1:nTurbines
        for k in 1:nTasks
            assigned_vessel = "-"
            for v in 1:nVessels
                if sum(round(Int, x[i,k,v,t]) for t in 1:nTime) > 0
                    assigned_vessel = v
                    break
                end
            end
            print(io, f"   {i:2d}   | {Tasks[k]:16s} |   {assigned_vessel:1d}    |")
            for t in 1:nTime
                if assigned_vessel != "-" && round(Int, x[i,k,assigned_vessel,t]) == 1
                    cell = "  O"
                elseif assigned_vessel != "-" && weather[assigned_vessel,t] == 0
                    cell = "  X"
                else
                    cell = "  ."
                end
                print(io, cell)
            end
            print(io, "\n")
        end
        println(io, "-" ^ length(header))
    end
    println(io, "Legend:")
    println(io, "X - Task is inactive due to weather")
    println(io, "O - Task is active")
    println(io, ". - Task could be active but is not")
end

# Plot cumulative cost over time
# At each timestep, accumulate vessel costs minus string revenue earned so far
cumulative_cost = zeros(Float64, nTime)
cumulative_vessel_cost = zeros(Float64, nTime)
cumulative_string_revenue = zeros(Float64, nTime)
for t in 1:nTime
    # Add vessel operating costs
    vessel_cost_t = sum(round(Int, x[i,k,v,t]) * taskCosts[v][k]
                       for i=1:nTurbines, k=1:nTasks, v=1:nVessels)

    # Subtract string revenue earned
    string_revenue_t = sum(round(Int, z[s,t]) * revenue * length(Str[s]) 
                          for s=1:nStrs)

    # Cumulative sums
    cumulative_vessel_cost[t]   = (t > 1 ? cumulative_vessel_cost[t-1]   : 0) + vessel_cost_t
    cumulative_string_revenue[t] = (t > 1 ? cumulative_string_revenue[t-1] : 0) + string_revenue_t
    cumulative_cost[t]          = cumulative_vessel_cost[t] - cumulative_string_revenue[t]
end

plot(1:nTime, cumulative_cost,
    xlabel = "Timestep (12-hr periods, starting at $(TimeStart))",
    ylabel = "Cumulative Cost (\$)",
    title  = "Cumulative Installation Cost Over Time",
    label  = "Net Cost",
    linewidth = 2,
    color = :blue,
    dpi = 300
)
plot!(1:nTime, cumulative_vessel_cost,
    label = "Vessel Cost",
    linewidth = 2,
    color = :red,
    linestyle = :dash
)
plot!(1:nTime, -cumulative_string_revenue,
    label = "String Revenue",
    linewidth = 2,
    color = :purple,
    linestyle = :dot
)

# Add vertical lines when each string comes online
for s in 1:nStrs
    # Find first timestep where string s is complete
    completion_t = findfirst(t -> round(Int, z[s,t]) == 1, 1:nTime)
    if !isnothing(completion_t)
        vline!([completion_t], 
               label = f"String {s} online",
               linestyle = :dash,
               color = s == 1 ? :green : :orange)
        annotate!(completion_t, cumulative_cost[completion_t], 
                 text(f"Str {s}", 8, :left))
    end
end

savefig("cost_over_time.png")

# Triangular grid sweep over the full weight simplex
nSteps = 5  # gives 21 weight combinations
results = []
for i in 0:nSteps
    for j in 0:nSteps-i
        w1 = i / nSteps
        w3 = j / nSteps
        w2 = 1 - w1 - w3
        local obj, x, z, t_end, emissions = optimizer(w1, w3; logLevel=0)
        cost_obj = sum(x[ii,k,v,t]*taskCosts[v][k] for ii=1:nTurbines,
                      k=1:nTasks, v=1:nVessels, t=1:nTime
                      if taskCosts[v][k] < int_inf)
        push!(results, (w1=w1, w2=w2, w3=w3,
                        time_days=t_end*0.5,
                        cost_k=cost_obj/1000,
                        emissions_kg=emissions))
    end
end

# Export results CSV for ternary plot — run plot_ternary.py to generate pareto_ternary.png
results_df = DataFrame(results)
CSV.write("pareto_results.csv", results_df)


####################################
########## Results Summary #########
####################################

scenarios = [
    ("Pure Time",      x_time,  z_time,  t_end_time,  em_time),
    ("Pure Emissions", x_emit,  z_emit,  t_end_emit,  em_emit),
    ("Equal Weight",   x_equal, z_equal, t_end_equal, em_equal),
]

println("\n" * "=" ^ 75)
println("RESULTS SUMMARY")
println("=" ^ 75)
println("Scenario           | Duration  |   Vessel Cost |       Revenue |      Net Cost |   CO2 (kg)")
println("-" ^ 75)

for (name, sx, sz, st_end, semissions) in scenarios
    total_vessel_cost = sum(
        round(Int, sx[i,k,v,t]) * taskCosts[v][k]
        for i=1:nTurbines, k=1:nTasks, v=1:nVessels, t=1:nTime
        if taskCosts[v][k] < int_inf
    )

    total_revenue = sum(
        round(Int, sz[s,t]) * revenue * length(Str[s])
        for s=1:nStrs, t=1:nTime
    )

    net_cost      = total_vessel_cost - total_revenue
    duration_days = st_end * 0.5  # 12-hr timesteps → days

    println(
        rpad(name, 18) * " | " *
        lpad(f"{duration_days:.1f}", 6) * " d | \$" *
        lpad(f"{total_vessel_cost:.0f}", 13) * " | \$" *
        lpad(f"{total_revenue:.0f}", 12) * " | \$" *
        lpad(f"{net_cost:.0f}", 12) * " | " *
        lpad(f"{semissions:.1f}", 10)
    )

    for s in 1:nStrs
        t_complete = findfirst(t -> round(Int, sz[s,t]) == 1, 1:nTime)
        if !isnothing(t_complete)
            println(f"  → String {s} online: timestep {t_complete + TimeStart - 1} (day {t_complete * 0.5:.1f})")
        end
    end
end

println("=" ^ 75)