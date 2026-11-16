
# =============================================================================
#  Phase 1 MDP for floating offshore wind installation: setting, states, observations, actions
#
#  @author: ryandavies19, annasylla, mat0209, Claude (Opus 5)
# =============================================================================

using Printf
using CSV
using DataFrames
# Algs from Kochenderfer, Wheeler & Wray, Algorithms for Decision Making (CC BY-NC-ND 4.0).
include(joinpath(@__DIR__, "dm_algorithms.jl"))

# -----------------------------------------------------------------------------
# 1. User Inputs
# -----------------------------------------------------------------------------

@enum VesselType AHTS TUG CLV

const TASK_NAMES = ["mooring", "tow", "hookup", "cable"]
const SEA_STATES = 0:9                       # W, WMO code 3700

"Weather input for Phase 1: the sea-state sequence ŵ_1, …, ŵ_{T+1}."
struct DeterministicWeather
    seq::Vector{Int}
end

Base.@kwdef struct InstallSetting
    T::Int = 30                                       # horizon (12-h steps)
    nturb::Int = 2                                    # N
    strings::Vector{Vector{Int}} = [[1, 2]]           # S, partition of I
    durations::Vector{Int} = [2, 2, 2, 2]             # D_k
    vname::Vector{String} = ["A", "B", "C"]
    vtype::Vector{VesselType} = [AHTS, TUG, CLV]      # τ(j)
    capable::Vector{Vector{Int}} = [[1], [2, 3], [4]] # K_j
    max_Hs::Dict{VesselType,Float64} =                # ORBIT limits (m)
        Dict(AHTS => 3.0, TUG => 3.0, CLV => 2.0)
    # All rewards in k$ per 12-h step.
    r::Float64 = 7.2                                 # revenue per turbine per step: 15 MW × 12 h × 50 % × 80 $/MWh
    # Day rates from Data/vessel_data.csv: A 107.187, B 37.516, C 41.171 k$/day.
    M::Vector{Float64} = [322.0, 113.0, 123.5]        # M_j, weather penalty at H_s = H^max_j (3 × day rate)
    C_d::Vector{Float64} = [54.0, 19.0, 20.6]         # C_{d,j}, cost per disrupted step (day rate / 2)
    γ::Float64 = 1.0                                  # discount (finite horizon)
    weather::DeterministicWeather
end

turbines(m::InstallSetting) = 1:m.nturb
tasks(m::InstallSetting) = eachindex(m.durations)
vessels(m::InstallSetting) = eachindex(m.vtype)
nvessels(m::InstallSetting) = length(m.vtype)

# P: total work units per turbine.
total_work(m::InstallSetting) = sum(m.durations)

# o_k: work units completed before task k starts.
offsets(m::InstallSetting) = [0; cumsum(m.durations)[1:end-1]]

"Check that the setting is well formed."
function validate(m::InstallSetting)
    @assert length(m.vname) == length(m.vtype) == length(m.capable) == length(m.M) == length(m.C_d)
    @assert all(d -> d >= 1, m.durations)
    @assert sort(vcat(m.strings...)) == collect(turbines(m)) "strings must partition I"
    @assert all(k -> k in tasks(m), vcat(m.capable...))
    @assert all(k -> any(j -> k in m.capable[j], vessels(m)), tasks(m)) "every task needs a vessel"
    @assert length(m.weather.seq) >= m.T + 1 "need ŵ_1..ŵ_{T+1}"
    @assert all(w -> w in SEA_STATES, m.weather.seq)
    return true
end

# --- weather helpers (ERA5 Hs → sea state) ------------------------------------

const DOUGLAS_UPPER = [0.0, 0.1, 0.5, 1.25, 2.5, 4.0, 6.0, 9.0, 14.0]  # codes 0–8

"WMO sea-state code from significant wave height Hs [m]."
function sea_state(Hs::Real)
    k = findfirst(ub -> Hs <= ub, DOUGLAS_UPPER)
    return k === nothing ? 9 : k - 1
end

"Average hourly Hs over 12-h blocks and convert each block to a sea-state code."
function sea_states_from_hourly(hs::AbstractVector{<:Real}; step::Int = 12)
    nblocks = length(hs) ÷ step
    return [sea_state(sum(hs[(b-1)*step+1:b*step]) / step) for b in 1:nblocks]
end

const WEATHER_CSV = joinpath(@__DIR__, "..", "Data",
                             "gulf_of_maine_weather_states_12hourly.csv")

"""
    load_weather(; t0, T, path)

Build the Phase 1 weather sequence ŵ_1 … ŵ_{T+1} from 12-h ERA5 significant wave heights.

Parameters
----------
t0 : Int
    CSV row to start at. Default of 200 matches ILP report starting point, chosen as a 
    stretch of workable weather.
T : Int
    Planning horizon in 12-h steps.
path : AbstractString
    Path to the 12-hourly weather CSV.

Returns
-------
DeterministicWeather
    Sea-state codes for rows `t0` through `t0 + T`.

Notes
-----
This doesn NOT use `df.weather_state` - that column holds the ILP's 
3-level operability class (values 1..3, from the wind/wave thresholds 
at ILP/main.jl:157-158), not a WMO 0-9 sea-state code.
"""
function load_weather(; t0::Int = 200, T::Int = 30, path::AbstractString = WEATHER_CSV)
    df = CSV.read(path, DataFrame)
    @assert t0 + T <= nrow(df) "weather window runs past the end of $(path)"
    seq = sea_state.(df.wave_height[t0:t0+T])
    @assert maximum(seq) <= 8 "sea state 9 -> Hs_upper = Inf -> -Inf reward"
    return DeterministicWeather(seq)
end

# -----------------------------------------------------------------------------
# 2. Progress helpers  κ(p) and ρ(p)
# -----------------------------------------------------------------------------

"κ(p): task k with o_k ≤ p < o_k + D_k (task in progress), 0 if p = P."
function current_task(m::InstallSetting, p::Int)
    o = offsets(m)
    for k in tasks(m)
        o[k] <= p < o[k] + m.durations[k] && return k
    end
    return 0
end

"ρ(p): task k with p = o_k (task ready to start), 0 otherwise."
function ready_task(m::InstallSetting, p::Int)
    o = offsets(m)
    for k in tasks(m)
        p == o[k] && return k
    end
    return 0
end

is_operating(m::InstallSetting, p::Int) = p == total_work(m)

"Readable stage of a turbine, e.g. \"tow 1/2 done\"."
function stage_name(m::InstallSetting, p::Int)
    p == 0 && return "nothing done"
    o = offsets(m)
    k = findlast(kk -> o[kk] < p, tasks(m))
    s = "$(TASK_NAMES[k]) $(p - o[k])/$(m.durations[k]) done"
    return is_operating(m, p) ? s * " = operating" : s
end

# -----------------------------------------------------------------------------
# 3. States
# -----------------------------------------------------------------------------

struct State
    t::Int
    p::Vector{Int}   # p^(i), i ∈ I
    v::Vector{Int}   # v^(j), j ∈ J
    w::Int           # w_t
end

# Vector fields need content-based equality and hashing to be used as Dict keys.
Base.:(==)(a::State, b::State) = a.t == b.t && a.p == b.p && a.v == b.v && a.w == b.w
Base.hash(s::State, h::UInt) = hash((s.t, s.p, s.v, s.w), h)

"s_1 = (1, 0, 0, ŵ_1)."
initial_state(m::InstallSetting) =
    State(1, zeros(Int, m.nturb), zeros(Int, nvessels(m)), m.weather.seq[1])

isterminal(m::InstallSetting, s::State) = s.t == m.T + 1

"All states at time t (Phase 1: w = ŵ_t), raw product space."
function states_at(m::InstallSetting, t::Int)
    w = m.weather.seq[t]
    P = total_work(m)
    out = State[]
    for pc in Iterators.product(ntuple(_ -> 0:P, m.nturb)...),
        vc in Iterators.product(ntuple(_ -> 0:m.nturb, nvessels(m))...)
        push!(out, State(t, collect(pc), collect(vc), w))
    end
    return out
end

"Full state space X (all t = 1..T+1)."
state_space(m::InstallSetting) = vcat((states_at(m, t) for t in 1:m.T+1)...)

"Upper bound on |(p, v)| per time step: (P+1)^N (N+1)^|J|."
state_bound_per_step(m::InstallSetting) = (total_work(m) + 1)^m.nturb * (m.nturb + 1)^nvessels(m)

function Base.show(io::IO, s::State)
    v = join(["$(j):" * (i == 0 ? "idle" : "T$i") for (j, i) in enumerate(s.v)], ", ")
    print(io, "s(t=$(s.t), p=$(Tuple(s.p)), v=[$v], w=$(s.w))")
end

# -----------------------------------------------------------------------------
# 4. Observations
# -----------------------------------------------------------------------------
# Phase 1: fully observable, o_t = s_t and O(o | s) = 1[o = s].
# Phase 2: replace `observe` / `obs_prob` with a weather observation model
# (e.g. forecasts); p and v stay fully observed.

const Observation = State

observe(m::InstallSetting, s::State) = s
obs_prob(m::InstallSetting, o::Observation, s::State) = o == s ? 1.0 : 0.0

# -----------------------------------------------------------------------------
# 5. Actions
# -----------------------------------------------------------------------------
# a = (a^(j))_{j ∈ J} ∈ A = ({0} ∪ I)^|J|
#   a^(j) = 0 : vessel j does not work this step
#   a^(j) = i : vessel j works at turbine i (starts ρ(p^(i)) if idle,
#               continues κ(p^(i)) if already there)

const Action = Vector{Int}

"Full action space A (unconstrained), |A| = (N+1)^|J|."
action_space(m::InstallSetting) =
    [collect(c) for c in Iterators.product(ntuple(_ -> 0:m.nturb, nvessels(m))...)]

"""
    isfeasible(m, s, a)

Test whether action `a` is in the feasible set A(s) ⊆ A.

Parameters
----------
m : InstallSetting
    Problem definition.
s : State
    Current state (t, p, v, w).
a : Action
    Candidate joint action, one entry per vessel.

Returns
-------
Bool
    True if vessel j may take a^(j) = i for every j.
"""
function isfeasible(m::InstallSetting, s::State, a::Action)
    P = total_work(m)
    busy = Set{Int}()
    for j in vessels(m) # check the action of each vessel
        # for the checks, this uses the syntax `<boolean expression> && return false`
        # which is the same as `if <boolean expression> then return false`
        i = a[j]
        i == 0 && continue
        i in busy && return false    # if i is busy, not feasible                                 
        push!(busy, i)               # if i not busy, add it to busy (turbine cannot have 2 actions simultaneously)
        s.p[i] == P && return false  # if turbine complete            
        k = current_task(m, s.p[i])
        (k == 0 || !(k in m.capable[j])) && return false # current task not feasible              
        (s.v[j] != 0 && s.v[j] != i) && return false     # vessel is active at another turbine          
        any(j2 -> j2 != j && s.v[j2] == i, vessels(m)) && return false  # any other vessels active on this turbine
    end
    return true
end

"Feasible actions A(s) ⊆ A."
actions(m::InstallSetting, s::State) = filter(a -> isfeasible(m, s, a), action_space(m))

# -----------------------------------------------------------------------------
# 6. Transitions and constraints
# -----------------------------------------------------------------------------
# Feasibility checked with `isfeasible` in section 5 above.

"""
    work_indicator(m, a, i)

η^(i)(A): is turbine i worked this step?

Parameters
----------
m : InstallSetting
    Problem definition.
a : Action
    Joint action.
i : Int
    Turbine index.

Returns
-------
Bool
    True if some vessel targets turbine i.
"""
work_indicator(m::InstallSetting, a::Action, i::Int) = any(==(i), a)

"""
    at_boundary(m, p)

β(p): does is p at the end of a task?
Equivalent to `ready_task(m,p) != 0 || is_operating(m,p)` because q_{k+1} = q_k + D_k.

Parameters
----------
m : InstallSetting
    Problem definition.
p : Int
    Progress value on one turbine.

Returns
-------
Bool
    True at {2,4,6,8} for D = (2,2,2,2).
"""
at_boundary(m::InstallSetting, p::Int) =
    p > 0 && (ready_task(m, p) != 0 || is_operating(m, p))

"""
    successor(m, s, a)

x′ = f(x, A), the deterministic next state (for phase 1).

At most one vessel per turbine, so η is the incremental increase.

Parameters
----------
m : InstallSetting
    Problem definition.
s : State
    Current state.
a : Action
    Joint action, assumed feasible.

Returns
-------
State
    The single successor at time t + 1.
"""
function successor(m::InstallSetting, s::State, a::Action)
    p = [s.p[i] + (work_indicator(m, a, i) ? 1 : 0) for i in turbines(m)]
    v = [a[j] == 0 ? 0 : (at_boundary(m, p[a[j]]) ? 0 : a[j]) for j in vessels(m)]
    return State(s.t + 1, p, v, m.weather.seq[s.t+1])
end

"""
    transition(m, s, a, sp)

T(x′ | x, A) = 1[x′ = f(x, A)]  (doc eq. 11).

Parameters
----------
m : InstallSetting
    Problem definition.
s : State
    Current state.
a : Action
    Joint action.
sp : State
    Candidate next state.

Returns
-------
Float64
    1.0 if `sp` is the successor of (s, a), else 0.0.

Notes
-----
Phase 2 only has to replace the weather factor with P_w(w′ | w) — nothing
downstream changes. Argument order matches the textbook's T(s, a, s′).
"""
transition(m::InstallSetting, s::State, a::Action, sp::State) =
    sp == successor(m, s, a) ? 1.0 : 0.0

# -----------------------------------------------------------------------------
# 7. Rewards
# -----------------------------------------------------------------------------
# R(s, a) = R_weather(s, a) + R_d(s, a) + R_rev(s)
# Objective: max_π E[ Σ_{t=1}^T γ^(t-1) R(s_t, a_t) ], with γ = 1.

"H_s(w): upper bound of the wave-height interval of sea state w (Inf for w = 9)."
Hs_upper(w::Int) = w == 9 ? Inf : DOUGLAS_UPPER[w+1]

"R_rev(s) = r Σ_S |S| 1[p^(i) = P ∀ i ∈ S]: revenue of fully operating strings."
function reward_revenue(m::InstallSetting, s::State)
    P = total_work(m)
    return m.r * sum(length(S) for S in m.strings if all(s.p[i] == P for i in S); init = 0)
end

"R_weather(s, a) = Σ_j 1[a^(j) ≠ 0] (1 - (1+M_j)^(H_s(w) / H^max_j)): -M_j at the vessel limit."
function reward_weather(m::InstallSetting, s::State, a::Action)
    Hs = Hs_upper(s.w)
    return sum(1 - (1 + m.M[j])^(Hs / m.max_Hs[m.vtype[j]]) for j in vessels(m) if a[j] != 0; init = 0.0)
end

"R_d(s, a) = -Σ_j C_{d,j} 1[v^(j) ≠ 0 ∧ a^(j) = 0]: vessel mid-task that does not work."
reward_disruption(m::InstallSetting, s::State, a::Action) =
    -sum(m.C_d[j] for j in vessels(m) if s.v[j] != 0 && a[j] == 0; init = 0.0)

"Total reward R(s, a)."
reward(m::InstallSetting, s::State, a::Action) =
    reward_weather(m, s, a) + reward_disruption(m, s, a) + reward_revenue(m, s)

# -----------------------------------------------------------------------------
# 8. Exact solution: finite-horizon backward induction
# -----------------------------------------------------------------------------

"""
    sindex(m, p, v)
    sindex(m, s)

Mixed-radix pack of (p, v) into a linear state index.

Parameters
----------
m : InstallSetting
    Problem definition.
p : Vector{Int}
    Progress per turbine.
v : Vector{Int}
    Vessel commitments.
s : State
    Alternatively, a state whose `p` and `v` are used.

Returns
-------
Int
    Index in 1 … (P+1)^N (N+1)^|J|.

Notes
-----
t and w are not encoded: t is the array column, and w is a deterministic function of t in
Phase 1.
"""
function sindex(m::InstallSetting, p::Vector{Int}, v::Vector{Int})
    P1, N1 = total_work(m) + 1, m.nturb + 1
    idx = 0
    for i in turbines(m)
        idx = idx * P1 + p[i]
    end
    for j in vessels(m)
        idx = idx * N1 + v[j]
    end
    return idx + 1
end
sindex(m::InstallSetting, s::State) = sindex(m, s.p, s.v)

"""
    build_mdp(m)

Wrap an `InstallSetting` in the textbook's MDP container.

Parameters
----------
m : InstallSetting
    Problem definition.

Returns
-------
MDP
    Algorithm 7.1 container (MDP/dm_algorithms.jl) with fields γ, 𝒮, 𝒜, T, R, TR.

Notes
-----
The field mapping is direct. `transition` already has the book's T(s, a, s′) argument order,
and `TR` is the sample-transition-and-reward pair the Chapter 9 online methods expect, which
is trivial here because the transition is deterministic.

`𝒮` is populated for faithfulness, but the specialized `lookahead` below never iterates it.
"""
build_mdp(m::InstallSetting) = MDP(
    m.γ,
    state_space(m),
    vec(action_space(m)),            # vec: action_space returns an (N+1)^|J| Array, not a Vector
    (s, a, sp) -> transition(m, s, a, sp),
    (s, a)     -> reward(m, s, a),
    (s, a)     -> (successor(m, s, a), reward(m, s, a)),
)

"""
    LayeredValue(m, U)

The value function as a (state index × time) table, callable as `U(s)`.

Parameters
----------
m : InstallSetting
    Problem definition, used for `sindex`.
U : Matrix{Float64}
    Values with `T + 1` columns; the last is the terminal value and stays zero.

Notes
-----
Being callable is what satisfies the textbook's value-function interface. The matrix is held
by reference, so writes made during the sweep are visible immediately.
"""
struct LayeredValue
    m::InstallSetting
    U::Matrix{Float64}
end
(Uf::LayeredValue)(s::State) = Uf.U[sindex(Uf.m, s), s.t]

"""
    lookahead(𝒫, U::LayeredValue, s, a)

Specialization of Algorithm 7.2 for this problem's deterministic transition.

Parameters
----------
𝒫 : MDP
    Textbook problem container.
U : LayeredValue
    Current value table.
s : State
    Current state.
a : Action
    Action to evaluate.

Returns
-------
Float64
    The state-action value R(s,a) + γ U(f(s,a)).

Notes
-----
The book's `lookahead` computes `R(s,a) + γ Σ_{s′∈𝒮} T(s,a,s′) U(s′)`. With T a point mass
that sum has exactly one non-zero term, so it is identically `U(f(s,a))` — an exact rewrite,
not an approximation. It is also the difference between a usable solver and an unusable one:
the generic version touches all 67,797 states per state-action pair, making
`solve(ValueIteration(k), 𝒫)` cost roughly k · 1.2e11 operations.

Dispatches ahead of both book methods because `LayeredValue` is more specific than their
untyped `U` and their `U::Vector`.
"""
lookahead(𝒫::MDP, U::LayeredValue, s::State, a::Action) =
    𝒫.R(s, a) + 𝒫.γ * U(successor(U.m, s, a))

"""
    greedy_feasible(𝒫, m, U, s)

Algorithm 7.5's `greedy`, restricted to A(s) and returning the action's index.

Parameters
----------
𝒫 : MDP
    Textbook problem container.
m : InstallSetting
    Problem definition, used for the feasibility test.
U : LayeredValue
    Current value table.
s : State
    State to act from.

Returns
-------
NamedTuple
    `(i, u)`: index of the best action in `𝒫.𝒜`, and its value.

Notes
-----
The textbook `MDP` carries one global `𝒜` with no per-state feasible set, so the restriction
has to happen here. Filtering is used rather than an `-Inf` reward for infeasible actions:
inside the book's generic `lookahead`, `0.0 * -Inf` is `NaN`, and `findmax` over NaNs
silently returns the first index. A(s) is never empty — all-idle always passes F1-F5.
"""
function greedy_feasible(𝒫::MDP, m::InstallSetting, U, s::State)
    best_u, best_i = -Inf, 1
    for (i, a) in enumerate(𝒫.𝒜)
        isfeasible(m, s, a) || continue
        u = lookahead(𝒫, U, s, a)
        if u > best_u
            best_u, best_i = u, i
        end
    end
    return (i = best_i, u = best_u)
end

"""
    backward_induction(m)

Exact finite-horizon dynamic programming, built on the textbook structures above.

Parameters
----------
m : InstallSetting
    Problem definition.

Returns
-------
U : Matrix{Float64}
    Values, `(state index) × (T + 1)`. Column `T + 1` is terminal and stays zero.
Π : Matrix{Int}
    Optimal action index into `A`, `(state index) × T`.
A : Vector{Action}
    The action space, in the order `Π` indexes.

Notes
-----
Because t is part of the state and advances by exactly one each step, the chain is a layered
DAG, so one reverse sweep over `states_at(m, t)` is exact rather than iterative. That is why
this uses a single pass instead of `solve(ValueIteration(k_max), 𝒫)`: value iteration would
need T sweeps to propagate information back from the horizon and each sweep would pay the
full Σ over 𝒮. Backward induction costs |S| · |A| ≈ 1.8e6 lookups and runs in under a second.
"""
function backward_induction(m::InstallSetting)
    𝒫  = build_mdp(m)
    nS = state_bound_per_step(m)
    U  = zeros(Float64, nS, m.T + 1)   # column T+1 is the terminal value, all zeros
    Π  = ones(Int, nS, m.T)
    Uf = LayeredValue(m, U)
    for t in m.T:-1:1, s in states_at(m, t)
        g = greedy_feasible(𝒫, m, Uf, s)
        U[sindex(m, s), t] = g.u
        Π[sindex(m, s), t] = g.i
    end
    return U, Π, 𝒫.𝒜
end

# -----------------------------------------------------------------------------
# 9. Rollout and Gantt chart
# -----------------------------------------------------------------------------

"""
    rollout(m, Π, A)

Run the policy forward from s_1.

Parameters
----------
m : InstallSetting
    Problem definition.
Π : Matrix{Int}
    Policy from `backward_induction`.
A : Vector{Action}
    Action space `Π` indexes into.

Returns
-------
traj : Vector{Tuple{State,Action}}
    The (state, action) pair visited at each step 1 … T.
x : BitArray{4}
    `x[i,k,j,t]` is true when vessel j worked task k at turbine i during step t.
s_final : State
    The state at t = T + 1.

Notes
-----
`x` uses the same indexing as the ILP's decision variable, so the two Gantt writers line up.
The task is read off κ(p^(i)) *before* the increment.
"""
function rollout(m::InstallSetting, Π, A)
    s = initial_state(m)
    traj = Tuple{State,Action}[]
    x = falses(m.nturb, length(m.durations), nvessels(m), m.T)
    for t in 1:m.T
        a = A[Π[sindex(m, s), t]]
        push!(traj, (s, copy(a)))
        for j in vessels(m)
            i = a[j]
            i == 0 && continue
            x[i, current_task(m, s.p[i]), j, t] = true
        end
        s = successor(m, s, a)
    end
    return traj, x, s
end

"""
    over_limit(m, j, w)

Display only: would the ILP's hard weather gate (c8, line 332) have blocked vessel j?

Parameters
----------
m : InstallSetting
    Problem definition.
j : Int
    Vessel index.
w : Int
    WMO sea-state code.

Returns
-------
Bool
    True if the sea state exceeds vessel j's maximum Hs.
"""
over_limit(m::InstallSetting, j::Int, w::Int) = Hs_upper(w) > m.max_Hs[m.vtype[j]]

"""
    print_gantt(io, m, x; t0)

Write the text Gantt chart to `io`.

Parameters
----------
io : IO
    Destination, e.g. `stdout` or an open file.
m : InstallSetting
    Problem definition.
x : BitArray{4}
    Activity grid from `rollout`.
t0 : Int
    First CSV row of the weather window, for the title line only.

Returns
-------
Nothing

Notes
-----
Mirrors the layout of `ILP/main.jl:374-417` so the two charts can be read side by side. Two
deliberate differences: the weather row carries the WMO 0-9 sea state where the ILP prints
its 3-level class, and `!` marks a vessel working beyond its Hs limit, which the MDP permits
and prices in R_weather but the ILP forbids outright.
"""
function print_gantt(io::IO, m::InstallSetting, x; t0::Int = 200)
    hdr  = "Turbine |       Task       | Vessel |" * join([@sprintf(" %2d", t) for t in 1:m.T])
    rule = "-"^length(hdr)
    @printf(io, "--- MDP Gantt Chart (start time: %d, r=%.2f, M=%.1f, C_d=%.1f) ---\n",
            t0, m.r, m.M, m.C_d)
    println(io, hdr)
    println(io, rule)
    print(io, "        |                  |        |")
    for t in 1:m.T
        @printf(io, "  %1d", m.weather.seq[t])
    end
    println(io)
    println(io, rule)
    for i in turbines(m)
        for k in tasks(m)
            j   = findfirst(jj -> any(@view x[i, k, jj, :]), vessels(m))
            lbl = j === nothing ? "-" : m.vname[j]
            @printf(io, "   %2d   | %16s |   %1s    |", i, TASK_NAMES[k], lbl)
            for t in 1:m.T
                working = j !== nothing && x[i, k, j, t]
                beyond  = j !== nothing && over_limit(m, j, m.weather.seq[t])
                print(io, working && beyond ? "  !" :
                          working           ? "  O" :
                          beyond            ? "  X" : "  .")
            end
            println(io)
        end
        println(io, rule)
    end
    println(io, "Weather row: WMO sea-state code 0-9 (Table 2 of the write-up).")
    println(io, "Legend:")
    println(io, "O - Task is active, vessel within its Hs limit")
    println(io, "! - Task is active but the vessel is OVER its Hs limit")
    println(io, "    (the MDP allows this and prices it in R_weather; the ILP forbids it)")
    println(io, "X - Task is inactive due to weather")
    println(io, ". - Task could be active but is not")
    return nothing
end

"""
    write_gantt(m, x; t0, path)

Write the Gantt chart to a file.

Parameters
----------
m : InstallSetting
    Problem definition.
x : BitArray{4}
    Activity grid from `rollout`.
t0 : Int
    First CSV row of the weather window, for the title line only.
path : AbstractString
    Output file. Covered by the repo's `*.txt` ignore rule, as the ILP's chart is.

Returns
-------
AbstractString
    The path written.
"""
function write_gantt(m::InstallSetting, x; t0::Int = 200,
                     path::AbstractString = joinpath(@__DIR__, "gantt_chart.txt"))
    open(io -> print_gantt(io, m, x; t0 = t0), path, "w")
    return path
end

# -----------------------------------------------------------------------------
# 10. Driver
# -----------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    t0, T = 200, 30                       # t0 matches ILP/main.jl:50; T = 21 overlays its window

    # Revenue in ILP dollars: $ per turbine per 12-h step, from ILP/main.jl:150
    # (9859.5 GWh AEP / 132 turbines / 365 d / 2 steps per day × $22.62/MWh).
    # The InstallSetting default r = 1.0 is a placeholder that leaves R_weather dominating
    # R_rev, under which the optimal policy is to do nothing and the Gantt comes out empty.
    # M stays at its default 300.0 and is the knob to tune from here.
    m = InstallSetting(T = T, r = 2314.47, weather = load_weather(t0 = t0, T = T))
    validate(m)

    el = @elapsed ((U, Π, A) = backward_induction(m))
    traj, x, sT = rollout(m, Π, A)

    # The chart itself, in the ILP's layout.
    print_gantt(stdout, m, x; t0 = t0)

    # Solver and policy summary, below the chart.
    done = findfirst(t -> all(traj[t][1].p[i] == total_work(m) for i in turbines(m)), 1:m.T)
    rw = sum(reward_weather(m, s, a)    for (s, a) in traj)
    rd = sum(reward_disruption(m, s, a) for (s, a) in traj)
    rr = sum(reward_revenue(m, s)       for (s, _) in traj)

    println()
    @printf("states/step = %d   |A| = %d   backward induction %.2f s\n",
            state_bound_per_step(m), length(A), el)
    @printf("V*(s_1) = %.2f   (weather %.2f | disruption %.2f | revenue %.2f)\n",
            U[sindex(m, initial_state(m)), 1], rw, rd, rr)
    @printf("final progress p = %s  (P = %d), worked vessel-steps = %d\n",
            string(Tuple(sT.p)), total_work(m), count(x))
    println(done === nothing ? "string never completed within the horizon" :
            "all turbines operating from step $(done)")

    any(x) || @warn """EMPTY schedule — the optimal policy is to do nothing.
        R_weather dominates R_rev. Raise r or lower M."""

    println("wrote ", write_gantt(m, x; t0 = t0))
end

