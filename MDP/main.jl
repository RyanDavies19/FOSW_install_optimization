
# =============================================================================
#  Phase 1 MDP for floating offshore wind installation: setting, states, observations, actions
#
#  @author: ryandavies19, annasylla, mat0209, Claude (Opus 5)
#
#  Notation follows AA228_CS238_Project_Final.pdf. Doc symbol -> code name:
#
#    x, x′      state and next state         `x`, `x′` (type `State`)
#    X          state space                  `state_space(m)`
#    t, T       time step, horizon           `x.t`, `m.T`
#    𝒯 = 1..N   turbine set                  `turbines(m)`      (doc writes this T; the
#                                                                code keeps T for the horizon)
#    𝒥          vessel set                   `vessels(m)`
#    𝒦          task set                     `tasks(m)`, `m.capable[j]` = K_j
#    𝒲          sea-state codes              `SEA_STATES`
#    S          strings (partition of 𝒯)     `m.strings`
#    p^(i)      turbine progress             `x.p[i]`
#    v^(j)      vessel status                `x.v[j]`
#    w          sea state                    `x.w`;  ŵ_t = `m.weather.seq[t]`
#    A          joint action (all vessels)   `A`       (type `Action`)
#    a^(j)      one vessel's action          `A[j]`
#    𝒜          action space                 `action_space(m)`; A(x) = `actions(m, x)`
#    q_k, D_k, P                             `offsets(m)[k]`, `m.durations[k]`, `total_work(m)`
#    κ(p), ρ(p)                              `current_task(m, p)`, `ready_task(m, p)`
#    η^(i)(A), β(p)                          `work_indicator(m, A, i)`, `at_boundary(m, p)`
#    f(x, A)                                 `successor(m, x, A)`
#    T(x′ | x, A)                            `transition(m, x, A, x′)`
#    H_s(w), H^max_j                         `Hs_upper(w)`, `m.max_Hs[m.vtype[j]]`
#    r, M_j, C_{d,j}                         `m.r`, `m.M[j]`, `m.C_d[j]`
#
#  Doc symbols that return ∅ are 0 in code (κ, ρ), since the task indices start at 1.
#
#  The solver is built from Chapter 7 of Kochenderfer, Wheeler & Wray, *Algorithms for
#  Decision Making* (see MDP/dm_algorithms.jl). Section 8 below specializes `lookahead`,
#  `greedy` and `backup` on this problem's value-function type and adds a finite-horizon
#  `solve`, so the book's `ValueFunctionPolicy` works here unchanged.
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
const SEA_STATES = 0:9                       # 𝒲, WMO code 3700

"Weather input for Phase 1: the sea-state sequence ŵ_1, …, ŵ_{T+1}."
struct DeterministicWeather
    seq::Vector{Int}
end

Base.@kwdef struct InstallSetting
    T::Int = 30                                       # horizon (12-h steps)
    nturb::Int = 2                                    # N
    strings::Vector{Vector{Int}} = [[1, 2]]           # S, partition of 𝒯
    durations::Vector{Int} = [2, 2, 2, 2]             # D_k
    vname::Vector{String} = ["A", "B", "C"]
    vtype::Vector{VesselType} = [AHTS, TUG, CLV]      # τ(j)
    capable::Vector{Vector{Int}} = [[1], [2, 3], [4]] # K_j
    max_Hs::Dict{VesselType,Float64} =                # ORBIT limits (m)
        Dict(AHTS => 3.0, TUG => 3.0, CLV => 2.0)
    # All rewards in k$ per 12-h step.
    r::Float64 = 7.2                                 # revenue per turbine per step: 15 MW × 12 h × 50 % × 80 $/MWh
    # Day rates from Data/vessel_data.csv: A 107.187, B 37.516, C 41.171 k$/day.
    M::Vector{Float64} = [322.0, 113.0, 123.5]        # M_j, weather penalty per vessel at H_s = H^max_j (3 × day rate)
    C_d::Vector{Float64} = [54.0, 19.0, 20.6]         # C_{d,j}, cost per disrupted step (day rate / 2)
    γ::Float64 = 1.0                                  # discount (finite horizon)
    weather::DeterministicWeather
end

turbines(m::InstallSetting) = 1:m.nturb
tasks(m::InstallSetting) = eachindex(m.durations)
vessels(m::InstallSetting) = eachindex(m.vtype)
nvessels(m::InstallSetting) = length(m.vtype)

# P: total work units per turbine (doc eq. 1).
total_work(m::InstallSetting) = sum(m.durations)

# q_k: work units completed before task k starts (doc eq. 1).
offsets(m::InstallSetting) = [0; cumsum(m.durations)[1:end-1]]

"Check that the setting is well formed."
function validate(m::InstallSetting)
    @assert length(m.vname) == length(m.vtype) == length(m.capable) == length(m.M) == length(m.C_d)
    @assert all(d -> d >= 1, m.durations)
    @assert sort(vcat(m.strings...)) == collect(turbines(m)) "strings must partition 𝒯"
    @assert all(k -> k in tasks(m), vcat(m.capable...))
    @assert all(k -> any(j -> k in m.capable[j], vessels(m)), tasks(m)) "every task needs a vessel"
    @assert length(m.weather.seq) >= m.T + 1 "need ŵ_1..ŵ_{T+1}"
    @assert all(w -> w in SEA_STATES, m.weather.seq)
    @assert all(>(0), m.M) "M_j > 0, else the weather term rewards bad weather"
    @assert all(>=(0), m.C_d) "C_{d,j} >= 0, else disruption is incentivized"
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

"κ(p): task k with q_k ≤ p < q_k + D_k (task in progress), 0 if p = P (doc eq. 3)."
function current_task(m::InstallSetting, p::Int)
    q = offsets(m)
    for k in tasks(m)
        q[k] <= p < q[k] + m.durations[k] && return k
    end
    return 0
end

"ρ(p): task k with p = q_k (task ready to start), 0 otherwise (doc eq. 4)."
function ready_task(m::InstallSetting, p::Int)
    q = offsets(m)
    for k in tasks(m)
        p == q[k] && return k
    end
    return 0
end

is_operating(m::InstallSetting, p::Int) = p == total_work(m)

"Readable stage of a turbine, e.g. \"tow 1/2 done\" (doc Table 3)."
function stage_name(m::InstallSetting, p::Int)
    p == 0 && return "nothing done"
    q = offsets(m)
    k = findlast(kk -> q[kk] < p, tasks(m))
    txt = "$(TASK_NAMES[k]) $(p - q[k])/$(m.durations[k]) done"
    return is_operating(m, p) ? txt * " = operating" : txt
end

# -----------------------------------------------------------------------------
# 3. States
# -----------------------------------------------------------------------------
# x = (t, p^(i), v^(j), w) ∈ X   (doc eq. 2)

struct State
    t::Int
    p::Vector{Int}   # p^(i), i ∈ 𝒯
    v::Vector{Int}   # v^(j), j ∈ 𝒥
    w::Int           # w_t
end

# Vector fields need content-based equality and hashing to be used as Dict keys.
Base.:(==)(x1::State, x2::State) = x1.t == x2.t && x1.p == x2.p && x1.v == x2.v && x1.w == x2.w
Base.hash(x::State, h::UInt) = hash((x.t, x.p, x.v, x.w), h)

"x_1 = (1, 0, 0, ŵ_1)."
initial_state(m::InstallSetting) =
    State(1, zeros(Int, m.nturb), zeros(Int, nvessels(m)), m.weather.seq[1])

isterminal(m::InstallSetting, x::State) = x.t == m.T + 1

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

"Upper bound on |(p, v)| per time step: (P+1)^N (N+1)^|𝒥|."
state_bound_per_step(m::InstallSetting) = (total_work(m) + 1)^m.nturb * (m.nturb + 1)^nvessels(m)

function Base.show(io::IO, x::State)
    v = join(["$(j):" * (i == 0 ? "idle" : "T$i") for (j, i) in enumerate(x.v)], ", ")
    print(io, "x(t=$(x.t), p=$(Tuple(x.p)), v=[$v], w=$(x.w))")
end

# -----------------------------------------------------------------------------
# 4. Observations
# -----------------------------------------------------------------------------
# Phase 1: fully observable, o_t = x_t and O(o | x) = 1[o = x]  (doc eq. 18).
# Phase 2: replace `observe` / `obs_prob` with a weather observation model
# (e.g. forecasts); p and v stay fully observed.

const Observation = State

observe(m::InstallSetting, x::State) = x
obs_prob(m::InstallSetting, o::Observation, x::State) = o == x ? 1.0 : 0.0

# -----------------------------------------------------------------------------
# 5. Actions
# -----------------------------------------------------------------------------
# a^(j) ∈ {0} ∪ 𝒯                            (doc eq. 5)
# A = (a^(j))_{j ∈ 𝒥} ∈ 𝒜 = ({0} ∪ 𝒯)^|𝒥|   (doc eq. 6)
#   a^(j) = 0 : vessel j does not work this step
#   a^(j) = i : vessel j works at turbine i, continuing κ(p^(i))

const Action = Vector{Int}

"Full action space 𝒜 (unconstrained), |𝒜| = (N+1)^|𝒥|  (doc eq. 6)."
action_space(m::InstallSetting) =
    [collect(c) for c in Iterators.product(ntuple(_ -> 0:m.nturb, nvessels(m))...)]

"""
    isfeasible(m, x, A)

Test whether joint action `A` is in the feasible set A(x) ⊆ 𝒜.

Parameters
----------
m : InstallSetting
    Problem definition.
x : State
    Current state (t, p, v, w).
A : Action
    Candidate joint action, one entry a^(j) per vessel.

Returns
-------
Bool
    True if vessel j may take a^(j) = i for every j.

Notes
-----
The five rules enforced in this function are:

====  ==========================================================================
F1    at most one vessel per turbine per step
F2    vessel j may only work tasks in K_j, i.e. κ(p^(i)) ∈ K_j
F3    no work on a completed turbine (p^(i) = P)
F4    a committed vessel cannot switch turbines: v^(j) ∈ {0, i}
F5    a vessel cannot start at a turbine another vessel already holds
====  ==========================================================================
"""
function isfeasible(m::InstallSetting, x::State, A::Action)
    P = total_work(m)
    busy = Set{Int}()
    for j in vessels(m) # check the action of each vessel
        # for the checks, this uses the syntax `<boolean expression> && return false`
        # which is the same as `if <boolean expression> then return false`
        i = A[j]
        i == 0 && continue
        i in busy && return false    # F1 if i is busy, not feasible
        push!(busy, i)               # if i not busy, add it to busy
        x.p[i] == P && return false  # F3 if turbine complete
        k = current_task(m, x.p[i])
        (k == 0 || !(k in m.capable[j])) && return false # F2 current task not feasible
        (x.v[j] != 0 && x.v[j] != i) && return false     # F4 vessel active at another turbine
        any(j2 -> j2 != j && x.v[j2] == i, vessels(m)) && return false  # F5 turbine already held
    end
    return true
end

"Feasible actions A(x) ⊆ 𝒜."
actions(m::InstallSetting, x::State) = filter(A -> isfeasible(m, x, A), action_space(m))

# -----------------------------------------------------------------------------
# 6. Transitions and constraints
# -----------------------------------------------------------------------------
# Feasibility checked with `isfeasible` in section 5 above.

"""
    work_indicator(m, A, i)

η^(i)(A): is turbine i worked this step?  (doc eq. 7)

Parameters
----------
m : InstallSetting
    Problem definition.
A : Action
    Joint action.
i : Int
    Turbine index.

Returns
-------
Bool
    True if some vessel targets turbine i.
"""
work_indicator(m::InstallSetting, A::Action, i::Int) = any(==(i), A)

"""
    at_boundary(m, p)

β(p): does is p at the end of a task?  (doc eq. 7)
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
    successor(m, x, A)

x′ = f(x, A), the deterministic next state (doc eqs. 8-11, for phase 1).

At most one vessel per turbine, so η is the incremental increase.

Parameters
----------
m : InstallSetting
    Problem definition.
x : State
    Current state.
A : Action
    Joint action, assumed feasible.

Returns
-------
State
    The single successor at time t + 1.
"""
function successor(m::InstallSetting, x::State, A::Action)
    p = [x.p[i] + (work_indicator(m, A, i) ? 1 : 0) for i in turbines(m)]
    v = [A[j] == 0 ? 0 : (at_boundary(m, p[A[j]]) ? 0 : A[j]) for j in vessels(m)]
    return State(x.t + 1, p, v, m.weather.seq[x.t+1])
end

"""
    transition(m, x, A, x′)

T(x′ | x, A) = 1[x′ = f(x, A)]  (doc eq. 12).

Parameters
----------
m : InstallSetting
    Problem definition.
x : State
    Current state.
A : Action
    Joint action.
x′ : State
    Candidate next state.

Returns
-------
Float64
    1.0 if `x′` is the successor of (x, A), else 0.0.

Notes
-----
Phase 2 only has to replace the weather factor with P_w(w′ | w) — nothing
downstream changes. Argument order matches the textbook's T(s, a, s′).
"""
transition(m::InstallSetting, x::State, A::Action, x′::State) =
    x′ == successor(m, x, A) ? 1.0 : 0.0

# -----------------------------------------------------------------------------
# 7. Rewards
# -----------------------------------------------------------------------------
# R(x, A) = R_weather(x, A) + R_d(x, A) + R_rev(x)       (doc eq. 16)
# Objective: max_π E[ Σ_{t=1}^T γ^(t-1) R(x_t, A_t) ], with γ = 1   (doc eq. 17)
# All terms are in k$ per 12-h step.

"H_s(w): upper bound of the wave-height interval of sea state w (Inf for w = 9)."
Hs_upper(w::Int) = w == 9 ? Inf : DOUGLAS_UPPER[w+1]

"R_rev(x) = r Σ_S |S| 1[p^(i) = P ∀ i ∈ S]: revenue of fully operating strings (doc eq. 13)."
function reward_revenue(m::InstallSetting, x::State)
    P = total_work(m)
    return m.r * sum(length(S) for S in m.strings if all(x.p[i] == P for i in S); init = 0)
end

"R_weather(x, A) = Σ_j 1[a^(j) ≠ 0] (1 - (1+M_j)^(H_s(w) / H^max_j)): -M_j at the limit (doc eq. 14)."
function reward_weather(m::InstallSetting, x::State, A::Action)
    Hs = Hs_upper(x.w)
    return sum(1 - (1 + m.M[j])^(Hs / m.max_Hs[m.vtype[j]])
               for j in vessels(m) if A[j] != 0; init = 0.0)
end

"R_d(x, A) = -Σ_j C_{d,j} 1[v^(j) ≠ 0 ∧ a^(j) = 0]: vessel mid-task that does not work (doc eq. 15)."
reward_disruption(m::InstallSetting, x::State, A::Action) =
    -sum(m.C_d[j] for j in vessels(m) if x.v[j] != 0 && A[j] == 0; init = 0.0)

"Total reward R(x, A)  (doc eq. 16)."
reward(m::InstallSetting, x::State, A::Action) =
    reward_weather(m, x, A) + reward_disruption(m, x, A) + reward_revenue(m, x)

# -----------------------------------------------------------------------------
# 8. Exact solution: finite-horizon backward induction
# -----------------------------------------------------------------------------
# The book's generic methods take `U` untyped; ours take `U::LayeredValue`, which
# is more specific, so Julia dispatches to these without the book's file being edited:
#
#   book 7.2  lookahead(𝒫, U, s, a)  ->  lookahead(𝒫, U::LayeredValue, x, A)
#   book 7.5  greedy(𝒫, U, s)        ->  greedy(𝒫, U::LayeredValue, x)
#   book 7.7  backup(𝒫, U, s)        ->  backup(𝒫, U::LayeredValue, x)
#
# `solve(BackwardInduction, 𝒫)` below then plays the part of the book's Algorithm 7.8
# `solve(ValueIteration, 𝒫)`, which is not copied into dm_algorithms.jl because it is not
# usable at this scale — see `lookahead` and `BackwardInduction` for why.

"""
    xindex(m, p, v)
    xindex(m, x)

Mixed-radix pack of (p, v) into a linear state index.

Parameters
----------
m : InstallSetting
    Problem definition.
p : Vector{Int}
    Progress per turbine.
v : Vector{Int}
    Vessel commitments.
x : State
    Alternatively, a state whose `p` and `v` are used.

Returns
-------
Int
    Index in 1 … (P+1)^N (N+1)^|𝒥|.

Notes
-----
t and w are not encoded: t is the array column, and w is a deterministic function of t in
Phase 1.
"""
function xindex(m::InstallSetting, p::Vector{Int}, v::Vector{Int})
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
xindex(m::InstallSetting, x::State) = xindex(m, x.p, x.v)

"""
    build_mdp(m)

Wrap an `InstallSetting` in the textbook's MDP container (Algorithm 7.1).

Parameters
----------
m : InstallSetting
    Problem definition.

Returns
-------
MDP
    Algorithm 7.1 container with fields γ, 𝒮, 𝒜, T, R, TR.

Notes
-----
The field mapping is direct. `transition` already has the book's T(s, a, s′) argument order,
and `TR` is the sample-transition-and-reward pair the Chapter 9 online methods expect, which
is trivial here because the transition is deterministic.

`𝒮` is populated but the functions below never iterate it.
"""
build_mdp(m::InstallSetting) = MDP(
    m.γ,
    state_space(m),
    vec(action_space(m)),            # vec: action_space returns an (N+1)^|𝒥| Array, not a Vector
    (x, A, x′) -> transition(m, x, A, x′),
    (x, A)     -> reward(m, x, A),
    (x, A)     -> (successor(m, x, A), reward(m, x, A)),
)

"""
    LayeredValue(m, U)

The value function as a (state index × time) table, callable as `U(x)`.

Parameters
----------
m : InstallSetting
    Problem definition, used for `xindex` and for the feasibility test in `greedy`.
U : Matrix{Float64}
    Values with `T + 1` columns; the last is the terminal value and stays zero.

Notes
-----
Being callable is what satisfies the book's value-function interface, so `lookahead`,
`greedy`, `backup` and `ValueFunctionPolicy` all accept it. Carrying `m` is what lets the
specializations below restrict to A(x) without an extra argument. The matrix is held by
reference, so writes made during the sweep are visible immediately.
"""
struct LayeredValue
    m::InstallSetting
    U::Matrix{Float64}
end
(Uf::LayeredValue)(x::State) = Uf.U[xindex(Uf.m, x), x.t]

"""
    lookahead(𝒫, U::LayeredValue, x, A)

Algorithm 7.2 specialized for this problem's deterministic transition.

Parameters
----------
𝒫 : MDP
    Textbook problem container.
U : LayeredValue
    Current value table.
x : State
    Current state.
A : Action
    Joint action to evaluate.

Returns
-------
Float64
    The state-action value R(x,A) + γ U(f(x,A)).

Notes
-----
The book's `lookahead` computes `R(s,a) + γ Σ_{s′∈𝒮} T(s,a,s′) U(s′)`. With T a point mass
that sum has exactly one non-zero term, so it is identically `U(f(x,A))` — an exact rewrite,
not an approximation. This is required to keep it solvable. The original alg. uses all 67,797 
states per state-action pair, Algorithm 7.8 value iteration cost roughly k · 1.2e11 operations 
for k sweeps. That is why we make a custom lookahead here.
"""
lookahead(𝒫::MDP, U::LayeredValue, x::State, A::Action) =
    𝒫.R(x, A) + 𝒫.γ * U(successor(U.m, x, A))

"""
    greedy(𝒫, U::LayeredValue, x)

Algorithm 7.5's `greedy`, restricted to the feasible set A(x).

Parameters
----------
𝒫 : MDP
    Textbook problem container.
U : LayeredValue
    Current value table; its `m` supplies the feasibility test.
x : State
    State to act from.

Returns
-------
NamedTuple
    `(a, u)`: the best feasible joint action and its value — the same shape the book returns,
    so `ValueFunctionPolicy` works unmodified.

Notes
-----
The book's `MDP` carries one global `𝒜` with no per-state feasible set, so the restriction
has to happen here. Filtering is used rather than an `-Inf` reward for infeasible actions:
inside the book's generic `lookahead`, `0.0 * -Inf` is `NaN`, and `findmax` over NaNs
silently returns the first index. A(x) is never empty — all-idle always passes F1-F5, and it
is `𝒫.𝒜[1]`, so it is also the fallback.
"""
function greedy(𝒫::MDP, U::LayeredValue, x::State)
    best_a, best_u = 𝒫.𝒜[1], -Inf
    for A in 𝒫.𝒜
        isfeasible(U.m, x, A) || continue
        u = lookahead(𝒫, U, x, A)
        if u > best_u
            best_a, best_u = A, u
        end
    end
    return (a = best_a, u = best_u)
end

"""
    backup(𝒫, U::LayeredValue, x)

Algorithm 7.7's `backup`, restricted to the feasible set A(x).

Parameters
----------
𝒫 : MDP
    Textbook problem container.
U : LayeredValue
    Current value table.
x : State
    State to back up.

Returns
-------
Float64
    max over A ∈ A(x) of the lookahead value — the book's `backup` with `𝒜` replaced by A(x).
"""
backup(𝒫::MDP, U::LayeredValue, x::State) = greedy(𝒫, U, x).u

"""
    BackwardInduction(m)

Finite-horizon solver in the shape of the book's Algorithm 7.8, `ValueIteration`.

Parameters
----------
m : InstallSetting
    Problem definition; supplies the horizon and the per-step state layers.
"""
struct BackwardInduction
    m::InstallSetting
end

"""
    solve(M::BackwardInduction, 𝒫)

One reverse sweep of `backup` over the time layers, returning the book's policy type.

Parameters
----------
M : BackwardInduction
    Solver settings (the `InstallSetting`).
𝒫 : MDP
    Problem container from `build_mdp`.

Returns
-------
ValueFunctionPolicy
    Algorithm 7.5's policy. Call it as `π(x)` for the optimal joint action; read values off
    `π.U(x)`.

Notes
-----
This makes the same three moves as the book's Algorithm 7.8 (p. 142) — zero the values, back
up every state repeatedly, wrap the result in a `ValueFunctionPolicy` — with 7.8's loop over
`k = 1:k_max` replaced by one pass over t = T … 1. Column T+1 is the terminal value and
stays zero. Cost is |X| · |𝒜| ≈ 1.8e6 lookups, under a second.
"""
function solve(M::BackwardInduction, 𝒫::MDP)
    m  = M.m
    U  = zeros(Float64, state_bound_per_step(m), m.T + 1)
    Uf = LayeredValue(m, U)
    for t in m.T:-1:1, x in states_at(m, t)
        U[xindex(m, x), t] = backup(𝒫, Uf, x)
    end
    return ValueFunctionPolicy(𝒫, Uf)
end

"Solve `m` exactly and return Algorithm 7.5's `ValueFunctionPolicy`."
backward_induction(m::InstallSetting) = solve(BackwardInduction(m), build_mdp(m))

# -----------------------------------------------------------------------------
# 9. Rollout and Gantt chart
# -----------------------------------------------------------------------------

"""
    rollout(m, π)

Run the policy forward from x_1.

Parameters
----------
m : InstallSetting
    Problem definition.
π : ValueFunctionPolicy
    Policy from `solve`/`backward_induction`; `π(x)` gives the joint action.

Returns
-------
traj : Vector{Tuple{State,Action}}
    The (state, joint action) pair visited at each step 1 … T.
active : BitArray{4}
    `active[i,k,j,t]` is true when vessel j worked task k at turbine i during step t.
x_final : State
    The state at t = T + 1.

Notes
-----
`active` uses the same indexing as the ILP's decision variable `x`, so the two Gantt writers
line up. It is named `active` here because `x` is the state in this file. The task is read
off κ(p^(i)) *before* the increment.
"""
function rollout(m::InstallSetting, π::ValueFunctionPolicy)
    x = initial_state(m)
    traj = Tuple{State,Action}[]
    active = falses(m.nturb, length(m.durations), nvessels(m), m.T)
    for t in 1:m.T
        A = π(x)
        push!(traj, (x, copy(A)))
        for j in vessels(m)
            i = A[j]
            i == 0 && continue
            active[i, current_task(m, x.p[i]), j, t] = true
        end
        x = successor(m, x, A)
    end
    return traj, active, x
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
    print_gantt(io, m, active; t0)

Write the text Gantt chart to `io`.

Parameters
----------
io : IO
    Destination, e.g. `stdout` or an open file.
m : InstallSetting
    Problem definition.
active : BitArray{4}
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
function print_gantt(io::IO, m::InstallSetting, active; t0::Int = 200)
    hdr  = "Turbine |       Task       | Vessel |" * join([@sprintf(" %2d", t) for t in 1:m.T])
    rule = "-"^length(hdr)
    vec_str(v) = "(" * join((@sprintf("%.1f", z) for z in v), ",") * ")"
    @printf(io, "--- MDP Gantt Chart (start time: %d, r=%.1f, M_j=%s, C_dj=%s, all k\$) ---\n",
            t0, m.r, vec_str(m.M), vec_str(m.C_d))
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
            j   = findfirst(jj -> any(@view active[i, k, jj, :]), vessels(m))
            lbl = j === nothing ? "-" : m.vname[j]
            @printf(io, "   %2d   | %16s |   %1s    |", i, TASK_NAMES[k], lbl)
            for t in 1:m.T
                working = j !== nothing && active[i, k, j, t]
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
    write_gantt(m, active; t0, path)

Write the Gantt chart to a file.

Parameters
----------
m : InstallSetting
    Problem definition.
active : BitArray{4}
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
function write_gantt(m::InstallSetting, active; t0::Int = 200,
                     path::AbstractString = joinpath(@__DIR__, "gantt_chart.txt"))
    open(io -> print_gantt(io, m, active; t0 = t0), path, "w")
    return path
end

# -----------------------------------------------------------------------------
# 10. Driver
# -----------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    # t0 matches ILP/main.jl:50. The ILP's own window is 21 steps (rows 200-220).
    #
    # T = 40, not 30. Revenue only accrues once the whole string is operating, and in this
    # window the CLV's only affordable cable weather (sea state <= 3) is steps 23-29, so the
    # string cannot come online before step 29. At T <= 35 there is no runway left to repay
    # the install cost and the optimal policy is to do nothing (V* = 0, empty chart). From
    # T = 40 the schedule is fixed and V* grows by exactly r*|S| = 14.4 k$ per added step.
    t0, T = 200, 40

    # Reward parameters are the InstallSetting defaults: r = 7.2 k$ per turbine per step,
    # M_j = (322, 113, 123.5) k$, C_{d,j} = (54, 19, 20.6) k$, day rates from
    # Data/vessel_data.csv. Nothing is overridden here.
    m = InstallSetting(T = T, weather = load_weather(t0 = t0, T = T))
    validate(m)

    𝒫 = build_mdp(m)
    el = @elapsed (π = solve(BackwardInduction(m), 𝒫))
    traj, active, x_final = rollout(m, π)

    # The chart itself, in the ILP's layout.
    print_gantt(stdout, m, active; t0 = t0)

    # Solver and policy summary, below the chart.
    done = findfirst(t -> all(traj[t][1].p[i] == total_work(m) for i in turbines(m)), 1:m.T)
    rw = sum(reward_weather(m, x, A)    for (x, A) in traj)
    rd = sum(reward_disruption(m, x, A) for (x, A) in traj)
    rr = sum(reward_revenue(m, x)       for (x, _) in traj)
    nover = count(active[i, k, j, t] && over_limit(m, j, m.weather.seq[t])
                  for i in turbines(m), k in tasks(m), j in vessels(m), t in 1:m.T)

    println()
    @printf("states/step = %d   |𝒜| = %d   backward induction %.2f s\n",
            state_bound_per_step(m), length(𝒫.𝒜), el)
    @printf("V*(x_1) = %.2f k\$   (weather %.2f | disruption %.2f | revenue %.2f)\n",
            π.U(initial_state(m)), rw, rd, rr)
    @printf("final progress p = %s  (P = %d), worked vessel-steps = %d, over-limit steps = %d\n",
            string(Tuple(x_final.p)), total_work(m), count(active), nover)
    println(done === nothing ? "string never completed within the horizon" :
            "all turbines operating from step $(done)")

    any(active) || @warn """EMPTY schedule — the optimal policy is to do nothing.
        R_weather dominates R_rev. Raise r or lower M_j."""

    println("wrote ", write_gantt(m, active; t0 = t0))
end
