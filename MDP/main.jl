
# =============================================================================
#  Phase 1 MDP for floating offshore wind installation: setting, states, observations, actions
#
#  Notation:
#     s_t = (t, p_t, v_t, w_t)
#     p^(i) ∈ {0,…,P}    completed 12-h work units on turbine i  (P = Σ D_k)
#     v^(j) ∈ {0} ∪ I    0 = idle, i = vessel j is mid-task on turbine i
#     w_t   ∈ {0,…,9}    WMO sea-state code
#     a^(j) ∈ {0} ∪ I    0 = do not work, i = work at turbine i
# =============================================================================

using Printf

# -----------------------------------------------------------------------------
# 1. Setting
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
    r::Float64 = 1.0                                  # revenue per turbine per step
    M::Float64 = 300.0                                # weather penalty at H_s = H^max
    C_d::Float64 = 1.0                                # cost per disrupted step
    γ::Float64 = 1.0                                  # discount (finite horizon)
    weather::DeterministicWeather
end

turbines(m::InstallSetting) = 1:m.nturb
tasks(m::InstallSetting) = eachindex(m.durations)
vessels(m::InstallSetting) = eachindex(m.vtype)
nvessels(m::InstallSetting) = length(m.vtype)

"P: total work units per turbine."
total_work(m::InstallSetting) = sum(m.durations)

"o_k: work units completed before task k starts."
offsets(m::InstallSetting) = [0; cumsum(m.durations)[1:end-1]]

"Check that the setting is well formed."
function validate(m::InstallSetting)
    @assert length(m.vname) == length(m.vtype) == length(m.capable)
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
Hook for the constraints (precedence, capability, one task per vessel,
weather limits). Returns true for now, i.e. A(s) = A.
"""
isfeasible(m::InstallSetting, s::State, a::Action) = true

"Feasible actions A(s) ⊆ A."
actions(m::InstallSetting, s::State) = filter(a -> isfeasible(m, s, a), action_space(m))





# TODO:

# Transitions and constraints

# TODO: Ryan

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

"R_weather(s, a) = Σ_j 1[a^(j) ≠ 0] (1 - (1+M)^(H_s(w) / H^max_j)): -M at the vessel limit."
function reward_weather(m::InstallSetting, s::State, a::Action)
    Hs = Hs_upper(s.w)
    return sum(1 - (1 + m.M)^(Hs / m.max_Hs[m.vtype[j]]) for j in vessels(m) if a[j] != 0; init = 0.0)
end

"R_d(s, a) = -C_d Σ_j 1[v^(j) ≠ 0 ∧ a^(j) = 0]: vessel mid-task that does not work."
reward_disruption(m::InstallSetting, s::State, a::Action) =
    -m.C_d * count(j -> s.v[j] != 0 && a[j] == 0, vessels(m))

"Total reward R(s, a)."
reward(m::InstallSetting, s::State, a::Action) =
    reward_weather(m, s, a) + reward_disruption(m, s, a) + reward_revenue(m, s)

