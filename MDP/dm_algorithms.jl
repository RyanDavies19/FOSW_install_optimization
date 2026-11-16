# =============================================================================
#  Algorithms excerpted from:
#
#    M. J. Kochenderfer, T. A. Wheeler, and K. H. Wray,
#    *Algorithms for Decision Making*, MIT Press, 2022.
#    https://algorithmsbook.com
#
#  The book is released under CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/). 
#  It also states in the preface: "Permission is granted, free of charge, to use the code snippets associated with
#  this book, subject to the condition that the source of the code is acknowledged." Thus, the code here is NOT 
#  covered by the BSD-3-Clause license that applies to the rest of this repository, instead falling under the books
#  permissions.
#
#  Specializations written for this project live in MDP/main.jl, not in this file.
#  Nothing below has been edited from the books representation.
# =============================================================================

"""
Algorithm 7.1 — MDP

Data structure for a Markov decision process. Chapter 7 (Exact Solution Methods), p. 134.
"""
struct MDP
    γ  # discount factor
    𝒮  # state space
    𝒜  # action space
    T  # transition function
    R  # reward function
    TR # sample transition and reward
end

"""
Algorithm 7.2 — lookahead

The lookahead state-action value from state `s` taking action `a`, given a value function
estimate `U`. Chapter 7, p. 137.
"""
function lookahead(𝒫::MDP, U, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U(s′) for s′ in 𝒮)
end
function lookahead(𝒫::MDP, U::Vector, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U[i] for (i,s′) in enumerate(𝒮))
end

"""
Algorithm 7.5 — ValueFunctionPolicy, greedy

A policy extracted greedily from a value function `U`. Chapter 7, p. 140.
"""
struct ValueFunctionPolicy
    𝒫 # problem
    U # utility function
end

function greedy(𝒫::MDP, U, s)
    u, ai = findmax(a->lookahead(𝒫, U, s, a), 𝒫.𝒜)
    return (a=𝒫.𝒜[ai], u=u)
end

(π::ValueFunctionPolicy)(s) = greedy(π.𝒫, π.U, s).a

"""
Algorithm 7.7 — backup

The backup procedure, which improves a value function `U` at state `s`. Chapter 7, p. 141.
"""
function backup(𝒫::MDP, U, s)
    return maximum(lookahead(𝒫, U, s, a) for a in 𝒫.𝒜)
end
