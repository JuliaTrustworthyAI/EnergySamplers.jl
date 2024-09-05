using Optimisers

@doc raw"""
    SGLD(a::Real=1.0, b::Real=1.0, γ::Real=0.5)

Stochastic Gradient Langevin Dynamics ([SGLD](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)) optimizer.

# Examples
```julia
opt = SGLD()
opt = SGLD(2.0, 100.0, 0.9)
```
"""
struct SGLD <: AbstractSamplingRule
    a::Float64
    b::Float64
    gamma::Float64
end
SGLD(; a::Real = 10.0, b::Real = 1000.0, γ::Real = 0.9) = SGLD(a, b, γ)

function Optimisers.apply!(o::SGLD, state, x::AbstractArray{T}, Δ) where {T}

    a, b, γ = T(o.a), T(o.b), T(o.gamma)

    εt = @.(a * (b + state)^-γ)
    ηt = εt .* T.(randn(size(Δ)))

    Δ = T.(@.(0.5εt * Δ + ηt))

    state += 1

    return state, Δ
end

Optimisers.init(o::SGLD, x::AbstractArray) = 1

@doc raw"""
    ImproperSGLD(α::Real=2.0, σ::Real=0.01)

Improper [SGLD](https://openreview.net/pdf?id=Hkxzx0NtDB) optimizer.

# Examples
```julia
opt = ImproperSGLD()
```
"""
struct ImproperSGLD <: AbstractSamplingRule
    alpha::Float64
    sigma::Float64
end
ImproperSGLD(; α::Real = 2.0, σ::Real = 0.01) = ImproperSGLD(α, σ)

function Optimisers.apply!(o::ImproperSGLD, state, x::AbstractArray{T}, Δ) where {T}
    α, σ = T(o.alpha), T(o.sigma)

    ηt = σ .* T.(randn(size(Δ)))
    Δ = T.(@.(0.5 * α * Δ + ηt))

    return state, Δ
end

Optimisers.init(o::ImproperSGLD, x::AbstractArray) = nothing
