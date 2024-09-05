module EnergySamplers
using CategoricalArrays
using Distributions
using Flux
using Flux.Optimise: apply!, Optimiser
using MLUtils
using Optimisers
using Tables

"Base type for sampling rules."
abstract type AbstractSamplingRule <: Optimisers.AbstractRule end

"Base type for samplers."
abstract type AbstractSampler end

export AbstractSampler, AbstractSamplingRule
export ConditionalSampler, UnconditionalSampler, JointSampler
export PMC
export energy

include("utils.jl")
include("optimizers.jl")

"""
    (sampler::AbstractSampler)(
        model,
        rule::AbstractSamplingRule;
        niter::Int = 100,
        clip_grads::Union{Nothing,AbstractFloat} = 1e-2,
        n_samples::Union{Nothing,Int} = nothing,
        kwargs...,
    )

Sampling method for `AbstractSampler`. This method generates samples from the model's learned distribution. 

# Arguments

- `sampler::AbstractSampler`: The sampler to use.
- `model`: The model to sample from.
- `rule::AbstractSamplingRule`: The sampling rule to use.
- `niter::Int=100`: The number of iterations to perform.
- `clip_grads::Union{Nothing,AbstractFloat}=nothing`: The value to clip the gradients. This is useful to prevent exploding gradients when training joint energy models. If `nothing`, no clipping is performed.
- `n_samples::Union{Nothing,Int}=nothing`: The number of samples to generate.
- `kwargs...`: Additional keyword arguments.

# Returns

- `input_samples`: The samples generated by the sampler.
"""
function (sampler::AbstractSampler)(
    model,
    rule::AbstractSamplingRule;
    niter::Int=100,
    clip_grads::Union{Nothing,AbstractFloat}=1e-2,
    n_samples::Union{Nothing,Int}=nothing,
    kwargs...,
)
    n_samples = isnothing(n_samples) ? sampler.batch_size : n_samples

    # Initialize chain:
    reinit = rand(Binomial(1, (1.0 - sampler.prob_buffer)))
    if reinit == 1
        # Initialize chain from random samples:
        input_samples = Float32.(rand(sampler.𝒟x, sampler.input_size..., n_samples))
    else
        # Initialize chain from buffer:
        input_samples = selectdim(
            sampler.buffer,
            ndims(sampler.buffer),
            rand(1:size(sampler.buffer, ndims(sampler.buffer)), n_samples),
        )
    end

    # Perform MCMC sampling:
    rule = if isnothing(clip_grads)
        rule
    else
        Optimisers.OptimiserChain(Optimisers.ClipGrad(clip_grads), rule)
    end
    Flux.testmode!(model)
    input_samples = mcmc_samples(
        sampler, model, rule, input_samples; niter=niter, kwargs...
    )
    Flux.trainmode!(model)

    # Update buffer:
    sampler.buffer = cat(input_samples, sampler.buffer; dims=ndims(sampler.buffer))
    _end = minimum([size(sampler.buffer, ndims(sampler.buffer)), sampler.max_len])
    sampler.buffer = selectdim(sampler.buffer, ndims(sampler.buffer), 1:_end)

    return input_samples
end

"""
    PMC(
        sampler::AbstractSampler,
        model,
        rule::AbstractSamplingRule;
        ntransitions::Int = 100,
        niter::Int = 100,
        kwargs...,
    )

Runs a Persistent Markov Chain (PMC) using the sampler and model. Persistent Markov Chains are used, for example, for Persistent Contrastive Convergence ([Tieleman (2008)](https://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf)), a variant of the Contrastive Divergence (CD) algorithm. The main difference is that PCD uses a persistent chain to estimate the negative phase of the gradient. This is done by keeping the state of the Markov chain between iterations. 

In our context, the sampler is the persistent chain and the model is a supervised model. The sampler generates samples from the model's learned distribution. 

# Note

This function does not perform any training. It only generates samples from the model. In other words, there is no Contrastive Divergence. For training Joint Energy Models, see [JointEnergyModels.jl](https://github.com/JuliaTrustworthyAI/JointEnergyModels.jl).

# Arguments

- `sampler::AbstractSampler`: The sampler to use.
- `model`: The model to sample from.
- `rule::AbstractSamplingRule`: The sampling rule to use.
- `ntransitions::Int=100`: The number of transitions to perform.
- `niter::Int=100`: The number of iterations to perform.
- `kwargs...`: Additional keyword arguments.

# Returns

- `sampler.buffer`: The buffer containing the samples generated by the sampler.
"""
function PMC(
    sampler::AbstractSampler,
    model,
    rule::AbstractSamplingRule;
    ntransitions::Int=100,
    niter::Int=100,
    kwargs...,
)
    i = 1
    while i < ntransitions
        sampler(model, rule; niter=niter, kwargs...)
        i += 1
    end
    return sampler.buffer
end

@doc raw"""
    ConditionalSampler <: AbstractSampler

Generates conditional samples: $x \sim p(x|y).$
"""
mutable struct ConditionalSampler <: AbstractSampler
    𝒟x::Distribution
    𝒟y::Distribution
    input_size::Dims
    batch_size::Int
    buffer::AbstractArray
    max_len::Int
    prob_buffer::AbstractFloat
end

"""
    ConditionalSampler(
        𝒟x::Distribution, 𝒟y::Distribution;
        input_size::Dims, batch_size::Int,
        max_len::Int=10000, prob_buffer::AbstractFloat=0.95
    )

Outer constructor for `ConditionalSampler`.
"""
function ConditionalSampler(
    𝒟x::Distribution,
    𝒟y::Distribution;
    input_size::Dims,
    batch_size::Int=1,
    max_len::Int=10000,
    prob_buffer::AbstractFloat=0.95,
)
    @assert batch_size <= max_len "batch_size must be <= max_len"
    buffer = Float32.(rand(𝒟x, input_size..., batch_size))
    return ConditionalSampler(𝒟x, 𝒟y, input_size, batch_size, buffer, max_len, prob_buffer)
end

"""
    energy(sampler::ConditionalSampler, model, x, y)

Energy function for `ConditionalSampler`.
"""
function energy(sampler::ConditionalSampler, model, x, y; agg=mean)
    return _energy(model, x, y; agg=agg)
end

"""
    mcmc_samples(
        sampler::ConditionalSampler,
        model,
        rule::Optimisers.AbstractRule,
        input_samples::AbstractArray;
        niter::Int,
        y::Union{Nothing,Int} = nothing,
    )

Sampling method for `ConditionalSampler`.
"""
function mcmc_samples(
    sampler::ConditionalSampler,
    model,
    rule::Optimisers.AbstractRule,
    input_samples::AbstractArray;
    niter::Int,
    y::Union{Nothing,Int}=nothing,
)
    # Setup
    if isnothing(y)
        y = rand(sampler.𝒟y)
    end
    mod = (inputs=input_samples, energy=energy)
    s = Optimisers.setup(rule, mod)
    ntotal = size(input_samples, ndims(input_samples))
    dl = DataLoader((1:ntotal,); batchsize=sampler.batch_size)

    # Training:
    i = 1
    while i <= niter
        for (i,) in dl
            grad = gradient(mod) do m  # calculate the gradients
                m.energy(sampler, model, m.inputs[:, i], y)
            end
            s, mod = Optimisers.update(s, mod, grad[1])
        end
        i += 1
    end

    return mod.inputs
end

@doc raw"""
    UnonditionalSampler <: AbstractSampler

Generates unconditional samples: $x \sim p(x).$
"""
mutable struct UnconditionalSampler <: AbstractSampler
    𝒟x::Distribution
    𝒟y::Union{Distribution,Nothing}
    input_size::Dims
    batch_size::Int
    buffer::AbstractArray
    max_len::Int
    prob_buffer::AbstractFloat
end

"""
    UnconditionalSampler(
        𝒟x::Distribution,
        𝒟y::Union{Distribution,Nothing};
        input_size::Dims,
        batch_size::Int = 1,
        max_len::Int = 10000,
        prob_buffer::AbstractFloat = 0.95,
    )

Outer constructor for `UnonditionalSampler`.
"""
function UnconditionalSampler(
    𝒟x::Distribution,
    𝒟y::Union{Distribution,Nothing};
    input_size::Dims,
    batch_size::Int=1,
    max_len::Int=10000,
    prob_buffer::AbstractFloat=0.95,
)
    @assert batch_size <= max_len "batch_size must be <= max_len"
    buffer = Float32.(rand(𝒟x, input_size..., batch_size))
    return UnconditionalSampler(
        𝒟x, 𝒟y, input_size, batch_size, buffer, max_len, prob_buffer
    )
end

"""
    energy(sampler::UnconditionalSampler, model, x, y)

Energy function for `UnconditionalSampler`.
"""
function energy(sampler::UnconditionalSampler, model, x, y)
    return _energy(model, x; agg=mean)
end

"""
    mcmc_samples(
        sampler::UnconditionalSampler,
        model,
        rule::Optimisers.AbstractRule,
        input_samples::AbstractArray;
        niter::Int,
        y::Union{Nothing,Int} = nothing,
    )

Sampling method for `UnconditionalSampler`.
"""
function mcmc_samples(
    sampler::UnconditionalSampler,
    model,
    rule::Optimisers.AbstractRule,
    input_samples::AbstractArray;
    niter::Int,
    y::Union{Nothing,Int}=nothing,
)

    # Setup:
    mod = (inputs=input_samples, energy=energy)
    s = Optimisers.setup(rule, mod)
    ntotal = size(input_samples, ndims(input_samples))
    dl = DataLoader((1:ntotal,); batchsize=sampler.batch_size)

    # Training:
    i = 1
    while i <= niter
        for (i,) in dl
            grad = gradient(mod) do m  # calculate the gradients
                m.energy(sampler, model, m.inputs[:, i], nothing)
            end
            s, mod = Optimisers.update(s, mod, grad[1])
        end
        i += 1
    end

    return mod.inputs
end

@doc raw"""
    JointSampler <: AbstractSampler

Generates unconditional samples by drawing directly from joint distribution: $x \sim p(x, y).$
"""
mutable struct JointSampler <: AbstractSampler
    𝒟x::Distribution
    𝒟y::Distribution
    input_size::Dims
    batch_size::Int
    buffer::AbstractArray
    max_len::Int
    prob_buffer::AbstractFloat
end

"""
    JointSampler(
        𝒟x::Distribution, 𝒟y::Distribution, input_size::Dims, batch_size::Int;
        max_len::Int=10000, prob_buffer::AbstractFloat=0.95
    )

Outer constructor for `JointSampler`.
"""
function JointSampler(
    𝒟x::Distribution,
    𝒟y::Distribution;
    input_size::Dims,
    batch_size::Int=1,
    max_len::Int=10000,
    prob_buffer::AbstractFloat=0.95,
)
    @assert batch_size <= max_len "batch_size must be <= max_len"
    buffer = Float32.(rand(𝒟x, input_size..., batch_size))
    return JointSampler(𝒟x, 𝒟y, input_size, batch_size, buffer, max_len, prob_buffer)
end

"""
    energy(sampler::JointSampler, model, x, y)

Energy function for `JointSampler`.
"""
function energy(sampler::JointSampler, model, x, y)
    return _energy(model, x, y)
end

"""
    mcmc_samples(
        sampler::JointSampler,
        model,
        rule::Optimisers.AbstractRule,
        input_samples::AbstractArray;
        niter::Int,
        y::Union{Nothing,Int} = nothing,
    )

Sampling method for `JointSampler`.
"""
function mcmc_samples(
    sampler::JointSampler,
    model,
    rule::Optimisers.AbstractRule,
    input_samples::AbstractArray;
    niter::Int,
    y::Union{Nothing,Int}=nothing,
)

    # Setup:
    mod = (inputs=input_samples, energy=energy)
    s = Optimisers.setup(rule, mod)
    ntotal = size(input_samples, ndims(input_samples))
    dl = DataLoader((1:ntotal,); batchsize=sampler.batch_size)

    # Training:
    i = 1
    while i <= niter
        y = rand(sampler.𝒟y)
        for (i,) in dl
            grad = gradient(mod) do m  # calculate the gradients
                m.energy(sampler, model, m.inputs[:, i], y)
            end
            s, mod = Optimisers.update(s, mod, grad[1])
        end
        i += 1
    end

    return mod.inputs
end

end
