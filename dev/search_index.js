var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = EnergySamplers","category":"page"},{"location":"#EnergySamplers","page":"Home","title":"EnergySamplers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for EnergySamplers.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [EnergySamplers]","category":"page"},{"location":"#EnergySamplers.AbstractSampler","page":"Home","title":"EnergySamplers.AbstractSampler","text":"Base type for samplers.\n\n\n\n\n\n","category":"type"},{"location":"#EnergySamplers.AbstractSampler-Tuple{Any, AbstractSamplingRule}","page":"Home","title":"EnergySamplers.AbstractSampler","text":"(sampler::AbstractSampler)(\n    model,\n    rule::AbstractSamplingRule;\n    niter::Int = 100,\n    clip_grads::Union{Nothing,AbstractFloat} = 1e-2,\n    n_samples::Union{Nothing,Int} = nothing,\n    kwargs...,\n)\n\nSampling method for AbstractSampler. This method generates samples from the model's learned distribution. \n\nArguments\n\nsampler::AbstractSampler: The sampler to use.\nmodel: The model to sample from.\nrule::AbstractSamplingRule: The sampling rule to use.\nniter::Int=100: The number of iterations to perform.\nclip_grads::Union{Nothing,AbstractFloat}=nothing: The value to clip the gradients. This is useful to prevent exploding gradients when training joint energy models. If nothing, no clipping is performed.\nn_samples::Union{Nothing,Int}=nothing: The number of samples to generate.\nkwargs...: Additional keyword arguments.\n\nReturns\n\ninput_samples: The samples generated by the sampler.\n\n\n\n\n\n","category":"method"},{"location":"#EnergySamplers.AbstractSamplingRule","page":"Home","title":"EnergySamplers.AbstractSamplingRule","text":"Base type for sampling rules.\n\n\n\n\n\n","category":"type"},{"location":"#EnergySamplers.ConditionalSampler","page":"Home","title":"EnergySamplers.ConditionalSampler","text":"ConditionalSampler <: AbstractSampler\n\nGenerates conditional samples: x sim p(xy)\n\n\n\n\n\n","category":"type"},{"location":"#EnergySamplers.ConditionalSampler-Tuple{Distributions.Distribution, Distributions.Distribution}","page":"Home","title":"EnergySamplers.ConditionalSampler","text":"ConditionalSampler(\n    𝒟x::Distribution, 𝒟y::Distribution;\n    input_size::Dims, batch_size::Int,\n    max_len::Int=10000, prob_buffer::AbstractFloat=0.95\n)\n\nOuter constructor for ConditionalSampler.\n\n\n\n\n\n","category":"method"},{"location":"#EnergySamplers.ImproperSGLD","page":"Home","title":"EnergySamplers.ImproperSGLD","text":"ImproperSGLD(α::Real=2.0, σ::Real=0.01)\n\nImproper SGLD optimizer.\n\nExamples\n\nopt = ImproperSGLD()\n\n\n\n\n\n","category":"type"},{"location":"#EnergySamplers.JointSampler","page":"Home","title":"EnergySamplers.JointSampler","text":"JointSampler <: AbstractSampler\n\nGenerates unconditional samples by drawing directly from joint distribution: x sim p(x y)\n\n\n\n\n\n","category":"type"},{"location":"#EnergySamplers.JointSampler-Tuple{Distributions.Distribution, Distributions.Distribution}","page":"Home","title":"EnergySamplers.JointSampler","text":"JointSampler(\n    𝒟x::Distribution, 𝒟y::Distribution, input_size::Dims, batch_size::Int;\n    max_len::Int=10000, prob_buffer::AbstractFloat=0.95\n)\n\nOuter constructor for JointSampler.\n\n\n\n\n\n","category":"method"},{"location":"#EnergySamplers.SGLD","page":"Home","title":"EnergySamplers.SGLD","text":"SGLD(a::Real=1.0, b::Real=1.0, γ::Real=0.5)\n\nStochastic Gradient Langevin Dynamics (SGLD) optimizer.\n\nExamples\n\nopt = SGLD()\nopt = SGLD(2.0, 100.0, 0.9)\n\n\n\n\n\n","category":"type"},{"location":"#EnergySamplers.UnconditionalSampler","page":"Home","title":"EnergySamplers.UnconditionalSampler","text":"UnonditionalSampler <: AbstractSampler\n\nGenerates unconditional samples: x sim p(x)\n\n\n\n\n\n","category":"type"},{"location":"#EnergySamplers.UnconditionalSampler-Tuple{Distributions.Distribution, Union{Nothing, Distributions.Distribution}}","page":"Home","title":"EnergySamplers.UnconditionalSampler","text":"UnconditionalSampler(\n    𝒟x::Distribution,\n    𝒟y::Union{Distribution,Nothing};\n    input_size::Dims,\n    batch_size::Int = 1,\n    max_len::Int = 10000,\n    prob_buffer::AbstractFloat = 0.95,\n)\n\nOuter constructor for UnonditionalSampler.\n\n\n\n\n\n","category":"method"},{"location":"#EnergySamplers.PMC-Tuple{AbstractSampler, Any, AbstractSamplingRule}","page":"Home","title":"EnergySamplers.PMC","text":"PMC(\n    sampler::AbstractSampler,\n    model,\n    rule::AbstractSamplingRule;\n    ntransitions::Int = 100,\n    niter::Int = 100,\n    kwargs...,\n)\n\nRuns a Persistent Markov Chain (PMC) using the sampler and model. Persistent Markov Chains are used, for example, for Persistent Contrastive Convergence (Tieleman (2008)), a variant of the Contrastive Divergence (CD) algorithm. The main difference is that PCD uses a persistent chain to estimate the negative phase of the gradient. This is done by keeping the state of the Markov chain between iterations. \n\nIn our context, the sampler is the persistent chain and the model is a supervised model. The sampler generates samples from the model's learned distribution. \n\nNote\n\nThis function does not perform any training. It only generates samples from the model. In other words, there is no Contrastive Divergence. For training Joint Energy Models, see JointEnergyModels.jl.\n\nArguments\n\nsampler::AbstractSampler: The sampler to use.\nmodel: The model to sample from.\nrule::AbstractSamplingRule: The sampling rule to use.\nntransitions::Int=100: The number of transitions to perform.\nniter::Int=100: The number of iterations to perform.\nkwargs...: Additional keyword arguments.\n\nReturns\n\nsampler.buffer: The buffer containing the samples generated by the sampler.\n\n\n\n\n\n","category":"method"},{"location":"#EnergySamplers._energy-Tuple{Any, Any, Int64}","page":"Home","title":"EnergySamplers._energy","text":"energy(f, x, y::Int; agg=mean)\n\nComputes the energy for conditional samples x sim p_theta(xy): E(x)=- f_theta(x)y.\n\n\n\n\n\n","category":"method"},{"location":"#EnergySamplers._energy-Tuple{Any, Any}","page":"Home","title":"EnergySamplers._energy","text":"energy(f, x)\n\nComputes the energy for unconditional samples x sim p_theta(x): E(x)=-textLogSumExp_y f_theta(x)y.\n\n\n\n\n\n","category":"method"},{"location":"#EnergySamplers.energy-Tuple{ConditionalSampler, Any, Any, Any}","page":"Home","title":"EnergySamplers.energy","text":"energy(sampler::ConditionalSampler, model, x, y)\n\nEnergy function for ConditionalSampler.\n\n\n\n\n\n","category":"method"},{"location":"#EnergySamplers.energy-Tuple{JointSampler, Any, Any, Any}","page":"Home","title":"EnergySamplers.energy","text":"energy(sampler::JointSampler, model, x, y)\n\nEnergy function for JointSampler.\n\n\n\n\n\n","category":"method"},{"location":"#EnergySamplers.energy-Tuple{UnconditionalSampler, Any, Any, Any}","page":"Home","title":"EnergySamplers.energy","text":"energy(sampler::UnconditionalSampler, model, x, y)\n\nEnergy function for UnconditionalSampler.\n\n\n\n\n\n","category":"method"},{"location":"#EnergySamplers.mcmc_samples-Tuple{ConditionalSampler, Any, Optimisers.AbstractRule, AbstractArray}","page":"Home","title":"EnergySamplers.mcmc_samples","text":"mcmc_samples(\n    sampler::ConditionalSampler,\n    model,\n    rule::Optimisers.AbstractRule,\n    input_samples::AbstractArray;\n    niter::Int,\n    y::Union{Nothing,Int} = nothing,\n)\n\nSampling method for ConditionalSampler.\n\n\n\n\n\n","category":"method"},{"location":"#EnergySamplers.mcmc_samples-Tuple{JointSampler, Any, Optimisers.AbstractRule, AbstractArray}","page":"Home","title":"EnergySamplers.mcmc_samples","text":"mcmc_samples(\n    sampler::JointSampler,\n    model,\n    rule::Optimisers.AbstractRule,\n    input_samples::AbstractArray;\n    niter::Int,\n    y::Union{Nothing,Int} = nothing,\n)\n\nSampling method for JointSampler.\n\n\n\n\n\n","category":"method"},{"location":"#EnergySamplers.mcmc_samples-Tuple{UnconditionalSampler, Any, Optimisers.AbstractRule, AbstractArray}","page":"Home","title":"EnergySamplers.mcmc_samples","text":"mcmc_samples(\n    sampler::UnconditionalSampler,\n    model,\n    rule::Optimisers.AbstractRule,\n    input_samples::AbstractArray;\n    niter::Int,\n    y::Union{Nothing,Int} = nothing,\n)\n\nSampling method for UnconditionalSampler.\n\n\n\n\n\n","category":"method"}]
}
