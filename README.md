

# EnergySamplers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaTrustworthyAI.github.io/EnergySamplers.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaTrustworthyAI.github.io/EnergySamplers.jl/dev/) [![Build Status](https://github.com/JuliaTrustworthyAI/EnergySamplers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaTrustworthyAI/EnergySamplers.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/JuliaTrustworthyAI/EnergySamplers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaTrustworthyAI/EnergySamplers.jl) [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

\[32m\[1m Activating\[22m\[39m project at `~/code/EnergySamplers.jl/docs`

`EnergySamplers.jl` is a small and lightweight package for sampling from probability distributions using methods from energy-based modelling (EBM). Its functionality is used in other \[Taija\] packages, including [JointEnergyModels.jl](https://github.com/JuliaTrustworthyAI/JointEnergyModels.jl) and [CounterfactualExplanations.jl](https://github.com/JuliaTrustworthyAI/CounterfactualExplanations.jl).

## Extensions to [Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/)

The package adds two new optimisers that are compatible with the [Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/) interface:

1.  Stochastic Gradient Langevin Dynamics (SGLD) \[@welling2011bayesian\] – [`SGLD`](@ref).
2.  Improper SGLD (see, for example, @grathwohl2020your) – [`ImproperSGLD`](@ref).

SGLD is an efficient gradient-based Markov Chain Monte Carlo (MCMC) method that can be used in the context of EBM to draw samples from the model posterior \[@murphy2023probabilistic\]. Formally, we can draw from $p_{\theta}(\mathbf{x})$ as follows

$$
\begin{aligned}
    \mathbf{x}_{j+1} &\leftarrow \mathbf{x}_j - \frac{\epsilon_j^2}{2} \nabla_x \mathcal{E}_{\theta}(\mathbf{x}_j) + \epsilon_j \mathbf{r}_j, && j=1,...,J
\end{aligned}
$$

where $\mathbf{r}_j \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ is a stochastic term and the step-size $\epsilon_j$ is typically polynomially decayed \[@welling2011bayesian\]. To allow for faster sampling, it is common practice to choose the step-size $\epsilon_j$ and the standard deviation of $\mathbf{r}_j$ separately. While $\mathbf{x}_J$ is only guaranteed to distribute as $p_{\theta}(\mathbf{x})$ if $\epsilon \rightarrow 0$ and $J \rightarrow \infty$, the bias introduced for a small finite $\epsilon$ is negligible in practice \[@murphy2023probabilistic\]. We denote this form of sampling as Improper SGLD.
