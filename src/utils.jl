using Flux
using StatsBase

"""
    get_logits(f::Flux.Chain, x)

Retrieves the logits (linear predictions) of a `Chain` for the input `x`.
"""
get_logits(f::Flux.Chain, x) = f[end] isa Function ? f[1:(end - 1)](x) : f(x)

@doc raw"""
    _energy(f, x; agg=mean)

Computes the energy for unconditional samples $x \sim p_{\theta}(x)$: $E(x)=-\text{LogSumExp}_y f_{\theta}(x)[y]$. $DOC_Grathwohl
"""
function _energy(f, x; agg=mean)
    if f isa Flux.Chain
        ŷ = get_logits(f, x)
    else
        ŷ = f(x)
    end
    if ndims(ŷ) > 1
        E = 0.0
        E = agg(map(y -> -logsumexp(y), eachslice(ŷ; dims=ndims(ŷ))))
        return E
    else
        return -logsumexp(ŷ)
    end
end

@doc raw"""
    _energy(f, x, y::Int; agg=mean)

Computes the energy for conditional samples $x \sim p_{\theta}(x|y)$: $E(x)=- f_{\theta}(x)[y]$. $DOC_Grathwohl
"""
function _energy(f, x, y::Int; agg=mean)
    if f isa Flux.Chain
        ŷ = get_logits(f, x)
    else
        ŷ = f(x)
    end
    _E(y, idx) = length(y) > 1 ? -y[idx] : (idx == 2 ? -y[1] : -(1.0 - y[1]))
    if ndims(ŷ) > 1
        E = 0.0
        E = agg(map(_y -> _E(_y, y), eachslice(ŷ; dims=ndims(ŷ))))
        return E
    else
        return _E(ŷ, y)
    end
end

@doc raw"""
    energy_differential(f, xgen, xsampled, y::Int; agg=mean)

Computes the energy differential between a conditional sample ``x_{\text{gen}} \sim p_{\theta}(x|y)`` and an observed sample ``x_{\text{sample}} \sim p(x|y)`` as ``E(x_{\text{sample}}|y) - E(x_{\text{gen}}|y)`` with ``E(x|y) = -f_{\theta}(x)[y]``. $DOC_Grathwohl
"""
function energy_differential(f, xgen, xsampled, y::Int; agg=mean)
    neg_loss = _energy(f, xgen, y; agg=agg)         # negative loss associated with generated samples
    pos_loss = _energy(f, xsampled, y; agg=agg)     # positive loss associated with sampled samples
    ℓ = pos_loss - neg_loss
    return ℓ
end

@doc raw"""
    energy_penalty(f, xgen, xsampled, y::Int; agg=mean)

Computes the a Ridge penalty for the overall energies of the conditional samples ``x_{\text{gen}} \sim p_{\theta}(x|y)`` and an observed sample ``x_{\text{sample}} \sim p(x|y)``. $DOC_Grathwohl
"""
function energy_penalty(f, xgen, xsampled, y::Int; agg=mean, p=1)
    neg_loss = _energy(f, xgen, y; agg=agg)         # negative loss associated with generated samples
    pos_loss = _energy(f, xsampled, y; agg=agg)     # positive loss associated with sampled samples
    ℓ = neg_loss .^ 2 .+ pos_loss .^ 2
    return ℓ
end
