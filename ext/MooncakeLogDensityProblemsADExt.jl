# This file is largely copy + pasted + modified from the Zygote extension in
# LogDensityProblemsAD.jl. 

module MooncakeLogDensityProblemsADExt

using ADTypes
using Base: IEEEFloat
using LogDensityProblemsAD: ADGradientWrapper
import LogDensityProblemsAD: ADgradient, logdensity_and_gradient, dimension, logdensity
import Mooncake

mutable struct MooncakeGradientLogDensity{L} <: ADGradientWrapper
    const ℓ::L
    const debug_mode::Bool
    rule
    MooncakeGradientLogDensity(ℓ::L, debug_mode::Bool) where {L} = new{L}(ℓ, debug_mode)
    function MooncakeGradientLogDensity(ℓ::L, debug_mode::Bool, rule) where {L}
        return new{L}(ℓ, debug_mode, rule)
    end
end

const MGLD{L} = MooncakeGradientLogDensity{L}

dimension(∇l::MGLD) = dimension(Mooncake.primal(∇l.ℓ))

logdensity(∇l::MGLD, x::Vector{<:IEEEFloat}) = logdensity(Mooncake.primal(∇l.ℓ), x)

"""
    ADgradient(Val(:Mooncake), ℓ)

Gradient using algorithmic/automatic differentiation via Mooncake.
"""
function ADgradient(::Val{:Mooncake}, ℓ; debug_mode::Bool=false, rule=nothing)
    l = Mooncake.uninit_fcodual(ℓ)
    return rule === nothing ? MGLD(l, debug_mode) : MGLD(l, debug_mode, rule) 
end

function Base.show(io::IO, ∇ℓ::MGLD)
    return print(io, "Mooncake AD wrapper for ", Mooncake.primal(∇ℓ.ℓ))
end

# We only permit simple types. Rule out anything that's not a `Vector{<:IEEEFloat}`.
function logdensity_and_gradient(::MGLD, ::AbstractVector)
    msg = "Only Vector{<:IEEEFloat} presently supported for logdensity_and_gradients."
    throw(ArgumentError(msg))
end

function logdensity_and_gradient(∇l::MGLD, x::Vector{P}) where {P<:IEEEFloat}
    if !isdefined(∇l, :rule)
        primal_sig = Tuple{typeof(logdensity), typeof(Mooncake.primal(∇l.ℓ)), Vector{P}}
        debug_mode = ∇l.debug_mode
        ∇l.rule = Mooncake.build_rrule(Mooncake.get_interpreter(), primal_sig; debug_mode)
    end
    dx = zero(x)
    y, pb!! = ∇l.rule(Mooncake.zero_fcodual(logdensity), ∇l.ℓ, Mooncake.CoDual(x, dx))
    @assert Mooncake.primal(y) isa P
    pb!!(one(P))
    return Mooncake.primal(y), dx
end

# Interop with ADTypes.
function getconfig(x::ADTypes.AutoMooncake)
    c = x.config
    return isnothing(c) ? Mooncake.Config() : c
end
function ADgradient(x::ADTypes.AutoMooncake, ℓ)
    debug_mode = getconfig(x).debug_mode
    if debug_mode
        msg = "Running Mooncake in debug mode. This mode is computationally expensive, " *
            "should only be used when debugging a problem with AD, and turned off in " *
            "general use. Do this by using " *
            "AutoMooncake(; config=Mooncake.Config(debug_mode=false))."
        @info msg
    end
    return ADgradient(Val(:Mooncake), ℓ; debug_mode)
end

Base.parent(x::MGLD) = Mooncake.primal(x.ℓ)

end
