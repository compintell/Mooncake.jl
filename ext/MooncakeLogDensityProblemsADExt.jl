# This file is largely copy + pasted + modified from the Zygote extension in
# LogDensityProblemsAD.jl. 

module MooncakeLogDensityProblemsADExt

if isdefined(Base, :get_extension)
    using ADTypes
    using LogDensityProblemsAD: ADGradientWrapper
    import LogDensityProblemsAD: ADgradient, logdensity_and_gradient, dimension, logdensity
    import Mooncake
else
    using ADTypes
    using ..LogDensityProblemsAD: ADGradientWrapper
    import ..LogDensityProblemsAD: ADgradient, logdensity_and_gradient, dimension, logdensity
    import ..Mooncake
end

struct MooncakeGradientLogDensity{Trule, L} <: ADGradientWrapper
    rule::Trule
    ℓ::L
end

dimension(∇l::MooncakeGradientLogDensity) = dimension(Mooncake.primal(∇l.ℓ))

function logdensity(∇l::MooncakeGradientLogDensity, x::Vector{Float64})
    return logdensity(Mooncake.primal(∇l.ℓ), x)
end

"""
    ADgradient(Val(:Mooncake), ℓ)

Gradient using algorithmic/automatic differentiation via Mooncake.
"""
function ADgradient(::Val{:Mooncake}, ℓ; debug_mode::Bool=false, rule=nothing)
    if isnothing(rule)
        primal_sig = Tuple{typeof(logdensity), typeof(ℓ), Vector{Float64}}
        rule = Mooncake.build_rrule(Mooncake.get_interpreter(), primal_sig; debug_mode)
    end
    return MooncakeGradientLogDensity(rule, Mooncake.uninit_fcodual(ℓ))
end

function Base.show(io::IO, ∇ℓ::MooncakeGradientLogDensity)
    return print(io, "Mooncake AD wrapper for ", Mooncake.primal(∇ℓ.ℓ))
end

# We only test Mooncake with `Float64`s at the minute, so make strong assumptions about the
# types supported in order to prevent silent errors.
function logdensity_and_gradient(::MooncakeGradientLogDensity, ::AbstractVector)
    msg = "Only Vector{Float64} presently supported for logdensity_and_gradients."
    throw(ArgumentError(msg))
end

function logdensity_and_gradient(∇l::MooncakeGradientLogDensity, x::Vector{Float64})
    dx = zeros(length(x))
    y, pb!! = ∇l.rule(Mooncake.zero_fcodual(logdensity), ∇l.ℓ, Mooncake.CoDual(x, dx))
    @assert Mooncake.primal(y) isa Float64
    pb!!(1.0)
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

Base.parent(x::MooncakeGradientLogDensity) = Mooncake.primal(x.ℓ)

end
