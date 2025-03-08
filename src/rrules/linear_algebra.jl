@is_primitive MinimalCtx Tuple{typeof(exp),Matrix{<:IEEEFloat}}

struct ExpPullback{P}
    pb
    Ȳ::Matrix{P}
    X̄::Matrix{P}
end

function (pb::ExpPullback)(::NoRData)
    _, X̄_inc = pb.pb(pb.Ȳ)
    pb.X̄ .+= X̄_inc
    return NoRData(), NoRData()
end

function rrule!!(::CoDual{typeof(exp)}, X::CoDual{Matrix{P}}) where {P<:IEEEFloat}
    Y, pb = ChainRules.rrule(exp, X.x)
    Ȳ = zero(Y)
    return CoDual(Y, Ȳ), ExpPullback{P}(pb, Ȳ, X.dx)
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:linear_algebra})
    rng = rng_ctor(123)
    Ps = [Float64, Float32]
    test_cases = vcat(
        map_prod([3, 7], Ps) do (N, P)
            return (false, :none, nothing, exp, randn(rng, P, N, N))
        end,
    )
    memory = Any[]
    return test_cases, memory
end

@is_primitive MinimalCtx Tuple{
    typeof(Flux.Losses.mse),Matrix{<:IEEEFloat},Matrix{<:IEEEFloat}
}

function rrule!!(
    ::CoDual{typeof(Flux.Losses.mse)},
    X::CoDual{<:AbstractMatrix{P}},
    Y::CoDual{<:AbstractMatrix{P}},
) where {P<:IEEEFloat}
    N = 2.0 / length(X.x)
    function FluxMSE_adjoint(dloss::P)
        # adjoints got by VJP reverse pass equations.
        X.dx .+= N * dloss .* (X.x - Y.x)
        Y.dx .+= -1.0 .* X.dx
        return NoRData(), NoRData(), NoRData()
    end
    # in forward pass it returns codual float64, hence coduals dx is nofdata().
    return CoDual(Flux.Losses.mse(X.x, Y.x), NoFData()), FluxMSE_adjoint
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:linear_algebra})
    return Any[], Any[]
end
