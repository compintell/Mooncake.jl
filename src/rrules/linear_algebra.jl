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

@is_primitive MinimalCtx Tuple{typeof(Losses.mse),Matrix{<:IEEEFloat},Matrix{<:IEEEFloat}}

function rrule!!(
    ::CoDual{typeof(Losses.mse)},
    X::CoDual{<:AbstractMatrix{P}},
    Y::CoDual{<:AbstractMatrix{P}},
) where {P<:IEEEFloat}
    N = 2.0 / length(X.x)
    function FluxMSE_pullback(dloss::P)
        # adjoints got by VJP reverse pass equations.
        X.dx .+= N * dloss .* (X.x - Y.x)
        Y.dx .+= -1.0 .* X.dx
        return NoRData(), NoRData(), NoRData()
    end
    # in forward pass it returns codual float64, hence coduals dx is NoFData().
    return CoDual(Losses.mse(X.x, Y.x), NoFData()), FluxMSE_pullback
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:linear_algebra})
    test_cases = reduce(
        vcat,
        map([Float16, Float32, Float64]) do P
            return Any[
                (
                    false,
                    :stability_and_allocs,
                    nothing,
                    Losses.mse,
                    P.([1, 1, 1]),
                    P.([1, 1, 1]),
                ),
                (
                    false,
                    :stability_and_allocs,
                    nothing,
                    Losses.mse,
                    P.([1e1, 1e2, 1e3]),
                    P.([1e3, 1e4, 1e5]),
                ),
            ]
        end,
    )
    memory = Any[]
    return test_cases, memory
end
