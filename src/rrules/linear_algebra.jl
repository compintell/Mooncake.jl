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

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:linear_algebra})
    return Any[], Any[]
end

@is_primitive DefaultCtx Tuple{
    typeof(Losses.mse),AbstractArray{<:IEEEFloat},AbstractArray{<:IEEEFloat}
}

function rrule!!(
    ::CoDual{typeof(Losses.mse)},
    X::CoDual{<:AbstractArray{P}},
    Y::CoDual{<:AbstractArray{P}},
) where {P<:IEEEFloat}
    N = P(2) / P(length(X.x))

    function FluxMSE_pullback(dloss::P)
        # adjoints got by VJP reverse pass equations.
        @. X.dx += dloss * N * (X.x - Y.x)
        @. Y.dx -= X.dx
        return NoRData(), NoRData(), NoRData()
    end

    # in forward pass it returns codual float64, hence coduals dx is NoFData().
    return CoDual(Losses.mse(X.x, Y.x), NoFData()), FluxMSE_pullback
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:linear_algebra})
    test_cases = reduce(
        vcat,
        # wierd behaviour for Float 32,64 in _dot_internal while comparing 
        # FDM and AD results in test_rule_correctness
        map([Float16, Float32, Float64]) do P
            return Any[
                (true, :none, nothing, Losses.mse, P.([13, 13, 13]), P.([13, 13, 13])),
                (true, :none, nothing, Losses.mse, P.([1e1, 1e2, 1e3]), P.([1e3, 1e4, 0])),
                (
                    true,
                    :none,
                    nothing,
                    Losses.mse,
                    P.([1e-3, 1e-4, 0]),
                    P.([1e-1, 1e-2, 1e-3]),
                ),
            ]
        end,
    )
    memory = Any[]
    return test_cases, memory
end
