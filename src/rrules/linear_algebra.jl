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
    N = P(2) / P(length(X.x)) .* (X.x - Y.x)

    function FluxMSE_pullback(dloss::P)
        # adjoints got by VJP reverse pass equations.
        temp = N .* dloss
        @. X.dx += temp
        @. Y.dx -= temp
        return NoRData(), NoRData(), NoRData()
    end

    # in forward pass it returns codual float64, hence coduals dx is NoFData().
    return CoDual(Losses.mse(X.x, Y.x), NoFData()), FluxMSE_pullback
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:linear_algebra})
    # Testing specific floating point precisions as FDM is unstable for certain tests.
    # Larger differences can compare well with AD for higher precisions as FDM is relatively stable.
    # for smaller differences we get worse FDM gradients for all floating points as FDM error and instability scales
    #  first test case success starts at a difference of ~ 1e4

    test_cases = vcat(
        map([Float64]) do P
            return (
                false, :none, nothing, Losses.mse, P.([1e1, 1e2, 1e3]), P.([1e3, 1e4, 0])
            )
        end,
        map([Float16, Float32, Float64]) do P
            return (
                true,
                :none,
                nothing,
                Losses.mse,
                P.([1e-3, 1e-4, 0]),
                P.([1e-1, 1e-2, 1e-3]),
            )
        end,
        (true, :none, nothing, Losses.mse, Float64.([13, 13, 13]), Float64.([13, 13, 13])),
    )

    memory = Any[]
    return test_cases, memory
end
