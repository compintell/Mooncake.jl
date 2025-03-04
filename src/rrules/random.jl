const KnownRNGs = Union{MersenneTwister, RandomDevice, TaskLocalRNG, Xoshiro}

@zero_adjoint MinimalCtx Tuple{typeof(randn), KnownRNGs}
@zero_adjoint MinimalCtx Tuple{typeof(randexp), KnownRNGs}
@zero_adjoint MinimalCtx Tuple{typeof(randn), KnownRNGs, Type{<:IEEEFloat}}
@zero_adjoint MinimalCtx Tuple{typeof(randexp), KnownRNGs, Type{<:IEEEFloat}}

# These cannot be zero -- we need to write proper rules for this.
@zero_adjoint MinimalCtx Tuple{typeof(randn!), MersenneTwister, Array{Float64}}
@zero_adjoint MinimalCtx Tuple{typeof(randexp!), MersenneTwister, Array{Float64}}
@zero_adjoint MinimalCtx Tuple{typeof(randn!), Union{Xoshiro, TaskLocalRNG}, Array{Float64}}
@zero_adjoint MinimalCtx Tuple{typeof(randexp!), Union{Xoshiro, TaskLocalRNG}, Array{Float64}}

_rngs() = [MersenneTwister(123), RandomDevice(), TaskLocalRNG(123), Xoshiro(123)]
__rngs() = [MersenneTwister(123), TaskLocalRNG(123), Xoshiro(123)]

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:random})
    rng = rng_ctor(123)
    Ps = [Float64, Float32]
    test_cases = Any[
        (true, :stability_and_allocs, nothing, randn, Xoshiro(123)),
        (true, :stability_and_allocs, nothing, randn, Xoshiro(123), Float32),
        (true, :stability_and_allocs, nothing, randn, Xoshiro(123), Float64),

        (true, :stability_and_allocs, nothing, randexp, Xoshiro(123)),
        (true, :stability_and_allocs, nothing, randexp, Xoshiro(123), Float32),
        (true, :stability_and_allocs, nothing, randexp, Xoshiro(123), Float64),

        (true, :stability_and_allocs, nothing, randn!, MersenneTwister(123), randn(5)),
        (true, :stability_and_allocs, nothing, randexp!, MersenneTwister(123), randn(5)),
        (true, :stability_and_allocs, nothing, randn!, TaskLocalRNG(), randn(5)),
        (true, :stability_and_allocs, nothing, randexp!, TaskLocalRNG(), randn(5)),
        (true, :stability_and_allocs, nothing, randn!, Xoshiro(123), randn(5)),
        (true, :stability_and_allocs, nothing, randexp!, Xoshiro(123), randn(5)),
    ]
    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:random})
    return Any[], Any[]
end
