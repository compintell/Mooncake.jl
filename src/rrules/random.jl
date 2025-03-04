const KnownRNGs = Union{MersenneTwister, RandomDevice, TaskLocalRNG, Xoshiro}
const SpecialisedRNGs = Union{MersenneTwister, TaskLocalRNG, Xoshiro}

@zero_adjoint MinimalCtx Tuple{typeof(randn), KnownRNGs}
@zero_adjoint MinimalCtx Tuple{typeof(randexp), KnownRNGs}
@zero_adjoint MinimalCtx Tuple{typeof(randn), KnownRNGs, Type{<:IEEEFloat}}
@zero_adjoint MinimalCtx Tuple{typeof(randexp), KnownRNGs, Type{<:IEEEFloat}}

# Needed to match specialised method in Random.jl.
@is_primitive MinimalCtx Tuple{typeof(randn!), SpecialisedRNGs, Array{Float64}}
function rrule!!(
    ::CoDual{typeof(randn!)}, rng::CoDual{<:SpecialisedRNGs}, x::CoDual{<:Array{Float64}}
)
    current_x = copy(x.x)
    current_dx = copy(x.dx)
    randn!(rng.x, x.x)
    x.dx .= 0.0
    function randn!_adjoint(::NoRData)
        x.x .= current_x
        x.dx .= current_dx
        return NoRData(), NoRData(), NoRData()
    end
    return x, randn!_adjoint
end

# Needed to match specialised method in Random.jl.
@is_primitive MinimalCtx Tuple{typeof(randexp!), SpecialisedRNGs, Array{Float64}}
function rrule!!(
    ::CoDual{typeof(randexp!)}, rng::CoDual{<:SpecialisedRNGs}, x::CoDual{<:Array{Float64}}
)
    current_x = copy(x.x)
    current_dx = copy(x.dx)
    randexp!(rng.x, x.x)
    x.dx .= 0.0
    function randexp!_adjoint(::NoRData)
        x.x .= current_x
        x.dx .= current_dx
        return NoRData(), NoRData(), NoRData()
    end
    return x, randexp!_adjoint
end

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

        (true, :stability, nothing, randn!, MersenneTwister(123), randn(5)),
        (true, :stability, nothing, randexp!, MersenneTwister(123), randn(5)),
        (true, :stability, nothing, randn!, TaskLocalRNG(), randn(5)),
        (true, :stability, nothing, randexp!, TaskLocalRNG(), randn(5)),
        (true, :stability, nothing, randn!, Xoshiro(123), randn(5)),
        (true, :stability, nothing, randexp!, Xoshiro(123), randn(5)),
    ]
    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:random})
    return Any[], Any[]
end
