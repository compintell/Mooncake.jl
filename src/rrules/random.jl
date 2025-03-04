const KnownRNGs = Union{MersenneTwister,RandomDevice,TaskLocalRNG,Xoshiro}
@zero_adjoint MinimalCtx Tuple{typeof(randn),KnownRNGs}
@zero_adjoint MinimalCtx Tuple{typeof(randexp),KnownRNGs}
@zero_adjoint MinimalCtx Tuple{typeof(randn),KnownRNGs,Type{<:IEEEFloat}}
@zero_adjoint MinimalCtx Tuple{typeof(randexp),KnownRNGs,Type{<:IEEEFloat}}

const SpecialisedRNGs = Union{MersenneTwister,TaskLocalRNG,Xoshiro}
for f in [randn!, randexp!]
    @eval @is_primitive MinimalCtx Tuple{typeof($f),SpecialisedRNGs,Array{Float64}}
    @eval function rrule!!(
        ::CoDual{typeof($f)}, rng::CoDual{<:SpecialisedRNGs}, x::CoDual{<:Array{Float64}}
    )
        current_x = copy(x.x)
        current_dx = copy(x.dx)
        $f(rng.x, x.x)
        x.dx .= 0.0
        function rand_adjoint(::NoRData)
            x.x .= current_x
            x.dx .= current_dx
            return NoRData(), NoRData(), NoRData()
        end
        return x, rand_adjoint
    end
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:random})
    rngs = [MersenneTwister(123), TaskLocalRNG(), Xoshiro(123)]
    all_rngs = vcat(rngs, RandomDevice())
    test_cases = vcat(
        map_prod([randn, randexp], all_rngs) do (f, rng)
            (true, :stability_and_allocs, nothing, f, rng)
        end...,
        map_prod([Float64, Float32], [randn, randexp], all_rngs) do (P, f, rng)
            (true, :stability_and_allocs, nothing, f, rng, P)
        end...,
        map_prod([randn!, randexp!], rngs) do (f, rng)
            (true, :stability, nothing, f, rng, randn(5))
        end...,
    )
    return test_cases, Any[]
end

generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:random}) = Any[], Any[]
