for N in 0:32
    @eval @inline function _new_(::Type{T}, x::Vararg{Any, $N}) where {T}
        return $(Expr(:new, :T, map(n -> :(x[$n]), 1:N)...))
    end
    @eval function _new_pullback!!(dy, d_new_, d_T, dx::Vararg{Any, $N})
        return d_new_, d_T, map((x, y) -> increment!!(x, _value(y)), dx, Tuple(dy.fields))...
    end
    @eval function _new_pullback!!(
        dy::Union{Tuple, NamedTuple}, d_new_, d_T, dx::Vararg{Any, $N}
    )
        return d_new_, d_T, map(increment!!, dx, Tuple(dy))...
    end
    @eval function _new_pullback!!(::NoTangent, d_new_, d_T, dx::Vararg{Any, $N})
        return d_new_, NoTangent(), dx...
    end
    @eval function rrule!!(
        ::CoDual{typeof(_new_)}, ::CoDual{Type{P}}, x::Vararg{CoDual, $N}
    ) where {P}
        y = $(Expr(:new, :P, map(n -> :(primal(x[$n])), 1:N)...))
        T = tangent_type(P)
        dy = T == NoTangent ? NoTangent() : build_tangent(P, tuple_map(tangent, x)...)
        return CoDual(y, dy), _new_pullback!!
    end
end

@is_primitive MinimalCtx Tuple{typeof(_new_), Vararg}

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:new})
    test_cases = Any[
        (false, :stability_and_allocs, nothing, _new_, Tuple{}),
        (false, :stability_and_allocs, nothing, _new_, Tuple{Float64, Int}, 5.0, 4),
        (false, :stability_and_allocs, nothing, _new_, Tuple{Float64, Float64}, 5.0, 4.0),
        (false, :stability_and_allocs, nothing, _new_, Tuple{Int, Int}, 5, 5),
        (false, :stability_and_allocs, nothing, _new_, @NamedTuple{}),
        (false, :stability_and_allocs, nothing, _new_, @NamedTuple{y::Float64}, 5.0),
        (
            false, :stability_and_allocs, nothing,
            _new_, @NamedTuple{y::Float64, x::Int}, 5.0, 4,
        ),
        (false, :stability_and_allocs, nothing, _new_, @NamedTuple{y::Int, x::Int}, 5, 4),
        (
            false, :stability_and_allocs, nothing,
            _new_, TestResources.TypeStableStruct{Float64}, 5, 4.0,
        ),
        (
            false, :stability_and_allocs, nothing,
            _new_, TestResources.TypeStableMutableStruct{Float64}, 5.0, 4.0,
        ),
        (
            false, :none, nothing,
            _new_, TestResources.TypeStableMutableStruct{Any}, 5.0, 4.0,
        ),
        (false, :stability_and_allocs, nothing, _new_, UnitRange{Int64}, 5, 4),
    ]
    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:new})
    test_cases = Any[]
    memory = Any[]
    return test_cases, memory
end