
struct New{T} end

@generated (::New{T})(x...) where {T} = Expr(:new, T, map(n -> :(x[$n]), 1:length(x))...)

tangent_type(::Type{<:New}) = NoTangent

@is_primitive MinimalCtx Tuple{New, Vararg}
Umlaut.isprimitive(::RMC, ::New, args...) = true

@generated function New_pullback(dy, d__new__, dxs::Vararg{Any, N}) where {N}
    inc_exprs = map(n -> :(increment!!(dxs[$n], _value(fs[$n]))), 1:N)
    return quote
        fs = dy.fields
        return $(Expr(:tuple, :d__new__, inc_exprs...))
    end
end

@generated function New_pullback(
    dy::Union{Tuple, NamedTuple}, d__new__, dxs::Vararg{Any, N}
) where {N}
    inc_exprs = map(n -> :(increment!!(dxs[$n], dy[$n])), 1:N)
    return quote
        return $(Expr(:tuple, :d__new__, inc_exprs...))
    end
end

@generated function rrule!!(::CoDual{New{P}}, xs::Vararg{Any, N}) where {P, N}
    return quote
        x_ps = map(primal, xs)
        y = $(Expr(:new, P, map(n -> :(x_ps[$n]), 1:N)...))
        dy = build_tangent(P, map(tangent, xs)...)
        return CoDual(y, dy), New_pullback
    end
end

for N in 0:32
    @eval @inline function _new_(::Type{T}, x::Vararg{Any, $N}) where {T}
        return $(Expr(:new, :T, map(n -> :(x[$n]), 1:N)...))
    end
    @eval function _new_pullback!!(dy, d_new_, d_T, dx::Vararg{Any, $N})
        return d_new_, d_T, map((x, y) -> increment!!(x, _value(y)), dx, dy.fields)...
    end
    @eval function _new_pullback!!(
        dy::Union{Tuple, NamedTuple}, d_new_, d_T, dx::Vararg{Any, $N}
    )
        return d_new_, d_T, map(increment!!, dx, dy)...
    end
    @eval function rrule!!(
        ::CoDual{typeof(_new_)}, ::CoDual{Type{P}}, x::Vararg{CoDual, $N}
    ) where {P}
        y = $(Expr(:new, :P, map(n -> :(primal(x[$n])), 1:N)...))
        dy = build_tangent(P, tuple_map(tangent, x)...)
        return CoDual(y, dy), _new_pullback!!
    end
end
@is_primitive MinimalCtx Tuple{typeof(_new_), Vararg}

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:new})
    test_cases = Any[
        (false, :stability, nothing, New{Tuple{Float64, Int}}(), 5.0, 4),
        (false, :stability, nothing, New{Tuple{Float64, Float64}}(), 5.0, 4.0),
        (
            false, :stability, nothing,
            New{TestResources.TypeStableStruct{Float64}}(), 5, 4.0,
        ),
        (
            false, :stability, nothing,
            New{TestResources.TypeStableMutableStruct{Float64}}(), 5.0, 4.0,
        ),
        (
            false, :none, nothing,
            New{TestResources.TypeStableMutableStruct{Any}}(), 5.0, 4.0,
        ),
    ]
    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:new})
    test_cases = Any[]
    memory = Any[]
    return test_cases, memory
end