#
# Performance-only rules. These should be able to be removed, and everything still works,
# just a bit slower. The effect of these is typically to remove many nodes from the tape.
# Ideally, it would be the case that acitivty analysis eliminates any run-time improvements
# that these rules provide. Possibly they would still be useful in order to avoid having to
# deduce that these bits of code are inactive though.
#

for name in [
    :size,
    :(LinearAlgebra.lapack_size),
    :(Base.require_one_based_indexing),
    :in,
    :iszero,
    :isempty,
    :isbitstype,
    :sizeof,
    :promote_type,
    :(Base.elsize),
    :(Core.Compiler.sizeof_nothrow),
    :(Base.datatype_haspadding),
    :(Base.datatype_nfields),
    :(Base.datatype_pointerfree),
    :(Base.datatype_alignment),
    :(Base.datatype_fielddesc_type),
    :(LinearAlgebra.chkstride1),
]
    @eval @is_primitive MinimalCtx Tuple{typeof($name), Vararg}
    @eval function rrule!!(::CoDual{Core.Typeof($name)}, args::CoDual...)
        v = $name(map(primal, args)...)
        return CoDual(v, zero_tangent(v)), NoPullback()
    end
end

@is_primitive MinimalCtx Tuple{Type, TypeVar, Type}
function rrule!!(x::CoDual{<:Type}, y::CoDual{<:TypeVar}, z::CoDual{<:Type})
    return CoDual(primal(x)(primal(y), primal(z)), NoTangent()), NoPullback()
end

"""
    lgetfield(x, f::Val)

An implementation of `getfield` in which the the field `f` is specified statically via a
`Val`. This enables the implementation to be type-stable even when it is not
possible to constant-propagate `f`. Moreover, it enable the pullback to also be type-stable.

It will always be the case that
```julia
getfield(x, :f) === lgetfield(x, Val(:f))
getfield(x, 2) === lgetfield(x, Val(2))
```

This approach is identical to the one taken by `Zygote.jl` to circumvent the same problem.
`Zygote.jl` calls the function `literal_getfield`, while we call it `lgetfield`.
"""
lgetfield(x, ::Val{f}) where {f} = getfield(x, f)

@is_primitive MinimalCtx Tuple{typeof(lgetfield), Any, Any}
function rrule!!(::CoDual{typeof(lgetfield)}, x::CoDual, ::CoDual{Val{f}}) where {f}
    lgetfield_pb!!(dy, df, dx, dsym) = df, increment_field!!(dx, dy, Val{f}()), dsym
    y = CoDual(getfield(primal(x), f), _get_tangent_field(primal(x), tangent(x), f))
    return y, lgetfield_pb!!
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:misc})

    # Data which needs to not be GC'd.
    _x = Ref(5.0)
    _dx = Ref(4.0)
    memory = Any[_x, _dx]

    test_cases = Any[
        # Rules to avoid pointer type conversions.
        (
            true,
            :stability,
            nothing,
            +,
            CoDual(
                bitcast(Ptr{Float64}, pointer_from_objref(_x)),
                bitcast(Ptr{Float64}, pointer_from_objref(_dx)),
            ),
            2,
        ),

        # Lack of activity-analysis rules:
        (false, :stability, nothing, Base.elsize, randn(5, 4)),
        (false, :stability, nothing, Base.elsize, view(randn(5, 4), 1:2, 1:2)),
        (false, :stability, nothing, Core.Compiler.sizeof_nothrow, Float64),
        (false, :stability, nothing, Base.datatype_haspadding, Float64),

        # Performance-rules that would ideally be completely removed.
        (false, :stability, nothing, size, randn(5, 4)),
        (false, :stability, nothing, LinearAlgebra.lapack_size, 'N', randn(5, 4)),
        (false, :stability, nothing, Base.require_one_based_indexing, randn(2, 3), randn(2, 1)),
        (false, :stability, nothing, in, 5.0, randn(4)),
        (false, :stability, nothing, iszero, 5.0),
        (false, :stability, nothing, isempty, randn(5)),
        (false, :stability, nothing, isbitstype, Float64),
        (false, :stability, nothing, sizeof, Float64),
        (false, :stability, nothing, promote_type, Float64, Float64),
        (false, :stability, nothing, LinearAlgebra.chkstride1, randn(3, 3)),
        (false, :stability, nothing, LinearAlgebra.chkstride1, randn(3, 3), randn(2, 2)),

        # Literal replacements for getfield and others.
        (false, :stability, nothing, lgetfield, (5.0, 4), Val(1)),
        (false, :stability, nothing, lgetfield, (5.0, 4), Val(2)),
        (false, :stability, nothing, lgetfield, (a=5.0, b=4), Val(1)),
        (false, :stability, nothing, lgetfield, (a=5.0, b=4), Val(2)),
        (false, :stability, nothing, lgetfield, (a=5.0, b=4), Val(:a)),
        (false, :stability, nothing, lgetfield, (a=5.0, b=4), Val(:b)),
    ]
    return test_cases, memory
end

generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:misc}) = Any[], Any[]
