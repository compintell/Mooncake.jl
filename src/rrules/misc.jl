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
    :(Threads.nthreads),
    :(Base.depwarn),
    :(Base.reduced_indices),
]
    @eval @is_primitive DefaultCtx Tuple{typeof($name), Vararg}
    @eval function rrule!!(::CoDual{_typeof($name)}, args::CoDual...)
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

# Specialise for non-differentiable arguments.
function rrule!!(::CoDual{typeof(lgetfield)}, x::CoDual{<:Any, NoTangent}, ::CoDual{Val{f}}) where {f}
    return uninit_codual(getfield(primal(x), f)), NoPullback()
end

lgetfield(x, ::Val{f}, ::Val{order}) where {f, order} = getfield(x, f, order)

@is_primitive MinimalCtx Tuple{typeof(lgetfield), Any, Any, Any}
function rrule!!(::CoDual{typeof(lgetfield)}, x::CoDual, ::CoDual{Val{f}}, ::CoDual{Val{order}}) where {f, order}
    lgetfield_pb!!(dy, df, dx, dsym, dorder) = df, increment_field!!(dx, dy, Val{f}()), dsym, dorder
    y = CoDual(getfield(primal(x), f), _get_tangent_field(primal(x), tangent(x), f))
    return y, lgetfield_pb!!
end

function rrule!!(::CoDual{typeof(lgetfield)}, x::CoDual{<:Any, NoTangent}, ::CoDual{Val{f}}, ::CoDual{Val{order}}) where {f, order}
    return uninit_codual(getfield(primal(x), f)), NoPullback()
end

"""
    lsetfield!(value, name::Val, x, [order::Val])

This function is to `setfield!` what `lgetfield` is to `getfield`. It will always hold that
```julia
setfield!(copy(x), :f, v) == lsetfield!(copy(x), Val(:f), v)
setfield!(copy(x), 2, v) == lsetfield(copy(x), Val(2), v)
```
"""
lsetfield!(value, ::Val{name}, x) where {name} = setfield!(value, name, x)

@is_primitive MinimalCtx Tuple{typeof(lsetfield!), Any, Any, Any}
function rrule!!(
    ::CoDual{typeof(lsetfield!)}, value::CoDual, ::CoDual{Val{name}}, x::CoDual
) where {name}
    save = isdefined(primal(value), name)
    old_x = save ? getfield(primal(value), name) : nothing
    old_dx = save ? val(getfield(tangent(value).fields, name)) : nothing
    function setfield!_pullback(dy, df, dvalue, dname, dx)
        new_dx = increment!!(dx, val(getfield(dvalue.fields, name)))
        new_dx = increment!!(new_dx, dy)
        old_x !== nothing && lsetfield!(primal(value), Val(name), old_x)
        old_x !== nothing && set_tangent_field!(tangent(value), name, old_dx)
        return df, dvalue, dname, new_dx
    end
    y = CoDual(
        lsetfield!(primal(value), Val(name), primal(x)),
        set_tangent_field!(tangent(value), name, tangent(x)),
    )
    return y, setfield!_pullback
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
        (false, :stability_and_allocs, nothing, Base.elsize, randn(5, 4)),
        (false, :stability_and_allocs, nothing, Base.elsize, view(randn(5, 4), 1:2, 1:2)),
        (false, :stability_and_allocs, nothing, Core.Compiler.sizeof_nothrow, Float64),
        (false, :stability_and_allocs, nothing, Base.datatype_haspadding, Float64),

        # Performance-rules that would ideally be completely removed.
        (false, :stability_and_allocs, nothing, size, randn(5, 4)),
        (false, :stability_and_allocs, nothing, LinearAlgebra.lapack_size, 'N', randn(5, 4)),
        (false, :stability_and_allocs, nothing, Base.require_one_based_indexing, randn(2, 3), randn(2, 1)),
        (false, :stability_and_allocs, nothing, in, 5.0, randn(4)),
        (false, :stability_and_allocs, nothing, iszero, 5.0),
        (false, :stability_and_allocs, nothing, isempty, randn(5)),
        (false, :stability_and_allocs, nothing, isbitstype, Float64),
        (false, :stability_and_allocs, nothing, sizeof, Float64),
        (false, :stability_and_allocs, nothing, promote_type, Float64, Float64),
        (false, :stability_and_allocs, nothing, LinearAlgebra.chkstride1, randn(3, 3)),
        (false, :stability_and_allocs, nothing, LinearAlgebra.chkstride1, randn(3, 3), randn(2, 2)),
        (false, :allocs, nothing, Threads.nthreads),

        # Literal replacements for getfield.
        (false, :stability_and_allocs, nothing, lgetfield, (5.0, 4), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (5.0, 4), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, (1, 4), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, ((), 4), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, (a=5.0, b=4), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (a=5.0, b=4), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, (a=5.0, b=4), Val(:a)),
        (false, :stability_and_allocs, nothing, lgetfield, (a=5.0, b=4), Val(:b)),

        # Literal replacement for setfield!.
        (
            false,
            :stability_and_allocs,
            nothing,
            lsetfield!,
            TestResources.MutableFoo(5.0, [1.0, 2.0]),
            Val(:a),
            4.0,
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            lsetfield!,
            TestResources.FullyInitMutableStruct(5.0, [1.0, 2.0]),
            Val(:y),
            [1.0, 3.0, 4.0],
        ),
    ]
    return test_cases, memory
end

generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:misc}) = Any[], Any[]
