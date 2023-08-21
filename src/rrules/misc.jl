#
# (in-principle) non-essential rules.
# Some of these might be removed in the future in favour of lower-level rules.
#

isprimitive(::RMC, ::typeof(sin), ::Float64) = true
function rrule!!(::CoDual{typeof(sin)}, x::CoDual{Float64})
    x = primal(x)
    y, partial = sincos(x)
    function sin_pullback!!(dy::Float64, dsin, dx::Float64)
        return dsin, increment!!(dx, dy * partial)
    end
    return CoDual(y, zero(y)), sin_pullback!!
end

isprimitive(::RMC, ::typeof(cos), ::Float64) = true
function rrule!!(::CoDual{typeof(cos)}, x::CoDual{Float64})
    x = primal(x)
    neg_partial, y = sincos(x)
    function cos_pullback!!(dy::Float64, dcos, dx::Float64)
        return dcos, increment!!(dx, -dy * neg_partial)
    end
    return CoDual(y, zero(y)), cos_pullback!!
end

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
]
    @eval isprimitive(::RMC, ::Core.Typeof($name), args...) = true
    @eval function rrule!!(::CoDual{Core.Typeof($name)}, args::CoDual...)
        v = $name(map(primal, args)...)
        return CoDual(v, zero_tangent(v)), NoPullback()
    end
end
