
"""
    ZeroRData()

Singleton type indicating zero-valued rdata. This should only ever appear as an
intermediate quantity in the reverse-pass of AD when the type of the primal is not fully
inferable, or a field of a type is abstractly typed.

If you see this anywhere in actual code, or if it appears in a hand-written rule, this is an
error -- please open an issue in such a situation.
"""
struct ZeroRData end

@inline increment!!(::ZeroRData, r::R) where {R} = r

"""
    zero_like_rdata_from_type(::Type{P}) where {P}
"""
zero_like_rdata_from_type(::Type{P}) where {P}

zero_like_rdata_from_type(::Type{P}) where {P<:IEEEFloat} = zero(P)

@generated function zero_like_rdata_from_type(::Type{P}) where {P}

    # Get types associated to primal.
    T = tangent_type(P)
    R = rdata_type(T)

    # If there's no reverse data, return no reverse data, e.g. for mutable types.
    R == NoRData && return NoRData()

    # If the type is itself abstract, it's reverse data could be anything.
    # The same goes for if the type has any undetermined type parameters.
    (isabstracttype(T) || !isconcretetype(T)) && return ZeroRData()

    # T ought to be a `Tangent`. If it's not, something has gone wrong.
    !(T <: Tangent) && return Expr(:call, error, "Unhandled type $T")
    rdata_field_zeros_exprs = ntuple(fieldcount(P)) do n
        R_field = rdata_field_type(P, n)
        if R_field <: PossiblyUninitTangent
            return :($R_field(zero_like_rdata_from_type($(fieldtype(P, n)))))
        else
            return :(zero_like_rdata_from_type($(fieldtype(P, n))))
        end
    end
    backing_data_expr = Expr(:call, :tuple, rdata_field_zeros_exprs...)
    backing_expr = :($(rdata_backing_type(P))($backing_data_expr))
    return Expr(:call, R, backing_expr)
end

@generated function zero_like_rdata_from_type(::Type{P}) where {P<:Tuple}
    # Get types associated to primal.
    T = tangent_type(P)
    R = rdata_type(T)

    # If there's no reverse data, return no reverse data, e.g. for mutable types.
    R == NoRData && return NoRData()

    # If the type is not concrete, then use the `ZeroRData` option.
    Base.isconcretetype(R) || return ZeroRData()

    return Expr(:call, tuple, map(p -> :(zero_like_rdata_from_type($p)), P.parameters)...)
end

@generated function zero_like_rdata_from_type(::Type{NamedTuple{names, Pt}}) where {names, Pt}

    # Get types associated to primal.
    P = NamedTuple{names, Pt}
    T = tangent_type(P)
    R = rdata_type(T)

    # If there's no reverse data, return no reverse data, e.g. for mutable types.
    R == NoRData && return NoRData()

    # If the type is not concrete, then use the `ZeroRData` option.
    Base.isconcretetype(R) || return ZeroRData()

    return :(NamedTuple{$names}(zero_like_rdata_from_type($Pt)))
end
