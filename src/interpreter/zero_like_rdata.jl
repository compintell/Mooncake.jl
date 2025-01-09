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
@inline increment!!(r::R, ::ZeroRData) where {R} = r
@inline increment!!(::ZeroRData, ::ZeroRData) = ZeroRData()

"""
    zero_like_rdata_type(::Type{P}) where {P}

Indicates the type which will be returned by `zero_like_rdata_from_type`. Will be the rdata
type for `P` if we can produce the zero rdata element given only `P`, and will be the union
of `R` and `ZeroRData` if an instance of `P` is needed.
"""
function zero_like_rdata_type(::Type{P}) where {P}
    R = rdata_type(tangent_type(P))
    return can_produce_zero_rdata_from_type(P) ? R : Union{R,ZeroRData}
end

"""
    zero_like_rdata_from_type(::Type{P}) where {P}

This is an internal implementation detail -- you should generally not use this function.

Returns _either_ the zero element of type `rdata_type(tangent_type(P))`, or a `ZeroRData`.
It is always valid to return a `ZeroRData`, 
"""
function zero_like_rdata_from_type(::Type{P}) where {P}
    return can_produce_zero_rdata_from_type(P) ? zero_rdata_from_type(P) : ZeroRData()
end
