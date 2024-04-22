
"""
    SafePullback(pb, y, x)

Construct a callable which is equivalent to `pb`, but which enforces type-based pre- and
post-conditions to `pb`. Let `dx = pb.pb(dy)`, for some rdata `dy`, then this function
- checks that `dy` has the correct rdata type for `y`, and
- checks that each element of `dx` has the correct rdata type for `x`.
"""
struct SafePullback{Tpb, Ty, Tx}
    pb::Tpb
    y::Ty
    x::Tx
end

"""
    (pb::SafePullback)(dy)

Apply type checking to enforce pre- and post-conditions on `pb.pb`. See the docstring for
`SafePullback` for details.
"""
@inline function (pb::SafePullback)(dy)
    verify_rvs(pb.y, dy)
    dx = pb.pb(dy)
    verify_rvs_output(pb, dx)
    return dx
end

@noinline function verify_rvs_output(pb, dx)
    # Number of arguments and number of elements in pullback must match. Have to check this
    # because `zip` doesn't require equal lengths for arguments.
    l_pb = length(pb.x)
    l_dx = length(dx)
    if l_pb != l_dx
        error("Number of args = $l_pb but number of rdata = $l_dx. They must to be equal.")
    end

    # Use for-loop to keep stack trace as simple as possible.
    for (x, dx) in zip(pb.x, dx)
        verify_rvs(x, dx)
    end
end

@noinline function verify_rvs(::P, dx::R) where {P, R}
    _R = rdata_type(tangent_type(P))
    R <: ZeroRData && return nothing
    (R <: _R) || throw(ArgumentError("Type $P has rdata type $_R, but got $R."))
end

"""
    SafeRRule(rule)

Construct a callable which is equivalent to `rule`, but inserts additional type checking.
In particular:
- check that the fdata in each argument is of the correct type for the primal
- check that the fdata in the `CoDual` returned from the rule is of the correct type for the
    primal.

Let `rule` returns `y, pb!!`, then `SafeRRule(rule)` returns `y, SafePullback(pb!!)`.
`SafePullback` inserts the same kind of checks as `SafeRRule`, but on the reverse-pass. See
the docstring for details.
"""
struct SafeRRule{Trule}
    rule::Trule
end

"""
    (rule::SafeRRule)(x::CoDual...)

Apply type checking to enforce pre- and post-conditions on `rule.rule`. See the docstring
for `SafeRRule` for details.
"""
@inline function (rule::SafeRRule)(x::Vararg{CoDual, N}) where {N}
    verify_fwds_inputs(x)
    y, pb = rule.rule(x...)
    verify_fwds(y)
    return y::CoDual, SafePullback(pb, primal(y), map(primal, x))
end

@noinline function verify_fwds_inputs(x::Tuple)
    # Use for-loop to keep the stack trace as simple as possible.
    for _x in x
        verify_fwds(_x)
    end
end

@noinline function verify_fwds(x::CoDual{P, F}) where {P, F}
    _F = fdata_type(tangent_type(P))
    isa(tangent(x), _F) || throw(ArgumentError("Type $P has fdata type $_F, but got $F."))
end
