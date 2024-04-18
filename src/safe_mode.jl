
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
function (pb::SafePullback)(dy)
    verify_rvs_type(pb.y, dy)
    dx = pb.pb(dy)

    # Number of arguments and number of elements in pullback must match. Have to check this
    # because `zip` doesn't require equal lengths for arguments.
    l_pb = length(pb.x)
    l_dx = length(dx)
    if l_pb != l_dx
        error("Number of args = $l_pb but number of rdata = $l_dx. They must to be equal.")
    end

    # Use for-loop to keep stack trace as simple as possible.
    for (x, dx) in zip(pb.x, dx)
        verify_rvs_type(x, dx)
    end
    return dx
end

function verify_rvs_type(::P, dx) where {P}
    _R = rdata_type(tangent_type(P))
    R = _typeof(dx)
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
function (rule::SafeRRule)(x::CoDual...)
    # Use for-loop to keep the stack trace as simple as possible.
    for _x in x
        verify_fwds_type(_x)
    end
    y, pb = rule.rule(x...)
    verify_fwds_type(y)
    return y, SafePullback(pb, primal(y), map(primal, x))
end

function verify_fwds_type(x::CoDual)
    P = _typeof(primal(x))
    F = _typeof(tangent(x))
    _F = fdata_type(tangent_type(_typeof(primal(x))))
    isa(tangent(x), _F) || throw(ArgumentError("Type $P has fdata type $_F, but got $F."))
end
