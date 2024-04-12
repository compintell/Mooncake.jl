
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
    # Use for-loop to keep stack trace as simple as possible.
    for (x, dx) in zip(pb.x, dx)
        verify_rvs_type(x, dx)
    end
    return dx
end

function verify_rvs_type(::P, dx) where {P}
    rvs_type = reverse_data_type(tangent_type(P))
    if !isa(dx, rvs_type)
        msg = "For primal of type $P rdata type must be $rvs_type, but got $(_typeof(dx))"
        throw(ArgumentError(msg))
    end
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
    fwds_type = fwds_codual_type(_typeof(primal(x)))
    if !isa(x, fwds_type)
        msg = "For primal of type $P fdata type must be $fwds_type, but got $(_typeof(x))"
        throw(ArgumentError(msg))
    end
end
