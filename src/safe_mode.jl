
"""
    SafePullback(pb, y, x)

Construct a callable which is equivalent to `pb`, but which enforces type-based pre- and
post-conditions to `pb`. Let `dx = pb.pb(dy)`, for some rdata `dy`, then this function
- checks that `dy` has the correct rdata type for `y`, and
- checks that each element of `dx` has the correct rdata type for `x`.
"""
struct SafePullback{Tpb, Ty, Tx}
    pb::Tpb
end

"""
    (pb::SafePullback)(dy)

Apply type checking to enforce pre- and post-conditions on `pb.pb`. See the docstring for
`SafePullback` for details.
"""
@inline function (pb::SafePullback{Tpb, Ty, Tx})(dy) where {Tpb, Ty, Tx}
    verify_rvs_input(Ty, dy)
    dx = pb.pb(dy)
    verify_rvs_output(Tx, dx)
    return dx
end

@noinline verify_rvs_input(::Type{Ty}, dy) where {Ty} = verify_rvs(Ty, dy)

@noinline function verify_rvs_output(::Type{Tx}, dx) where {Tx}
    @nospecialize pb dx

    # Number of arguments and number of elements in pullback must match. Have to check this
    # because `zip` doesn't require equal lengths for arguments.
    l_pb = length(Tx.parameters)
    l_dx = length(dx)
    if l_pb != l_dx
        error("Number of args = $l_pb but number of rdata = $l_dx. They must to be equal.")
    end

    # Use for-loop to keep stack trace as simple as possible.
    for (x, dx) in zip(Tx.parameters, dx)
        verify_rvs(x, dx)
    end
end

@noinline function verify_rvs(::Type{P}, dx::R) where {P, R}
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

Additionally, dynamic checks may be performed (e.g. that an fdata array of the same size as
its primal).

Let `rule` return `y, pb!!`, then `SafeRRule(rule)` returns `y, SafePullback(pb!!)`.
`SafePullback` inserts the same kind of checks as `SafeRRule`, but on the reverse-pass. See
the docstring for details.

*Note:* at any given point in time, the checks performed by this function constitute a
necessary but insufficient set of conditions to ensure correctness. If you find that an
error isn't being caught by these tests, but you believe it ought to be, please open an
issue or (better still) a PR.
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
    verify_fwds_output(x, y)
    return y::CoDual, SafePullback{_typeof(pb), _typeof(primal(y)), Tuple{tuple_map(_typeof ∘ primal, x)...}}(pb)
end

@noinline function verify_fwds_inputs(@nospecialize(x::Tuple))
    try
        # Use for-loop to keep the stack trace as simple as possible.
        for _x in x
            verify_fwds(_x)
        end
    catch e
        error("error in inputs to rule with input types $(_typeof(x))")
    end
end

@noinline function verify_fwds_output(@nospecialize(x), @nospecialize(y))
    try
        verify_fwds(y)
    catch e
        error("error in outputs of rule with input types $(_typeof(x))")
    end
end

@noinline function verify_fwds(x::CoDual{P, F}) where {P, F}
    _fdata_type_checker(P, F)
    verify_fwds_values(primal(x), tangent(x))
end

function verify_fwds_values(p::P, f::F) where {P, F}
    _fdata_type_checker(P, F)
    if F == NoFData
        return
    elseif P <: Array
        if size(p) != size(f)
            throw(ArgumentError("size of P is $(size(p)) but size of F is $(size(f))"))
        end
        for n in eachindex(p)
            !isassigned(p, n) && continue
            Fn = _typeof(f[n])
            Pn = _typeof(p[n])
            Tn = tangent_type(Pn)
            if Fn != Tn
                throw(ArgumentError(
                    "the type of each element of an fdata Array must be the tangent_type " *
                    "of the corresponding element of the primal array. Found that " *
                    "element $n of fdata array is of type $Fn, while primal is of " *
                    "type $Pn, whose tangent type is $Tn.",
                ))
            end
        end
    elseif isstructtype(P)
        return
    end
end

function _fdata_type_checker(P, F)
    _F = fdata_type(tangent_type(P))
    F == _F || throw(ArgumentError("type $P has fdata type $_F, but got $F."))
end
