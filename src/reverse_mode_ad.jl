struct ReverseModeADContext <: TapedContext end

const RMC = ReverseModeADContext

struct CoDual{Tx, Tdx}
    x::Tx
    dx::Tdx
end

# Always sharpen the first thing if it's a type, in order to preserve dispatch possibility.
function CoDual(x::Type{P}, dx::NoTangent) where {P}
    if @isdefined(P)
        return CoDual{Type{P}, NoTangent}(P, dx)
    else
        return CoDual{typeof(x), NoTangent}(x, dx)
    end
end

primal(x::CoDual) = x.x
tangent(x::CoDual) = x.dx
Base.copy(x::CoDual) = CoDual(copy(primal(x)), copy(tangent(x)))

"""
    zero_codual(x)

Equivalent to `CoDual(x, zero_tangent(x))`.
"""
zero_codual(x) = CoDual(x, zero_tangent(x))

"""
    uninit_codual(x)

See implementation for details, as this function is subject to change.
"""
uninit_codual(x) = CoDual(x, uninit_tangent(x))

"""
    randn_codual(rng::AbstractRNG, x)

Equivalent to `CoDual(x, randn_tangent(rng, x))`.
"""
randn_codual(rng::AbstractRNG, x) = CoDual(x, randn_tangent(rng, x))

set_tangent!!(x::CoDual, dx) = CoDual(primal(x), increment!!(set_to_zero!!(tangent(x)), dx))

function verify_codual_type(::CoDual{P, T}) where {P, T}
    Tt = tangent_type(P)
    if Tt !== T
        throw(error("for primal of type $P, expected tangent of type $Tt, but found $T"))
    end
end

struct NoPullback end

@inline (::NoPullback)(dy, dx...) = dx

might_be_active(args) = any(might_be_active ∘ typeof, args)

isprimitive(::RMC, args...) = might_be_active(args) ? false : true
isprimitive(::RMC, ::typeof(Umlaut.__new__), T, x...) = true
isprimitive(::RMC, ::typeof(Umlaut.__foreigncall__), args...) = true
isprimitive(::RMC, ::typeof(__intrinsic__), args...) = true
isprimitive(::RMC, ::Core.Builtin, x...) = true

"""
    rebind(x)

Primal evaluation is equivalent to the identity function. However, the `rrule!!` ensures
that something sensible happens with the tangent. Not to be used by downstream users.
Is basically a hack to ensure that double counting doesn't occur.

TODO: improve docstring + explain / understand semantics better.
"""
rebind(x) = x

rebind_pb!!(ȳ, f̄, x̄) = f̄, increment!!(x̄, ȳ)
function rrule!!(::CoDual{typeof(rebind)}, x::CoDual)
    return CoDual(primal(x), rebind_tangent(tangent(x))), rebind_pb!!
end

isprimitive(::RMC, ::typeof(rebind), x) = true


"""
    struct Deduplicated{<:Tuple} end

Used internally to replace calls with repeated arguments with calls with no repeated
arguments -- this is essential for correctness when handling repeated isbits types.

For example, we replace a call of the form `f(x, x)` with
`Deduplicated{Tuple{1, 2, 2}}()(f, x)`, where this function expands to code equivalent to
```julia
function (::Deduplicated{Tuple{1, 2, 2}})(x...)
    y = rebind(x[2])
    return x[1](x[2], y)
end
```
The type parameter of `Deduplicated` specifies the mapping between unique parameters, and
how they should be passed to into the function call.
For example, {1, 2, 2} states that there are two unique arguments (deduce this from the
number of unique integers in the tuple), the first of which is the function, the second of
which is passed as the 1st and 2nd positional arguments to the function.
"""
struct Deduplicated{T<:Tuple} end

@generated function (::Deduplicated{T})(x::Vararg{Any, N}) where {T, N}
    position_map = Int[x for x in T.parameters]
    @assert maximum(position_map) == N

    # Insert rebinding calls where needed.
    calls = Any[]
    args = Any[]

    foreach(enumerate(position_map)) do (call_pos, x_pos)
        # if findfirst(Base.Fix1(===, x_pos), position_map[1:call_pos-1]) === nothing
        get_sym = gensym(Symbol("get_$(x_pos)_for_$(call_pos)"))
        push!(calls, :($get_sym = getfield(x, $x_pos)))
        push!(args, get_sym)
        # else
        #     rebound_arg_sym = gensym(Symbol(call_pos))
        #     push!(calls, :($rebound_arg_sym = rebind(getfield(x, $x_pos))))
        #     push!(args, rebound_arg_sym)
        # end
    end

    # Push the actual call onto the list, shove everything in a block, and return it.
    push!(calls, Expr(:call, args...))
    return Expr(:block, calls...)
end
