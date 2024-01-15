"""
    normalise!(ir::IRCode, spnames::Vector{Symbol})

Apply a sequence of standardising transformations to `ir` which leaves its semantics
unchanged, but makes AD more straightforward. In particular, replace
1. `:invoke` `Expr`s with `:call`s,
2. `:foreigncall` `Expr`s with `:call`s to `Taped._foreigncall_`,
3. `:new` `Expr`s with `:call`s to `Taped._new_`,
4. `Core.IntrinsicFunction`s with counterparts from `Taped.IntrinsicWrappers`.

Additionally applies `rebind_multiple_usage!`, which is _essential_ for AD correctness.

`spnames` are the names associated to the static parameters of `ir`. These are needed when
handling `:foreigncall` expressions, in which it is not necessarily the case that all
static parameter names have been translated into either types, or `:static_parameter`
expressions.

Unfortunately, the static parameter names are not retained in `IRCode`, and the `Method`
from which the `IRCode` is derived must be consulted. `Taped.is_vararg_sig_and_sparam_names`
provides a convenient way to do this.
"""
function normalise!(ir::IRCode, spnames::Vector{Symbol})

    # Apply per-instruction transformations to each instruction.
    sp_map = Dict{Symbol, CC.VarState}(zip(spnames, ir.sptypes))
    for (n, inst) in enumerate(ir.stmts.inst)
        inst = invoke_to_call(inst)
        inst = foreigncall_to_call(inst, sp_map)
        inst = new_to_call(inst)
        inst = intrinsic_to_function(inst)
        ir.stmts.inst[n] = inst
    end

    # Apply multi-line transformations.
    ir = rebind_multiple_usage!(ir)
    return ir
end

"""
    invoke_to_call(inst)

If `inst` is an `:invoke` expression, return an equivalent `:call` expression. If anything
else just return `inst`.

Warning: this function does *not* check whether this transformation is safe to perform.
"""
invoke_to_call(inst) = Meta.isexpr(inst, :invoke) ? Expr(:call, inst.args[2:end]...) : inst

"""
    foreigncall_to_call(inst, sp_map::Dict{Symbol, CC.VarState})

If `inst` is a `:foreigncall` expression translate it into an equivalent `:call` expression.
If anything else, just return `inst`.

`sp_map` maps the names of the static parameters to their values. This function is intended
to be called in the context of an `IRCode`, in which case the values of `sp_map` are given
by the `sptypes` field of said `IRCode`. The keys should generally be obtained from the
`Method` from which the `IRCode` is derived. See `Taped.normalise!` for more details.
"""
function foreigncall_to_call(inst, sp_map::Dict{Symbol, CC.VarState})
    if Meta.isexpr(inst, :foreigncall)
        # See Julia's AST devdocs for info on `:foreigncall` expressions.
        args = inst.args
        name = __extract_foreigncall_name(args[1])
        RT = Val(interpolate_sparams(args[2], sp_map))
        AT = (map(x -> Val(interpolate_sparams(x, sp_map)), args[3])..., )
        nreq = Val(args[4])
        calling_convention = Val(args[5] isa QuoteNode ? args[5].value : args[5])
        x = args[6:end]
        return Expr(:call, _foreigncall_, name, RT, AT, nreq, calling_convention, x...)
    else
        return inst
    end
end

# Copied from Umlaut.jl.
__extract_foreigncall_name(x::Symbol) = Val(x)
function __extract_foreigncall_name(x::Expr)
    # Make sure that we're getting the expression that we're expecting.
    !Meta.isexpr(x, :call) && error("unexpected expr $x")
    !isa(x.args[1], GlobalRef) && error("unexpected expr $x")
    x.args[1].name != :tuple && error("unexpected expr $x")
    length(x.args) != 3 && error("unexpected expr $x")

    # Parse it into a name that can be passed as a type.
    v = eval(x)
    return Val((Symbol(v[1]), Symbol(v[2])))
end
__extract_foreigncall_name(v::Tuple) = Val((Symbol(v[1]), Symbol(v[2])))
__extract_foreigncall_name(x::QuoteNode) = __extract_foreigncall_name(x.value)
function __extract_foreigncall_name(x::GlobalRef)
    return __extract_foreigncall_name(getglobal(x.mod, x.name))
end

# Copied from Umlaut.jl. Originally, adapted from
# https://github.com/JuliaDebug/JuliaInterpreter.jl/blob/aefaa300746b95b75f99d944a61a07a8cb145ef3/src/optimize.jl#L239
function interpolate_sparams(@nospecialize(t::Type), sparams::Dict{Symbol, CC.VarState})
    t isa Core.TypeofBottom && return t
    while t isa UnionAll
        t = t.body
    end
    t = t::DataType
    if Base.isvarargtype(t)
        return Expr(:(...), t.parameters[1])
    end
    if Base.has_free_typevars(t)
        params = map(t.parameters) do @nospecialize(p)
            if isa(p, TypeVar)
                return sparams[p.name].typ.val
            elseif isa(p, DataType) && Base.has_free_typevars(p)
                return interpolate_sparams(p, sparams)
            elseif p isa CC.VarState
                @show "doing varstate"
                p.typ
            else
                return p
            end
        end
        T = t.name.Typeofwrapper.parameters[1]
        return T{params...}
    end
    return t
end

"""
    new_to_call(x)

If instruction `x` is a `:new` expression, replace if with a `:call` to `Taped._new_`.
Otherwise, return `x`.
"""
new_to_call(x) = Meta.isexpr(x, :new) ? Expr(:call, _new_, x.args...) : x

"""
    intrinsic_to_function(inst)

If `inst` is a `:call` expression to a `Core.IntrinsicFunction`, replace it with a call to
the corresponding `function` from `Taped.IntrinsicsWrappers`, else return `inst`.

`cglobal` is a special case -- it requires that its first argument be static in exactly the
same way as `:foreigncall`. See `IntrinsicsWrappers.__cglobal` for more info.
"""
function intrinsic_to_function(inst)
    return Meta.isexpr(inst, :call) ? Expr(:call, lift_intrinsic(inst.args...)...) : inst
end

lift_intrinsic(x...) = x
function lift_intrinsic(x::GlobalRef, args...)
    val = getglobal(x.mod, x.name)
    return val isa Core.IntrinsicFunction ? lift_intrinsic(val, args...) : (x, args...)
end
function lift_intrinsic(x::Core.IntrinsicFunction, v, args...)
    if x === cglobal
        return IntrinsicsWrappers.__cglobal, __extract_foreigncall_name(v), args...
    else
        return IntrinsicsWrappers.translate(Val(x)), v, args...
    end
end

"""
    rebind_multiple_usage!(ir::IRCode)

Transforms `ir` to ensure that all `:call` expressions have only a single usage of any
`Argument` or `SSAValue`.

For example, an expression such as
```julia
foo(%1, %2, %1)
```
uses the `SSAValue` `%1` twice. This can cause correctness issues issues on the reverse-pass
of AD if `%1` happens to be differentiable and a bits-type.

`rebind_multiple_usage!` transforms the above example, and generalisations
thereof, into
```julia
x = Taped.__rebind(%1)
foo(%1, %2, x)
```
where `Taped.__rebind` is equivalent to the `identity`, but is always a primitive.
"""
function rebind_multiple_usage!(ir::IRCode)
    for (n, inst) in enumerate(ir.stmts.inst)
        if Meta.isexpr(inst, :call)
            args = inst.args
            for (j, arg) in enumerate(args)
                isa(arg, Union{SSAValue, Argument}) || continue
                m = findfirst(==(arg), view(args, 1:j-1))
                m === nothing && continue
                new_inst = CC.NewInstruction(Expr(:call, __rebind, arg), Any)
                rebound_arg = CC.insert_node!(ir, n, new_inst, #=insert_after=#false)
                args[j] = rebound_arg
            end
        end
    end
    return CC.compact!(ir)
end

"""
    __rebind(x)

A different name for the `identity` function. `__rebind` is a primitive in the `MinimalCtx`,
and is used to ensure the correctness of AD. See `Taped.rebind_multiple_usage!` for details.
"""
__rebind(x) = x
@is_primitive MinimalCtx Tuple{typeof(__rebind), Any}
__rebind_pb!!(dy, df, dx) = df, increment!!(dx, dy)
rrule!!(::CoDual{typeof(__rebind)}, x::CoDual) = x, __rebind_pb!!
