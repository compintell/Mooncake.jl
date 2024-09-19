"""
    normalise!(ir::IRCode, spnames::Vector{Symbol})

Apply a sequence of standardising transformations to `ir` which leaves its semantics
unchanged, but makes AD more straightforward. In particular, replace
1. `:foreigncall` `Expr`s with `:call`s to `Mooncake._foreigncall_`,
2. `:new` `Expr`s with `:call`s to `Mooncake._new_`,
3. `:splatnew` Expr`s with `:call`s to `Mooncake._splat_new_`,
4. `Core.IntrinsicFunction`s with counterparts from `Mooncake.IntrinsicWrappers`,
5. `getfield(x, 1)` with `lgetfield(x, Val(1))`, and related transformations,
6. `gc_preserve_begin` / `gc_preserve_end` exprs so that memory release is delayed.

`spnames` are the names associated to the static parameters of `ir`. These are needed when
handling `:foreigncall` expressions, in which it is not necessarily the case that all
static parameter names have been translated into either types, or `:static_parameter`
expressions.

Unfortunately, the static parameter names are not retained in `IRCode`, and the `Method`
from which the `IRCode` is derived must be consulted. `Mooncake.is_vararg_and_sparam_names`
provides a convenient way to do this.
"""
function normalise!(ir::IRCode, spnames::Vector{Symbol})
    sp_map = Dict{Symbol, CC.VarState}(zip(spnames, ir.sptypes))
    for (n, inst) in enumerate(ir.stmts.inst)
        inst = foreigncall_to_call(inst, sp_map)
        inst = new_to_call(inst)
        inst = splatnew_to_call(inst)
        inst = intrinsic_to_function(inst)
        inst = lift_getfield_and_others(inst)
        inst = lift_gc_preservation(inst)
        ir.stmts.inst[n] = inst
    end
    return ir
end

"""
    foreigncall_to_call(inst, sp_map::Dict{Symbol, CC.VarState})

If `inst` is a `:foreigncall` expression translate it into an equivalent `:call` expression.
If anything else, just return `inst`. See `Mooncake._foreigncall_` for details.

`sp_map` maps the names of the static parameters to their values. This function is intended
to be called in the context of an `IRCode`, in which case the values of `sp_map` are given
by the `sptypes` field of said `IRCode`. The keys should generally be obtained from the
`Method` from which the `IRCode` is derived. See `Mooncake.normalise!` for more details.
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

If instruction `x` is a `:new` expression, replace it with a `:call` to `Mooncake._new_`.
Otherwise, return `x`.
"""
new_to_call(x) = Meta.isexpr(x, :new) ? Expr(:call, _new_, x.args...) : x

"""
    splatnew_to_call(x)

If instruction `x` is a `:splatnew` expression, replace it with a `:call` to
`Mooncake._splat_new_`. Otherwise return `x`.
"""
splatnew_to_call(x) = Meta.isexpr(x, :splatnew) ? Expr(:call, _splat_new_, x.args...) : x

"""
    intrinsic_to_function(inst)

If `inst` is a `:call` expression to a `Core.IntrinsicFunction`, replace it with a call to
the corresponding `function` from `Mooncake.IntrinsicsWrappers`, else return `inst`.

`cglobal` is a special case -- it requires that its first argument be static in exactly the
same way as `:foreigncall`. See `IntrinsicsWrappers.__cglobal` for more info.
"""
function intrinsic_to_function(inst)
    return Meta.isexpr(inst, :call) ? Expr(:call, lift_intrinsic(inst.args...)...) : inst
end

lift_intrinsic(x...) = x
lift_intrinsic(x::GlobalRef, args...) = lift_intrinsic(getglobal(x.mod, x.name), args...)
function lift_intrinsic(x::Core.IntrinsicFunction, v, args...)
    if x === cglobal
        return IntrinsicsWrappers.__cglobal, __extract_foreigncall_name(v), args...
    else
        return IntrinsicsWrappers.translate(Val(x)), v, args...
    end
end
lift_intrinsic(::typeof(Core._apply_iterate), args...) = _apply_iterate_equivalent, args...

"""
    lift_getfield_and_others(inst)

Converts expressions of the form `getfield(x, :a)` into `lgetfield(x, Val(:a))`. This has
identical semantics, but is performant in the absence of proper constant propagation.

Does the same for...
"""
function lift_getfield_and_others(inst)
    Meta.isexpr(inst, :call) || return inst
    f = __get_arg(inst.args[1])
    if f === getfield && length(inst.args) == 3 && inst.args[3] isa Union{QuoteNode, Int}
        field = inst.args[3]
        new_field = field isa Int ? Val(field) : Val(field.value)
        return Expr(:call, lgetfield, inst.args[2], new_field)
    elseif f === getfield && length(inst.args) == 4 && inst.args[3] isa Union{QuoteNode, Int} && inst.args[4] isa Bool
        field = inst.args[3]
        new_field = field isa Int ? Val(field) : Val(field.value)
        return Expr(:call, lgetfield, inst.args[2], new_field, Val(inst.args[4]))
    elseif f === setfield! && length(inst.args) == 4 && inst.args[3] isa Union{QuoteNode, Int}
        name = inst.args[3]
        new_name = name isa Int ? Val(name) : Val(name.value)
        return Expr(:call, lsetfield!, inst.args[2], new_name, inst.args[4])
    else
        return inst
    end
end

__get_arg(x::GlobalRef) = getglobal(x.mod, x.name)
__get_arg(x::QuoteNode) = x.value
__get_arg(x) = x

"""
    gc_preserve(xs...)

A no-op function. Its `rrule!!` ensures that the memory associated to `xs` is not freed
until the pullback that it returns is run.
"""
@inline gc_preserve(xs...) = nothing

@is_primitive MinimalCtx Tuple{typeof(gc_preserve), Vararg{Any, N}} where {N}

function rrule!!(f::CoDual{typeof(gc_preserve)}, xs::CoDual...)
    pb = NoPullback(f, xs...)
    gc_preserve_pb!!(::NoRData) = GC.@preserve xs pb(NoRData())
    return zero_fcodual(nothing), gc_preserve_pb!!
end

"""
    lift_gc_preserve(inst)

Expressions of the form
```julia
y = GC.@preserve x1 x2 foo(args...)
```
get lowered to
```julia
token = Expr(:gc_preserve_begin, x1, x2)
y = expr
Expr(:gc_preserve_end, token)
```
These expressions guarantee that any memory associated `x1` and `x2` not be freed until
the `:gc_preserve_end` expression is reached.

In the context of reverse-mode AD, we must ensure that the memory associated to `x1`, `x2`
and their fdata is available during the reverse pass code associated to `expr`.
We do this by preventing the memory from being freed until the `:gc_preserve_begin` is
reached on the reverse pass.

To achieve this, we replace the primal code with
```julia
# store `x` in `pb_gc_preserve` to prevent it from being freed.
_, pb_gc_preserve = rrule!!(zero_fcodual(gc_preserve), x1, x2)

# Differentiate the `:call` expression in the usual way.
y, foo_pb = rrule!!(zero_fcodual(foo), args...)

# Do not permit the GC to free `x` here.
nothing
```
The pullback should be something along the lines of
```julia
# no pullback associated to `nothing`.
nothing

# Run the pullback associated to `foo` in the usual manner. `x` must be available.
_, dargs... = foo_pb(dy)

# No-op pullback associated to `gc_preserve`.
pb_gc_preserve(NoRData())
```
"""
function lift_gc_preservation(inst)
    Meta.isexpr(inst, :gc_preserve_begin) && return Expr(:call, gc_preserve, inst.args...)
    Meta.isexpr(inst, :gc_preserve_end) && return nothing
    return inst
end
