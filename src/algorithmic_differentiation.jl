struct RMADContext
    cotangent_indices::Dict{Any, Int}
    pbs::Vector{Any}
    cotangents::Vector{Any}
end

struct Dual{Tx, Tx̄}
    x::Tx
    x̄::Tx̄
end

primal(x::Dual) = x.x
cotangent(x::Dual) = x.x̄

mutable struct Operation{Tyȳ<:Dual, Tpb!!, Tv_fargs}
    yȳ::Tyȳ
    pb!!::Tpb!!
    v_fargs::Tv_fargs
end

RMADContext() = RMADContext(Dict{Any, Int}(), [], [])

is_input(op::Operation) = op.pb!! === nothing

function assign_cotangent_index!(ctx::RMADContext, v::Umlaut.Variable)
    push!(ctx.cotangents, nothing)
    cotangent_index = length(ctx.cotangents)
    ctx.cotangent_indices[v.id] = cotangent_index
    return cotangent_index
end

function get_cotangent_index(ctx::RMADContext, v::Umlaut.Variable)
    return ctx.cotangent_indices[v.id]
end

is_constant(v) = v.op isa Umlaut.Constant

function get_cotangent(ctx::RMADContext, v::Umlaut.Variable)
    is_constant(v) && return NoTangent()
    return cotangent(ctx.pbs[get_cotangent_index(ctx, v)].yȳ)
end

get_cotangent(ctx::RMADContext, v) = NoTangent()

function set_cotangent!(ctx::RMADContext, v::Umlaut.Variable, new_ȳ)
    is_constant(v) && return NoTangent()
    op = ctx.pbs[get_cotangent_index(ctx, v)]
    op.yȳ = Dual(primal(op.yȳ), new_ȳ)
    return new_ȳ
end

set_cotangent!(ctx::RMADContext, v, va) = NoTangent()

zero_cotangent(::Float64) = zero(Float64)
zero_cotangent(::Bool) = NoTangent()
zero_cotangent(::Int) = NoTangent()
function zero_cotangent(f::F) where {F}
    fieldcount(F) == 0 || throw(error("f has fields. Unable to find zero cotangent"))
    return NoTangent()
end
zero_cotangent(x::Array{Float64}) = zero(x)

increment!!(::NoTangent, ::NoTangent) = NoTangent()
increment!!(x̄::Float64, dx̄::Float64) = x̄ + dx̄
function increment!!(x̄::Array, dx::Array)
    if !(x̄ === dx) # if x̄ aliases dx, then no need to increment.
        x̄ .= increment!!.(x̄, dx)
    end
    return x̄
end
function increment!!(x̄1::Tangent{P}, x̄2::Tangent{P}) where {P}
    b1 = getfield(x̄1, :backing)
    b2 = getfield(x̄2, :backing)
    names = tuple(union(fieldnames(typeof(b1)), fieldnames(typeof(b2)))...)
    vals = map(n -> increment!!(getfield(b1, n), getfield(b2, n)), names)
    return Tangent{P}(; NamedTuple{names}(vals)...)
end



function ConstructionBase.setproperties(t::Tangent{P,T}, patch::NamedTuple) where {P,T}
    return Tangent{P,T}(ConstructionBase.setproperties(getfield(t, :backing), patch))
end

function setproperty(value, name, x)
    return setproperties(value, NamedTuple{(name, )}((x, )))
end


isprimitive(::RMADContext, ::F, x...) where {F} = false

taped_rrule!!(::F, x...) where {F} = nothing

# Rules for intrinsics.
# Intrinsics reference:
# https://github.com/JuliaLang/julia/blob/628209c1f2f746e3fc21ccd7cb34e67289403d44/src/intrinsics.cpp#L1207
# Possibly useful for relating to LLVM instructions.
isprimitive(::RMADContext, ::Core.IntrinsicFunction, x...) = true

# For performance reasons, this will have to be handled via source re-writing generally,
# or spitting out a different "op" depending upon the value of `f` in a ReverseDiff-like
# system.
function taped_rrule!!(ff̄::Dual{<:Core.IntrinsicFunction}, x_x̄...)
    f = primal(ff̄)
    if f === Core.Intrinsics.sub_int
        x = map(primal, x_x̄)
        y = f(x...)
        function sub_int_intrinsic_pb!!(::NoTangent, ::NoTangent, ::NoTangent, ::NoTangent)
            return NoTangent(), NoTangent(), NoTangent()
        end
        return Dual(y, NoTangent()), sub_int_intrinsic_pb!!
    elseif f === Core.Intrinsics.slt_int
        x = map(primal, x_x̄)
        y = f(x...)
        function slt_int_intrinsic_pb!!(::NoTangent, ::NoTangent, ::NoTangent, ::NoTangent)
            return NoTangent(), NoTangent(), NoTangent()
        end
        return Dual(y, NoTangent()), slt_int_intrinsic_pb!!
    elseif f == Core.Intrinsics.sle_int
        a, b = map(primal, x_x̄)
        @assert a isa Base.BitSigned
        @assert b isa Base.BitSigned
        y = f(a, b)
        function sle_int_intrinsic_pb!!(ȳ::NoTangent, ::NoTangent, ::NoTangent, ::NoTangent)
            return NoTangent(), NoTangent(), NoTangent()
        end
        return Dual(y, NoTangent()), sle_int_intrinsic_pb!!
    elseif f == Core.Intrinsics.not_int
        a = primal(only(x_x̄))
        @assert a isa Union{Base.BitSigned, Bool}
        y = f(a)
        function not_int_intrinsic_pb!!(::NoTangent, ::NoTangent, ::NoTangent)
            return NoTangent(), NoTangent()
        end
        return Dual(y, NoTangent()), not_int_intrinsic_pb!!
    elseif f == Core.Intrinsics.add_int
        a, b = map(primal, x_x̄)
        @assert a isa Base.BitSigned
        @assert b isa Base.BitSigned
        y = f(a, b)
        function add_int_intrinsic_pb!!(::NoTangent, ::NoTangent, ::NoTangent, ::NoTangent)
            return NoTangent(), NoTangent(), NoTangent()
        end
        return Dual(y, NoTangent()), add_int_intrinsic_pb!!
    else
        throw(error("Unknown Core.Intrinsic function $f"))
    end
end

isprimitive(::RMADContext, ::typeof(Umlaut.__new__), T, x...) = true 
function taped_rrule!!(::Dual{typeof(Umlaut.__new__)}, T, x...)
    y = Umlaut.__new__(primal(T), map(primal, x)...)
    Ty = typeof(y)
    ȳ = Tangent{Ty}(; NamedTuple{fieldnames(Ty)}(map(cotangent, x))...)
    yȳ = Dual(y, ȳ)
    function __new__pullback!!(ȳ, ::NoTangent, ::NoTangent, x̄...)
        x̄ = map(increment!!, x̄, tuple(ȳ...))
        return NoTangent(), NoTangent(), x̄...
    end
    return Dual(y, ȳ), __new__pullback!!
end

# Rules for built-ins.
isprimitive(::RMADContext, ::typeof(Core.apply_type), x...) = true
function taped_rrule!!(::Dual{typeof(Core.apply_type)}, x...)
    yȳ = Dual(Core.apply_type(map(primal, x)...), NoTangent())
    apply_type_pullback!!(ȳ, ::NoTangent, x̄::NoTangent...) = NoTangent(), x̄...
    return yȳ, apply_type_pullback!!
end

isprimitive(::RMADContext, ::F, x...) where {F<:typeof(Core.typeassert)} = true
function taped_rrule!!(::Dual{typeof(Core.typeassert)}, x, T)
    typeassert_pullback!!(ȳ, ::NoTangent, x̄, ::NoTangent) = NoTangent(), x̄, NoTangent()
    yȳ = Dual(Core.typeassert(primal(x), primal(T)), cotangent(x))
    return yȳ, typeassert_pullback!!
end

isprimitive(::RMADContext, ::typeof(Core.arrayset), x...) = true
@eval function taped_rrule!!(::Dual{typeof(Core.arrayset)}, boundscheck, A, x, i1)
    prev_val = primal(A)[primal(i1)]
    y = Core.arrayset($(Expr(:boundscheck)), primal(A), primal(x), primal(i1))
    yȳ = Dual(y, cotangent(A))
    function arrayset_pb!!(Ā, ::NoTangent, ::NoTangent, ::Any, x̄, ::NoTangent)
        x̄ = increment!!(x̄, Ā[i1])
        Ā[i1] = zero_cotangent(prev_val)
        A[i1] = prev_val
        return NoTangent(), NoTangent(), Ā, x̄, NoTangent()
    end
    return yȳ, arrayset_pb!!
end

isprimitive(::RMADContext, ::typeof(tuple), x...) = true
function taped_rrule!!(::Dual{typeof(tuple)}, xx̄...)
    y = tuple(map(primal, xx̄)...)
    ȳ = tuple(map(cotangent, xx̄)...)
    function tuple_pb!!(ȳ::Tuple, ::NoTangent, x̄...)
        return NoTangent(), map(increment!!, x̄, ȳ)...
    end
    return Dual(y, ȳ), tuple_pb!!
end

isprimitive(::RMADContext, ::typeof(getfield), x, f) = true
function taped_rrule!!(::Dual{typeof(getfield)}, x, f)
    fname = primal(f)
    yȳ = Dual(getfield(primal(x), fname), getproperty(cotangent(x), fname))
    function getfield_pb!!(ȳ, ::NoTangent, x̄, ::NoTangent)
        x̄ = setproperty(x̄, fname, increment!!(getproperty(x̄, fname), ȳ))
        return NoTangent(), x̄, NoTangent()
    end
    return yȳ, getfield_pb!!
end
function taped_rrule!!(::Dual{typeof(getfield)}, x::Dual{<:Tuple}, index)
    yȳ = Dual(primal(x)[primal(index)], cotangent(x)[primal(index)])
    function getfield_tuple_pb!!(ȳ, ::NoTangent, x̄::Tuple, ::NoTangent)
        x̄ = ntuple(n -> n == primal(index) ? increment!!(x̄[n], ȳ) : x̄[n], length(x̄))
        return NoTangent(), x̄, NoTangent()
    end
    return yȳ, getfield_tuple_pb!!
end

isprimitive(::RMADContext, ::typeof(===), a, b) = true
function taped_rrule!!(::Dual{typeof(===)}, a, b)
    yȳ = Dual(===(primal(a), primal(b)), NoTangent())
    egal_pb!!(::NoTangent, ::NoTangent, ā, b̄) = NoTangent(), ā, b̄
    return yȳ, egal_pb!!
end

isprimitive(::RMADContext, ::typeof(typeof), x) = true
function taped_rrule!!(::Dual{typeof(typeof)}, x)
    yȳ = Dual(typeof(primal(x)), NoTangent())
    typeof_pb!!(::NoTangent, ::NoTangent, x̄) = NoTangent(), x̄
    return yȳ, typeof_pb!!
end

isprimitive(::RMADContext, ::typeof(fieldtype), T, fname) = true
function taped_rrule!!(::Dual{typeof(fieldtype)}, T, fname)
    yȳ = Dual(fieldtype(primal(T), primal(fname)), NoTangent())
    function fieldtype_pb!!(::NoTangent, ::NoTangent, ::NoTangent, ::NoTangent)
        return NoTangent(), NoTangent(), NoTangent()
    end
    return yȳ, fieldtype_pb!!
end

isprimitive(::RMADContext, ::typeof(setfield!), value, name, x) = true
function taped_rrule!!(::Dual{typeof(setfield!)}, value, name, x)
    fname = primal(name)
    old_v = getfield(primal(value), fname)
    old_v̄ = getproperty(cotangent(value), fname)
    yȳ = Dual(
        setfield!(primal(value), fname, primal(x)),
        setproperty(cotangent(value), fname, cotangent(x)),
    )
    function setfield!_pb!!(ȳ::Tangent, ::NoTangent, v̄::Tangent, ::NoTangent, x̄)
        x̄ = increment!!(x̄, getproperty(ȳ, fname))
        setfield!(primal(value), fname, old_v)
        setproperty(cotangent(value), fname, old_v̄)
        v̄ = increment!!(v̄, ȳ)
        return NoTangent(), v̄, NoTangent(), x̄
    end
    return yȳ, setfield!_pb!!
end


# Rules to work around limitations in Umlaut.

# Rules for differentiable functions.

for (M, f, arity) in DiffRules.diffrules(; filter_modules=[:Base])
    pb_name = Symbol(string(gensym(f)) * "_pullback")
    if arity == 1
        @eval isprimitive(::RMADContext, ::typeof($M.$(f)), ::Float64) = true
        @eval function taped_rrule!!(::Dual{typeof($M.$(f))}, xx̄::Dual{Float64})
            y, pb = rrule($M.$(f), primal(xx̄))
            function $pb_name(ȳ::Float64, ::NoTangent, x̄::Float64)
                _, x̄_inc = pb(ȳ)
                return NoTangent(), x̄ + x̄_inc
            end
            return Dual(y, zero_cotangent(y)), $pb_name
        end
    elseif arity == 2
        @eval isprimitive(::RMADContext, ::typeof($M.$(f)), ::Float64, ::Float64) = true
        @eval function taped_rrule!!(
            ::Dual{typeof($M.$(f))}, xx̄::Dual{Float64}, yȳ::Dual{Float64}
        )
            z, pb = rrule($M.$(f), primal(xx̄), primal(yȳ))
            function $pb_name(z̄, ::NoTangent, x̄::Float64, ȳ::Float64)
                _, x̄_inc, ȳ_inc = pb(z̄)
                return NoTangent(), x̄ + x̄_inc, ȳ + ȳ_inc
            end
            return Dual(z, zero_cotangent(z)), $pb_name
        end
    else
        throw(error("Expected arity = 1 or 2, got $arity"))
    end
end

ChainRulesCore.@non_differentiable verify(::ConditionalCheck, ::Any)

function Umlaut.inputs!(tape::Tape{<:RMADContext}, vals...)
    @assert(isempty(tape) || length(inputs(tape)) == length(vals) || get(tape.meta, :isva, false),
            "This tape contains $(length(inputs(tape))) inputs, but " *
            "$(length(vals)) value(s) were provided")
    if isempty(tape)
        # initialize inputs
        for val in vals
            push!(tape, Input(val))
        end
    else
        # rewrite input values
        if get(tape.meta, :isva, false)
            # group varargs into a single tuple
            nargs = length(inputs(tape))
            vals = (vals[1:nargs - 1]..., vals[nargs:end])
        end
        for (i, val) in enumerate(vals)
            tape[Umlaut.V(i)].val = val
        end
    end
    vs = [Umlaut.V(op) for op in tape.ops[1:length(vals)]]

    # Push stuff onto the tape.
    for (v, val) in zip(vs, vals)
        assign_cotangent_index!(tape.c, v)
        push!(tape.c.pbs, Operation(Dual(val, zero_cotangent(val)), nothing, ()))
    end

    return vs
end

eval_boundscheck(x) = x
function eval_boundscheck(x::Expr)
    println("in the boundschecking")
    @show x.head
    if x.head === :boundscheck
        @show "evaling"
        return false
    else
        return x
    end
end

function Umlaut.trace_call!(t::Umlaut.Tracer{C}, vs...) where {C<:RMADContext}
    fargs = Umlaut.var_values(vs)

    # Handle boundschecking
    fargs = map(eval_boundscheck, fargs)

    # I have no idea how to handle boundschecks properly.
    # Do usual stuff.
    if Umlaut.isprimitive(t.tape.c, fargs...) && !Umlaut.is_ho_tracable(t.tape.c, fargs...)
        return Umlaut.record_primitive!(t.tape, vs...)
    else
        return Umlaut.trace!(t, vs)
    end
end

function Umlaut.record_primitive!(tape::Tape{<:RMADContext}, v_fargs...)
    line = get(tape.meta, :line, nothing)
    fargs = [v isa Umlaut.V ? tape[v].val : v for v in v_fargs]
    fargs_dual = map((x, v_x) -> Dual(x, get_cotangent(tape.c, v_x)), fargs, v_fargs)
    rrule_output = taped_rrule!!(fargs_dual...)
    if rrule_output === nothing
        throw(error("No rrule for primitive $((fargs..., ))"))
    else
        yȳ, pb!! = rrule_output
        op = Operation(yȳ, pb!!, v_fargs)
        push!(tape.c.pbs, op)
        v_y = push!(tape, mkcall(v_fargs...; line=line, val=primal(yȳ)))
        assign_cotangent_index!(tape.c, v_y)
        return v_y
    end
end

function run_reverse_pass!(ctx::RMADContext, tape::Umlaut.Tape)

    # Seed the tape.
    v_y = tape.result
    set_cotangent!(ctx, v_y, 1.0)

    # Run the reverse-pass.
    for n in reverse(eachindex(ctx.pbs))
        op = ctx.pbs[n]
        is_input(op) && continue
        ȳ = cotangent(op.yȳ)
        x̄s = map(Base.Fix1(get_cotangent, ctx), op.v_fargs)
        new_x̄s = op.pb!!(ȳ, x̄s...)
        foreach((v, x̄) -> set_cotangent!(ctx, v, x̄), op.v_fargs, new_x̄s)
    end
    return nothing
end

function value_and_derivative(f, x::Float64)

    # Run forwards-pass.
    ctx = RMADContext()
    y, tape = trace(f, x; ctx)

    # Check that output is a Float64, and inform the user if it is not.
    y isa Float64 || throw(error("f(x) must return a Float64, but found a $(typeof(y))"))

    # Run reverse-pass.
    run_reverse_pass!(ctx, tape)
    return y, get_cotangent(ctx, inputs(tape)[2])
end
