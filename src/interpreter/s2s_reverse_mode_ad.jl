#=
    LineToADDataMap

The "AD data associated to line id" is all of the data that is shared between the forwards-
and reverse-passes associated to line `id` in an `BBCode`.

An `LineToADDataMap` is characterised by a `Dict{ID, Int}`. Each `key` the `ID` of a line
number in the primal `BBCode`, and each `value` is the unique position associated to it in a
`Tuple` which gets shared between the forwards- and reverse-passes.

This map will generally only have keys for a subset of the `ID`s in the primal `BBCode`
because many of the lines in primal `BBCode` do not need to share between the forwards- and
reverse-passes. This means that we can keep the size of the `Tuple` that must be shared
between the forwards- and reverse-passes as small as possible. For example, `PhiNode`s and
terminators never need to share information, nor do `:invoke` expressions which provably
have the `NoPullback` pullback.
=#
struct LineToADDataMap
    m::Dict{ID, Int}
    LineToADDataMap() = new(Dict{ID, Int}())
end

#=
    get_storage_location!(m::LineToADDataMap, line::ID)

Return the location in the `Tuple` shared between the forwards- and reverse-passes
associated to line `line`. If `m` does not already have an entry for `line`,
create one, insert it into `m`, and return it.
=#
function get_storage_location!(m::LineToADDataMap, line::ID)
    (line in keys(m.m)) || setindex!(m.m, length(m.m) + 1, line)
    return m.m[line]
end

#=
    ADInfo(terminator_block_id::ID, line_map::LineToADDataMap)

This data structure is used to hold global information which gets passed around, in
particular to `make_ad_stmts!`.

- `interp`: a `TapedInterpreter`.
- `terminator_block_id`: the ID of the block inserted to provide a unique exit point on the
    forwards-pass.
- `line_map`: a `LineToADDataMap`.
- `arg_types`: a map from `Argument` to its static type.
- `ssa_types`: a map from `ID` associated to lines to their static type.
=#
struct ADInfo
    interp::TInterp
    terminator_block_id::ID
    line_map::LineToADDataMap
    arg_types::Dict{Argument, Any}
    ssa_types::Dict{ID, Any}
end

# Returns the statically-inferred type associated to `x`.
get_primal_type(info::ADInfo, x::Argument) = info.arg_types[x]
get_primal_type(info::ADInfo, x::ID) = info.ssa_types[x]
get_primal_type(::ADInfo, x::QuoteNode) = _typeof(x.value)
get_primal_type(::ADInfo, x) = _typeof(x)
function get_primal_type(::ADInfo, x::GlobalRef)
    return isconst(x) ? _typeof(getglobal(x.mod, x.name)) : x.binding.ty
end

#=
    ADStmtInfo

Data structure which contains the result of `make_ad_stmts!`. Fields are
- `fwds`: the instruction which runs the forwards-pass of AD
- `rvs`: the instruction which runs the reverse-pass of AD / the pullback
- `data`: data which must be made available to the forwards- and reverse-pass of AD

For `rvs`, a value of `nothing` indicates that there should be no instruction associated
to the primal statement in the pullback.

For `data`, a value of `nothing` indicates that there is no data that needs to be shared
between the forwards-pass and pullback, and that there is no need to allocate an element of
the tuple which is shared between the forwards-pass and pullback to this primal line.
=#
struct ADStmtInfo
    fwds
    rvs
    data
end

#=
    make_ad_stmts(stmt, id::ID, info::ADInfo)::ADStmtInfo

Every line in the primal code is associated to exactly one line in the forwards-pass of AD,
and either one or zero lines in the pullback (many nodes do not need to appear in the
pullback at all). This function specifies this translation for every type of node.

Translates the statement `stmt`, associated to `id` in the primal, into a specification of
what should happen for this statement in the forwards- and reverse-passes of AD, and what
data should be shared between the forwards- and reverse-passes. Returns this in the form of
an `ADStmtInfo`.

`info` is a data structure containing various bits of global information that certain types
of nodes need access to.
=#
function make_ad_stmts! end

# `nothing` as a statement in Julia IR indicates the presence of a line which will later be
# removed. We emit a no-op on both the forwards- and reverse-passes. No shared data.
make_ad_stmts!(::Nothing, ::ID, ::ADInfo) = ADStmtInfo(nothing, nothing, nothing)

# If stmt.val is defined, then we have a regular return node, and should replace it with a
# goto to the exit block on the forwards pass, and a no-op on the reverse-pass. If
# stmts.val is not defined, then we have an unreachable node, and it should be left alone on
# the forwards-pass, and be a no-op on the reverse-pass. An unreachable node can occur, for
# example, immediately after a throw statement, because the compiler can be certain that
# execution will end at the throw statement. No shared data.
function make_ad_stmts!(stmt::ReturnNode, ::ID, info::ADInfo)
    if isdefined(stmt, :val)
        return ADStmtInfo(IDGotoNode(info.terminator_block_id), nothing, nothing)
    else
        return ADStmtInfo(stmt, nothing, nothing)
    end
end

# Identity forwards-pass, no-op reverse. No shared data.
make_ad_stmts!(stmt::IDGotoNode, ::ID, ::ADInfo) = ADStmtInfo(stmt, nothing, nothing)

# Identity forwards-pass, no-op reverse. No shared data.
make_ad_stmts!(stmt::IDGotoIfNot, ::ID, ::ADInfo) = ADStmtInfo(stmt, nothing, nothing)

# Identity forwards-pass, no-op reverse. No shared data.
make_ad_stmts!(stmt::IDPhiNode, ::ID, ::ADInfo) = ADStmtInfo(stmt, nothing, nothing)

function make_ad_stmts!(stmt::PiNode, line::ID, info::ADInfo)

end

# Replace statement with construction of zero `CoDual`. No shared data.
function make_ad_stmts!(stmt::GlobalRef, ::ID, ::ADInfo)
    return ADStmtInfo(Expr(:call, Taped.zero_codual, stmt), nothing, nothing)
end

# Replace statement with quote node for zero `CoDual`. No shared data.
function make_ad_stmts!(stmt::QuoteNode, ::ID, ::ADInfo)
    return ADStmtInfo(QuoteNode(zero_codual(stmt.value)), nothing, nothing)
end

# Literal statement. Replace with zero `CoDual`. For example, `5` becomes a quote node
# containing `CoDual(5, NoTangent())`, and `5.0` becomes a quote node `CoDual(5.0, 0.0)`.
# No shared data.
function make_ad_stmts!(stmt, ::ID, ::ADInfo)
    return ADStmtInfo(QuoteNode(zero_codual(stmt)), nothing, nothing)
end

# Taped does not yet handle `PhiCNode`s. Throw an error if one is encountered.
function make_ad_stmts!(stmt::Core.PhiCNode, ::ID, ::ADInfo)
    throw(error("Encountered PhiCNode: $stmt. Taped cannot yet handle such nodes."))
end

# Taped does not yet handle `UpsilonNode`s. Throw an error if one is encountered.
function make_ad_stmts!(stmt::Core.UpsilonNode, ::ID, ::ADInfo)
    throw(error("Encountered UpsilonNode: $stmt. Taped cannot yet handle such nodes."))
end

# There are quite a number of possible `Expr`s that can be encountered. Each case has its
# own comment, explaining what is going on.
function make_ad_stmts!(stmt::Expr, line::ID, info::ADInfo)
    is_invoke = Meta.isexpr(stmt, :invoke)
    if Meta.isexpr(stmt, :call) || is_invoke

        # Find the types of all arguments to this call / invoke.
        args = is_invoke ? stmt.args[2:end] : stmt.args
        arg_types = map(arg -> get_primal_type(info, arg), (args..., ))

        # Construct signature, and determine how the rrule is to be computed.
        rule = if is_invoke
            build_rrule(info.interp, Tuple{arg_types...})
        else
            error("Booo dynamic dispatch")
        end

        # Create data shared between the forwards- and reverse-passes.
        data = (
            rule=rule,
            pb_stack=build_pb_stack(rule, arg_types),
            my_tangent_stack=make_tangent_stack(get_primal_type(info, line)),
            arg_tangent_stacks=map(make_tangent_ref_stack ∘ tangent_ref_type_ub, arg_types),
        )

        # Get a location in the global captures in which `data` can live.
        capture_index = get_storage_location!(info.line_map, line)

        # Create a call to `fwds_pass!`, which runs the forwards-pass. `Argument(1)` always
        # contains the global collection of captures.
        fwds = Expr(:call, Taped.fwds_pass!, Argument(1), capture_index, args...)
        rvs = Expr(:call, Taped.rvs_pass!, Argument(1), capture_index)
        return ADStmtInfo(fwds, rvs, data)

    elseif Meta.isexpr(stmt, :throw_undef_if_not)
        # Expr(:throw_undef_if_not, name, cond) raises an error if `cond` evaluates to
        # false. `cond` will be a codual on the forwards-pass, so have to get its primal.
        fwds = Expr(:call, Taped.__throw_undef_if_not, stmt.args...)
        return ADStmtInfo(fwds, nothing, nothing)

    elseif stmt.head in [
        :boundscheck,
        :code_coverage_effect,
        :gc_preserve_begin,
        :gc_preserve_end,
        :loopinfo,
        :leave,
        :pop_exception,
    ]
        # Expressions which do not require any special treatment.
        return ADStmtInfo(stmt, nothing, nothing)

    else
        # Encountered an expression that we've not seen before.
        throw(error("Unrecognised expression $stmt"))
    end
end

function build_pb_stack(rule, arg_types)
    codual_sig = Tuple{map(codual_type, arg_types)...}
    possible_output_types = Base.return_types(rule, codual_sig)
    if length(possible_output_types) == 0
        throw(error("No return type inferred for rule with sig $codual_sig"))
    elseif length(possible_output_types) > 1
        @warn "Too many output types inferred"
        display(possible_output_types)
        println()
        throw(error("> 1 return type inferred for rule with sig $codual_sig "))
    end
    T_pb!! = only(possible_output_types)
    if T_pb!! <: Tuple && T_pb!! !== Union{}
        F = T_pb!!.parameters[2]
        return Base.issingletontype(F) ? SingletonStack{F}() : Stack{F}()
    else
        return Stack{Any}()
    end
end

# Used in `make_ad_stmts!` method for `Expr(:call, ...)` and `Expr(:invoke, ...)`.
#
# Executes the fowards-pass. `data` is the data shared between the forwards-pass and
# pullback. It must be a `NamedTuple` with fields `arg_tangent_stacks`, `rule`,
# `my_tangent_stack`, and `pb_stack`.
@inline function fwds_pass!(captures, capture_index::Int, raw_args...)

    # Extract this rules data from the global collection of captures.
    data = captures[capture_index]

    # Make anything that is not already a `CoDual` into one.
    args = tuple_map(x -> isa(x, CoDual) ? x : uninit_codual(x), raw_args)

    # Log the location of the tangents associated to each argument.
    tuple_map(data.arg_tangent_stacks, arg_slots) do tangent_stack_stack, arg
        push!(tangent_stack_stack, get_tangent_stack(arg))
    end

    # Run the rule.
    out, pb!! = data.rule(zero_codual(evaluator), args...)

    # Log the results and return.
    my_tangent_stack = data.my_tangent_stack
    push!(my_tangent_stack, tangent(out))
    push!(data.pb_stack, pb!!)
    return (out, top_ref(my_tangent_stack))
end

# Used in `make_ad_stmts!` method for `Expr(:call, ...)` and `Expr(:invoke, ...)`.
#
# Executes the reverse-pass. `data` is the `NamedTuple` shared with `fwds_pass!`.
# Much of this pass will be optimised away in practice.
@inline function rvs_pass!(captures, capture_index::Int)

    # Extract this rules data from the global collection of captures.
    data = captures[capture_index]

    # Get the tangent w.r.t. output, and the pullback, from this instructions' stacks.
    dout = pop!(data.my_tangent_stack)
    pb!! = pop!(data.pb_stack)

    # Get the tangent w.r.t. each argument of the primal.
    tangent_stacks = tuple_map(pop!, data.arg_tangent_stacks)

    # Run the pullback and increment the argument tangents.
    new_dargs = pb!!(dout, tuple_map(set_immutable_to_zero ∘ getindex, tangent_stacks)...)
    tuple_map(increment_ref!, tangent_stacks, new_dargs)

    return nothing
end

# Used in `make_ad_stmts!` method for `Expr(:throw_undef_if_not, ...)`.
@inline function __throw_undef_if_not(slotname::Symbol, cond_codual::CoDual)
    primal(cond_codual) || throw(UndefVarError(slotname))
    return nothing
end

"""
    build_rrule(interp::TInterp{C}, sig::Type{<:Tuple}) where {C}

Returns and `OpaqueClosure` which is an `rrule!!` for `sig` in context `C`.
"""
function build_rrule(interp::TInterp{C}, sig::Type{<:Tuple}) where {C}

    # If we have a hand-coded rule, always use that.
    is_primitive(C, sig) && return rrule!!

    # Grab code associated to the primal.
    ir, Treturn = lookup_ir(interp, sig)

    # Normalise the IR.
    is_vararg, spnames = is_vararg_sig_and_sparam_names(sig)
    ir = normalise!(ir, spnames)


end

