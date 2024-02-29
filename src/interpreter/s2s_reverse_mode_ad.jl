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
    ADInfo

This data structure is used to hold global information which gets passed around, in
particular to `make_ad_stmts!`.

- `interp`: a `TapedInterpreter`.
- `block_stack_id`: the ID associated to the block stack -- the stack which keeps track of
    which blocks we visited during the forwards-pass, and which is used on the reverse-pass
    to determine which blocks to visit. The location in the shared data storage associated
    to this can be retrieved using `block_stack_index`.
- `line_map`: a `LineToADDataMap`.
- `arg_types`: a map from `Argument` to its static type.
- `ssa_types`: a map from `ID` associated to lines to their static type.
=#
struct ADInfo
    interp::TInterp
    block_stack_id::ID
    line_map::LineToADDataMap
    arg_types::Dict{Argument, Any}
    ssa_types::Dict{ID, Any}
end

# The construct that you should use for ADInfo -- the fields that you don't need to provide
# to this constructor can be automatically generated.
function ADInfo(interp::TInterp, arg_types::Dict{Argument, Any}, ssa_types::Dict{ID, Any})
    return ADInfo(interp, ID(), LineToADDataMap(), arg_types, ssa_types)
end

# Returns the index in the shared data tuple which contains the block stack.
function block_stack_index(info::ADInfo)
    return get_storage_location!(info.line_map, info.block_stack_id)
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

# Identity forwards-pass, no-op reverse. No shared data.
make_ad_stmts!(stmt::ReturnNode, ::ID, info::ADInfo) = ADStmtInfo(stmt, nothing, nothing)

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
            pb_stack=build_pb_stack(_typeof(rule), arg_types),
            my_tangent_stack=make_tangent_stack(get_primal_type(info, line)),
            arg_tangent_stacks=map(make_tangent_ref_stack ∘ tangent_ref_type_ub, arg_types),
        )

        # Get a location in the global captures in which `data` can live.
        capture_index = get_storage_location!(info.line_map, line)

        # Create a call to `fwds_pass!`, which runs the forwards-pass. `Argument(1)` always
        # contains the global collection of captures.
        fwds = Expr(:call, __fwds_pass!, Argument(1), capture_index, args...)
        rvs = Expr(:call, __rvs_pass!, Argument(1), capture_index)
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

function build_pb_stack(Trule, arg_types)
    T_pb!! = Core.Compiler.return_type(Tuple{Trule, map(codual_type, arg_types)...})
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
@inline function __fwds_pass!(captures, capture_index::Int, raw_args...)

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
@inline function __rvs_pass!(captures, capture_index::Int)

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

#
# Runners for generated code.
#

struct Pullback{Tpb, Tret_ref}
    pb::Tpb
    ret_ref::Tret_ref
end

@inline function (pb::Pullback{P, Q})(dy, dargs::Vararg{Any, N}) where {P, Q, N}
    increment_ref!(pb.ret_ref, dy)
    return pb.pb(dargs...)
end

struct ForwardsPass{Tfwds_oc, Targ_tangent_stacks, Tpb}
    fwds_oc::Tfwds_oc
    arg_tangent_stacks::Targ_tangent_stacks
    pb::Tpb
end

@inline function (fwds::ForwardsPass{P, Q, S})(args::Vararg{Any, N}) where {P, Q, S, N}
    return fwds.fwds_oc(args...)[1], fwds.pb
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

    # Convert IR to BBCode.
    primal_ir = BBCode(ir)

    # Compute global info.
    arg_types = Dict{Argument, Any}(
        map(((n, t),) -> (Argument(n) => _get_type(t)), enumerate(ir.argtypes))
    )
    ssa_types = Dict{ID, Any}(map(t -> (ID(), _get_type(t)), ir.stmts.type))
    info = ADInfo(interp, arg_types, ssa_types)

    # For each block in the fwds and pullback BBCode, translate all statements.
    ad_stmts_blocks = map(primal_ir.blocks) do primal_blk
        ids = concatenate_ids(primal_blk)
        primal_stmts = concatenate_stmts(primal_blk)
        return (primal_blk.id, tuple.(ids, make_ad_stmts!.(primal_stmts, ids, Ref(info))))
    end

    # Construct captures for forwards-pass and pullback.
    shared_data_type = compute_shared_data_types(info, ad_stmts_blocks, Stack{Int})
    display(shared_data_type)

    # Construct BBCode for forwards-pass and pullback.
    fwds_ir = forwards_pass_ir(primal_ir, ad_stmts_blocks, info, shared_data_type)
    pb_ir = pullback_ir(primal_ir, Treturn, ad_stmts_blocks, info, shared_data_type)
    display(IRCode(fwds_ir))
    display(optimise_ir!(IRCode(fwds_ir)))

    arg_tangent_stacks = (map(make_tangent_stack ∘ _get_type, primal_ir.argtypes)..., )

    # Construct + return OpaqueClosures.
    return nothing
end

const ADStmts = Vector{Tuple{ID, Vector{Tuple{ID, ADStmtInfo}}}}

# Compute the type of the captured variables in the forwards-pass and pullback.
function compute_shared_data_types(info::ADInfo, ad_stmts_blocks::ADStmts, Tblock_stack)

    # Build map from ID to type.
    tmp = map(ad_stmts -> map(x -> (x[1], x[2].data), ad_stmts[2]), ad_stmts_blocks)
    id_to_type = Dict{ID, Any}([(info.block_stack_id, Tblock_stack), reduce(vcat, tmp)...])
    id_to_type = filter(p -> p.second !== nothing, id_to_type)

    # Build map from index to type.
    index_to_type = Dict(
        (get_storage_location!(info.line_map, id), id_to_type[id]) for id in keys(id_to_type)
    )

    # Compute Tuple type.
    return Tuple{map(last, sort(collect(index_to_type); by=x->x.first))...}
end

# Compute the type of the OpaqueClosure which runs the pullback.
function compute_pb_type(primal_arg_types, primal_return_type)
    arg_tangent_types = map(tangent_type ∘ _get_type, primal_arg_types)
    pb_arg_types = Tuple{tangent_type(primal_return_type), arg_tangent_types...}
    pb_ret_type = Tuple{arg_tangent_types...}
    return OpaqueClosure{pb_arg_types, pb_ret_type}
end

function forwards_pass_ir(ir::BBCode, ad_stmts_blocks::ADStmts, info::ADInfo, Tshared_data)

    # Construct augmented version of each basic block from the primal. For each block:
    # 1. pull the translated basic block statements from ad_stmts_blocks.
    # 2. insert a statement which logs the ID of the current block to the block stack.
    # 3. construct and return a BBlock.
    n = block_stack_index(info)
    blocks = map(ad_stmts_blocks) do (block_id, ad_stmts)
        fwds_stmts = Tuple{ID, Any}[(x[1], x[2].fwds) for x in ad_stmts]
        ins_loc = length(fwds_stmts) + (isa(fwds_stmts[end][2], Terminator) ? 0 : 1)
        ins_stmt = (ID(), Expr(:call, __push_block_stack!, Argument(1), n, block_id.id))
        return BBlock(block_id, insert!(fwds_stmts, ins_loc, ins_stmt))
    end

    # Create and return the `BBCode` for the forwards-pass.
    arg_types = vcat(Tshared_data, map(fwds_pass_arg_type ∘ _get_type, ir.argtypes))
    return BBCode(blocks, arg_types, ir.sptypes, ir.linetable, ir.meta)
end

# push a block index `id` onto the block_stack, which is found in `captures[n]`.
@inline __push_block_stack!(captures, n::Int, id::Int) = push!(captures[n], id)

fwds_pass_arg_type(P) = Tuple{codual_type(P), tangent_stack_type(P)}

function pullback_ir(ir::BBCode, Tret, ad_stmts_blocks::ADStmts, info::ADInfo, Tshared_data)

    # Create entry block, which pops the block_stack, and switches to whichever block we
    # were in at the end of the forwards-pass.
    entry_block = nothing

    # For each basic block in the primal:
    # 1. pull the translated basic block statements from ad_stmts_blocks
    # 2. reverse the statements
    # 3. pop block stack to get the predecessor block
    # 4. insert a switch statement to determine which block to jump to. Restrict blocks
    #   considered to only those which are predecessors of this one. If in the first block,
    #   check whether or not the block stack is empty. If empty, jump to the exit block.
    exit_block_id = ID()
    main_blocks = nothing

    # Create an exit block. Simply returns.
    exit_block = BBlock(exit_block_id, Tuple{ID, Any}[(ID(), ReturnNode(nothing))])

    # Create and return `BBCode` for the pullback.
    blocks = vcat(entry_block, main_blocks, exit_block)
    darg_types = map(tangent_type ∘ _get_type, ir.argtypes)
    arg_types = vcat(Tshared_data, tangent_type(Tret), darg_types)
    return BBCode(blocks, arg_types, ir.sptypes, ir.linetable, ir.meta)
end
