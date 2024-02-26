_id_count::Int = 0

"""
    ID()

An `ID` (read: unique name) is just a wrapper around an `Int`. Uniqueness is ensured via a
global counter, which is incremented each time that an `ID` is created.
"""
struct ID
    id::Int
    function ID()
        global _id_count += 1 
        return new(_id_count)
    end
end

Base.copy(id::ID) = id

"""
    IDPhiNode(edges::Vector{ID}, values::Vector{Any})

Like a `PhiNode`, but `edges` are `ID`s rather than `Int32`s.
"""
struct IDPhiNode
    edges::Vector{ID}
    values::Vector{Any}
end

Base.copy(node::IDPhiNode) = IDPhiNode(copy(node.edges), copy(node.values))

"""
    IDGotoNode(label::ID)

Like a `GotoNode`, but `label` is an `ID` rather than an `Int64`.
"""
struct IDGotoNode
    label::ID
end

Base.copy(node::IDGotoNode) = IDGotoNode(copy(node.label))

"""
    IDGotoIfNot(cond::Any, dest::ID)

Like a `GotoIfNot`, but `dest` is an `ID` rather than an `Int64`.
"""
struct IDGotoIfNot
    cond::Any
    dest::ID
end

Base.copy(node::IDGotoIfNot) = IDGotoIfNot(copy(node.cond), copy(node.dest))

"""
    Switch(conds::Vector{Any}, dests::Vector{ID})

A switch-statement node. This can be placed (temporarily) in Julia IR. Has the following
semantics:
```julia
goto dests[1] if not conds[1]
goto dests[2] if not conds[2]
...
goto dests[N] if not conds[N]
```
where the value associated to each element of `conds` is a `Bool`, and `dests` indicate
which block to jump to. If none of the conditions are met, then it falls through to the
next block.

`Switch` statements are lowered into the above sequence of `GotoIfNot` statements before
converting `BBCode` back into `IRCode`.
"""
struct Switch
    conds::Vector{Any}
    dests::Vector{ID}
    function Switch(conds::Vector{Any}, dests::Vector{ID})
        @assert length(conds) == length(dests)
        return new(conds, dests)
    end
end

"""
    Terminator = Union{Switch, IDGotoIfNot, IDGotoNode, ReturnNode}

A Union of the possible types of a terminator node.
"""
const Terminator = Union{Switch, IDGotoIfNot, IDGotoNode, ReturnNode}

"""
    BBlock(
        id::ID,
        phi_nodes::Vector{Tuple{ID, IDPhiNode}},
        stmts::Vector{<:Tuple{ID, Any}},
        terminator::Union{Nothing, Terminator},
    )

A basic block data structure (not called `BasicBlock` to avoid accidental confusion with
`CC.BasicBlock`). Forms a single basic block from a sequence of `phi_nodes`, followed by
a sequence of `stmts` (none of which are permitted to be either a `PhiNode` or a
`Terminator`), and finally a `terminator`. If `terminator` is `Nothing`, then there is no
terminator, and this block always falls through to the next block.

Each `BBlock` has an `ID` (a unique name). This makes it possible to refer to blocks in a
way that does not change when additional `BBlocks` are inserted into a `BBCode`.
This differs from the positional block numbering found in `IRCode`, in which the number
associated to a basic block changes when new blocks are inserted.

Note that `PhiNode`s, `GotoIfNot`s, and `GotoNode`s should all be replaced with their
`IDPhiNode`, `IDGotoIfNot`, and `IDGotoNode` equivalents.
"""
struct BBlock
    id::ID
    phi_nodes::Vector{Tuple{ID, IDPhiNode}}
    stmts::Vector{Tuple{ID, Any}}
    terminator::Union{Nothing, Terminator}
    function BBlock(
        id::ID,
        phi_nodes::Vector{Tuple{ID, IDPhiNode}},
        stmts::Vector{<:Tuple{ID, Any}},
        terminator::Union{Nothing, Terminator},
    )
        @assert all(x -> !isa(x[2], Union{PhiNode, Terminator}), stmts)
        return new(id, phi_nodes, stmts, terminator)
    end
end

has_terminator(bb::BBlock) = bb.terminator !== nothing

Base.length(bb::BBlock) = length(bb.phi_nodes) + length(bb.stmts) + Int(has_terminator(bb))

function Base.copy(bb::BBlock)
    new_terminator = has_terminator(bb) ? copy(bb.terminator) : nothing
    return BBlock(bb.id, copy(bb.phi_nodes), copy(bb.stmts), new_terminator)
end

function concatenate_ids(bb::BBlock)
    terminator_id = has_terminator(bb) ? [ID()] : ID[] 
    return vcat(first.(bb.phi_nodes), first.(bb.stmts), terminator_id)
end

function concatenate_stmts(bb::BBlock)
    terminator = has_terminator(bb) ? [bb.terminator] : Any[]
    return vcat(last.(bb.phi_nodes), last.(bb.stmts), terminator)
end

first_id(bb::BBlock) = first(concatenate_ids(bb))

"""
    BBCode(
        blocks::Vector{BBlock}
        argtypes::Vector{Any}
        sptypes::Vector{CC.VarState}
        linetable::Vector{Core.LineInfoNode}
        meta::Vector{Expr}
    )

TODO: WRITE THIS DOCSTRING
"""
struct BBCode
    blocks::Vector{BBlock}
    argtypes::Vector{Any}
    sptypes::Vector{CC.VarState}
    linetable::Vector{Core.LineInfoNode}
    meta::Vector{Expr}
end

"""
    BBCode(ir::Union{IRCode, BBCode}, new_blocks::Vector{Block})

Make a new `BBCode` whose `blocks` is given by `new_blocks`, and fresh copies are made of
all other fields from `ir`.
"""
function BBCode(ir::Union{IRCode, BBCode}, new_blocks::Vector{BBlock})
    return BBCode(
        new_blocks,
        CC.copy(ir.argtypes),
        CC.copy(ir.sptypes),
        CC.copy(ir.linetable),
        CC.copy(ir.meta),
    )
end

function predecessors(block::BBlock, ir::BBCode)
    return reduce(
        vcat,
        map(b -> is_successor(b, block.id, is_next(b, block, ir)) ? [b.id] : [], ir.blocks),
    )
end

function is_next(block::BBlock, other_block::BBlock, ir::BBCode)
    return location(block, ir) + 1 == location(other_block, ir)
end

location(block::BBlock, ir::BBCode) = findfirst(b -> b.id == block.id, ir.blocks)

is_successor(b::BBlock, id::ID, is_next::Bool) = is_successor(b.terminator, id, is_next)
is_successor(::Nothing, ::ID, is_next::Bool) = is_next
is_successor(x::IDGotoNode, id::ID, is_next::Bool) = x.label == id
is_successor(x::IDGotoIfNot, id::ID, is_next::Bool) = is_next || x.dest == id
is_successor(::ReturnNode, ::ID, ::Bool) = false
is_successor(x::Switch, id::ID, is_next::Bool) = is_next || any(==(id), x.conds)

find_block_ind(ir::BBCode, id::ID) = findfirst(b -> b.id == id, ir.blocks)

"""
    collect_stmts(ir::BBCode)

Produce a `Vector{Any}` containing all of the statements in `ir`. These are returned in
order, so it is safe to assume that element `n` refers to the `nth` element of the `IRCode`
associated to `ir`.
"""
function collect_stmts(ir::BBCode)::Vector{Any}
    return reduce(
        vcat,
        map(ir.blocks) do b
            if has_terminator(b)
                return vcat(b.phi_nodes, b.stmts, b.terminator)
            else
                vcat(b.phi_nodes, b.stmts)
            end
        end,
    )
end

"""
    id_to_line_map(ir::BBCode)

Produces a `Dict` mapping from each `ID` associated with a line in `ir` to its line number.
This is isomorphic to mapping to its `SSAValue` in `IRCode`. Terminators do not have `ID`s
associated to them, so not every line in the original `IRCode` is mapped to.
"""
function id_to_line_map(ir::BBCode)
    lines = collect_stmts(ir)
    lines_and_line_numbers = filter(
        x -> isa(x[1], Tuple) && x[1] !== nothing, collect(zip(lines, eachindex(lines)))
    )
    ids_and_line_numbers = map(x -> (x[1][1], x[2]), lines_and_line_numbers)
    return Dict(ids_and_line_numbers)
end

#
# Converting from IRCode to BBCode
#

"""
    BBCode(ir::IRCode)

Convert an `ir` into a `BBCode`. Creates a completely inependent data structure, so mutating
the `BBCode` returned will not mutate `ir`.

All `PhiNode`s, `GotoIfNot`s, and `GotoNode`s will be replaced with the `IDPhiNode`s,
`IDGotoIfNot`s, and `IDGotoNode`s respectively.
"""
function BBCode(ir::IRCode)

    # Produce a new set of statements with `IDs` rather than `SSAValues` and block numbers.
    ssa_ids, stmts = _ssas_to_ids(ir.stmts.inst)
    block_ids, stmts = _block_nums_to_ids(stmts, ir.cfg)

    # Chop up the new statements into `BBlocks`, according to the `CFG` in `ir`.
    blocks = map(zip(ir.cfg.blocks, block_ids)) do (bb, id)
        all_ids = ssa_ids[bb.stmts]
        all_stmts = stmts[bb.stmts]

        has_terminator = all_stmts[end] isa Terminator
        terminator_stmt = has_terminator ? all_stmts[end] : nothing

        first_stmt_ind = findfirst(stmt -> !isa(stmt, PhiNode), all_stmts)
        last_stmt_ind = has_terminator ? length(all_stmts) - 1 : length(all_stmts)
        phi_inds = 1:(first_stmt_ind - 1)
        stmt_inds = first_stmt_ind:last_stmt_ind

        phi_iterator = zip(all_ids[phi_inds], all_stmts[phi_inds])
        return BBlock(
            id,
            Tuple{ID, IDPhiNode}[(id, stmt) for (id, stmt) in phi_iterator],
            collect(zip(all_ids[stmt_inds], all_stmts[stmt_inds])),
            terminator_stmt,
        )
    end
    return BBCode(ir, blocks)
end

concatenate_ids(bb_code::BBCode) = reduce(vcat, map(concatenate_ids, bb_code.blocks))
concatenate_stmts(bb_code::BBCode) = reduce(vcat, map(concatenate_stmts, bb_code.blocks))

# Maps from positional names (SSAValues for nodes, Integers for basic blocks) to IDs.
const SSAToIdDict = Dict{SSAValue, ID}
const BlockNumToIdDict = Dict{Integer, ID}

function _ssas_to_ids(stmts::Vector{Any})
    ids = map(_ -> ID(), stmts)
    val_id_map = SSAToIdDict(zip(SSAValue.(eachindex(stmts)), ids))
    return ids, convert(Vector{Any}, map(Base.Fix1(_ssa_to_ids, val_id_map), stmts))
end

# Produce a new instance of `x` in which all instances of `SSAValue`s are replaced with
# the `ID`s prescribed by `d`, all basic block numbers are replaced with the `ID`s
# prescribed by `d`, and `GotoIfNot`, `GotoNode`, and `PhiNode` instances are replaced with
# the corresponding `ID` versions.
function _ssa_to_ids(d::SSAToIdDict, x::ReturnNode)
    return isdefined(x, :val) ? ReturnNode(get(d, x.val, x.val)) : x
end
_ssa_to_ids(d::SSAToIdDict, x::Expr) = Expr(x.head, map(a -> get(d, a, a), x.args)...)
_ssa_to_ids(d::SSAToIdDict, x::PiNode) = throw(error("Unhandled node"))
_ssa_to_ids(d::SSAToIdDict, x::QuoteNode) = x
_ssa_to_ids(d::SSAToIdDict, x) = x
function _ssa_to_ids(d::SSAToIdDict, x::PhiNode)
    return PhiNode(x.edges, Any[get(d, a, a) for a in x.values])
end
_ssa_to_ids(d::SSAToIdDict, x::GotoNode) = x
_ssa_to_ids(d::SSAToIdDict, x::GotoIfNot) = GotoIfNot(get(d, x.cond, x.cond), x.dest)

# Replace all integers corresponding to references to blocks with IDs.
function _block_nums_to_ids(stmts::Vector{Any}, cfg::CC.CFG)
    ids = map(_ -> ID(), cfg.blocks)
    block_num_id_map = BlockNumToIdDict(zip(eachindex(cfg.blocks), ids))
    return ids, map(Base.Fix1(_block_num_to_ids, block_num_id_map), stmts)
end

function _block_num_to_ids(d::BlockNumToIdDict, x::PhiNode)
    return IDPhiNode(ID[d[e] for e in x.edges], x.values)
end
_block_num_to_ids(d::BlockNumToIdDict, x::GotoNode) = IDGotoNode(d[x.label])
_block_num_to_ids(d::BlockNumToIdDict, x::GotoIfNot) = IDGotoIfNot(x.cond, d[x.dest])
_block_num_to_ids(d::BlockNumToIdDict, x) = x

#
# Converting from BBCode to IRCode
#

"""
    IRCode(bb_code::BBCode)

Produce an `IRCode` instance which is equivalent to `bb_code`. The resulting `IRCode`
shares no memory with `bb_code`, so can be safely mutated without modifying `bb_code`.

All `IDPhiNode`s, `IDGotoIfNot`s, and `IDGotoNode`s are converted back into `PhiNode`s,
`GotoIfNot`s, and `GotoNode`s respectively.

In the resulting `bb_code`, any `Switch` nodes are lowered into a semantically-equivalent
collection of `GotoIfNot` nodes.
"""
function CC.IRCode(bb_code::BBCode)
    bb_code = _lower_switch_statements(bb_code)
    bb_code = remove_double_edges(bb_code)
    stmts = _ids_to_line_positions(bb_code)
    cfg = CC.compute_basic_blocks(stmts)
    stmts = _lines_to_blocks(stmts, cfg)
    return IRCode(
        CC.InstructionStream( # fill in dummy values for types etc.
            stmts,
            fill(Any, length(stmts)),
            fill(CC.NoCallInfo(), length(stmts)),
            fill(Int32(1), length(stmts)),
            fill(CC.IR_FLAG_REFINED, length(stmts)),
        ),
        cfg,
        CC.copy(bb_code.linetable),
        CC.copy(bb_code.argtypes),
        CC.copy(bb_code.meta),
        CC.copy(bb_code.sptypes),
    )
end

# Converts all `Switch`s into a semantically-equivalent collection of `GotoIfNot`s. See the
# `Switch` docstring for an explanation of what is going on here.
function _lower_switch_statements(bb_code::BBCode)
    new_blocks = Vector{BBlock}(undef, 0)
    for block in bb_code.blocks
        terminator = block.terminator
        if terminator isa Switch

            # Create new block without the `Switch`.
            push!(new_blocks, BBlock(block.id, block.phi_nodes, block.stmts, nothing))

            # Create new blocks for each `GotoIfNot` from the `Switch`.
            foreach(terminator.conds, terminator.dests) do cond, dest
                gotoifnot = IDGotoIfNot(cond, dest)
                blk = BBlock(ID(), Tuple{ID, IDPhiNode}[], Tuple{ID, Any}[], gotoifnot)
                push!(new_blocks, blk)
            end
        else
            push!(new_blocks, block)
        end
    end
    return BBCode(
        new_blocks,
        CC.copy(bb_code.argtypes),
        CC.copy(bb_code.sptypes),
        CC.copy(bb_code.linetable),
        CC.copy(bb_code.meta),
    )
end

# Returns a `Vector{Any}` of statements in which each `ID` has been replaced by either an
# `SSAValue`, or an `Int64` / `Int32` which refers to an `SSAValue`.
function _ids_to_line_positions(bb_code::BBCode)

    # Construct map from `ID`s to `SSAValue`s.
    block_ids = [b.id for b in bb_code.blocks]
    block_lengths = map(length, bb_code.blocks)
    block_start_ssas = SSAValue.(vcat(1, cumsum(block_lengths)[1:end-1] .+ 1))
    line_ids = concatenate_ids(bb_code)
    line_ssas = SSAValue.(eachindex(line_ids))
    id_to_ssa_map = Dict(zip(vcat(block_ids, line_ids), vcat(block_start_ssas, line_ssas)))

    # Apply map.
    return map(Base.Fix1(_to_ssas, id_to_ssa_map), concatenate_stmts(bb_code))
end

# Like `_to_ids`, but converts IDs to SSAValues / (integers corresponding to ssas).
_to_ssas(d::Dict, x::ReturnNode) = isdefined(x, :val) ? ReturnNode(get(d, x.val, x.val)) : x
_to_ssas(d::Dict, x::Expr) = Expr(x.head, map(a -> get(d, a, a), x.args)...)
_to_ssas(d::Dict, x::PiNode) = throw(error("Unhandled node"))
_to_ssas(d::Dict, x::QuoteNode) = x
_to_ssas(d::Dict, x) = x
function _to_ssas(d::Dict, x::IDPhiNode)
    return PhiNode(
        map(e -> Int32(getindex(d, e).id), x.edges),
        Any[get(d, a, a) for a in x.values],
    )
end
_to_ssas(d::Dict, x::IDGotoNode) = GotoNode(d[x.label].id)
_to_ssas(d::Dict, x::IDGotoIfNot) = GotoIfNot(get(d, x.cond, x.cond), d[x.dest].id)

# Replaces references to blocks by line-number with references to block numbers.
function _lines_to_blocks(stmts::Vector{Any}, cfg::CC.CFG)
    return map(stmt -> __lines_to_blocks(cfg, stmt), stmts)
end

function __lines_to_blocks(cfg::CC.CFG, stmt::GotoNode)
    return GotoNode(CC.block_for_inst(cfg, stmt.label))
end
function __lines_to_blocks(cfg::CC.CFG, stmt::GotoIfNot)
    return GotoIfNot(stmt.cond, CC.block_for_inst(cfg, stmt.dest))
end
function __lines_to_blocks(cfg::CC.CFG, stmt::PhiNode)
    return PhiNode(Int32[CC.block_for_inst(cfg, Int(e)) for e in stmt.edges], stmt.values)
end
function __lines_to_blocks(cfg::CC.CFG, stmt::Expr)
    Meta.isexpr(stmt, :enter) && throw(error("Cannot handle enter yet"))
    return stmt
end
__lines_to_blocks(cfg::CC.CFG, stmt) = stmt

#
# Non AD-related Compiler passes
#

"""
    make_unique_return_block(bb_ir::BBCode)

Transforms the code to have a single `ReturnNode` without changing the semantics. Achieves
this by creating a new `BBCode` with an additional block at the end with a phi node and a
return node.
The return nodes in each other block are replaced with goto nodes pointing towards this new
block. The PhiNode evaluates to whichever value the predecessor block would have returned.

For example, the function `foo(x::Int) = x > 0 ? 1 : 0` has IR
```julia
1 1 ─ %1 = Base.slt_int(0, _2)
  └──      goto #3 if not %1
  2 ─      return 1
  3 ─      return 0   
```
This function translates this into something equivalent to
```julia
1 1 ─ %1 = Base.slt_int(0, _2)
  └──      goto #3 if not %1
  2 ─      goto #4
  3 ─      goto #4
  4 ─ %5 = φ (#2 => 1, #3 => 0)
  └──      return %5
```
"""
function make_unique_return_block(ir::BBCode)

    # Declare the ID of the block that we will add at the end of the BBCode.
    return_block_id = ID()

    # For each return node, log its block and what it returns, then replace it with a goto.
    edges = Vector{ID}(undef, 0)
    values = Vector{Any}(undef, 0)
    goto_return_block = IDGotoNode(return_block_id)
    new_blocks = map(ir.blocks) do block
        terminator = block.terminator
        if terminator isa ReturnNode && isdefined(terminator, :val)
            push!(edges, block.id)
            push!(values, terminator.val)
            return BBlock(block.id, block.phi_nodes, block.stmts, goto_return_block)
        else
            return copy(block)
        end
    end

    # Add an additional block to the end of the ir. The PhiNode chooses the value to be
    # returned based on which return node we would have hit in the original ir.
    phi_id = ID()
    ident_id = ID()
    return_block = BBlock(
        return_block_id,
        [(phi_id, IDPhiNode(edges, values))],
        Tuple{ID, Any}[(ident_id, Expr(:call, GlobalRef(Base, :identity), phi_id))],
        ReturnNode(ident_id),
    )
    return BBCode(ir, vcat(new_blocks, return_block))
end

# If the `dest` field of a `GotoIfNot` node points towards the next block, replace it with
# a `GotoNode`.
function remove_double_edges(ir::BBCode)
    new_blks = map(enumerate(ir.blocks)) do (n, blk)
        t = blk.terminator
        if t isa IDGotoIfNot && t.dest == ir.blocks[n+1].id
            return BBlock(blk.id, blk.phi_nodes, blk.stmts, IDGotoNode(t.dest))
        else
            return blk
        end
    end
    return BBCode(ir, new_blks)
end
