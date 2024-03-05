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
    Switch(conds::Vector{Any}, dests::Vector{ID}, fallthrough_dest::ID)

A switch-statement node. This can be placed (temporarily) in Julia IR. Has the following
semantics:
```julia
goto dests[1] if not conds[1]
goto dests[2] if not conds[2]
...
goto dests[N] if not conds[N]
goto fallthrough_dest
```
where the value associated to each element of `conds` is a `Bool`, and `dests` indicate
which block to jump to. If none of the conditions are met, then we go to whichever block is
specified by `fallthrough_dest`.

`Switch` statements are lowered into the above sequence of `GotoIfNot`s and `GotoNode`s
before converting `BBCode` back into `IRCode`.
"""
struct Switch
    conds::Vector{Any}
    dests::Vector{ID}
    fallthrough_dest::ID
    function Switch(conds::Vector{Any}, dests::Vector{ID}, fallthrough_dest::ID)
        @assert length(conds) == length(dests)
        return new(conds, dests, fallthrough_dest)
    end
end

"""
    Terminator = Union{Switch, IDGotoIfNot, IDGotoNode, ReturnNode}

A Union of the possible types of a terminator node.
"""
const Terminator = Union{Switch, IDGotoIfNot, IDGotoNode, ReturnNode}

"""
    BBlock(id::ID, stmts::Vector{<:Tuple{ID, Any}})

A basic block data structure (not called `BasicBlock` to avoid accidental confusion with
`CC.BasicBlock`). Forms a single basic block from a sequence of `stmts`.

Each `BBlock` has an `ID` (a unique name). This makes it possible to refer to blocks in a
way that does not change when additional `BBlocks` are inserted into a `BBCode`.
This differs from the positional block numbering found in `IRCode`, in which the number
associated to a basic block changes when new blocks are inserted.

Note that `PhiNode`s, `GotoIfNot`s, and `GotoNode`s should all be replaced with their
`IDPhiNode`, `IDGotoIfNot`, and `IDGotoNode` equivalents.
"""
mutable struct BBlock
    id::ID
    stmts::Vector{Tuple{ID, Any}}
end

Base.length(bb::BBlock) = length(bb.stmts)

Base.copy(bb::BBlock) = BBlock(bb.id, copy(bb.stmts))

concatenate_ids(bb::BBlock) = first.(bb.stmts)

concatenate_stmts(bb::BBlock) = last.(bb.stmts)

first_id(bb::BBlock) = first(concatenate_ids(bb))

terminator(bb::BBlock) = isa(bb.stmts[end][2], Terminator) ? bb.stmts[end][2] : nothing

"""
    BBCode(
        blocks::Vector{BBlock}
        argtypes::Vector{Any}
        sptypes::Vector{CC.VarState}
        linetable::Vector{Core.LineInfoNode}
        meta::Vector{Expr}
    )

A `BBCode` is a data structure which is similar to `IRCode`, but adds additional structure.

In particular, a `BBCode` comprises a sequence of basic blocks (`BBlock`s), each of which
comprise a sequence of statements. Moreover, each `BBlock` has its own unique `ID`, as does
each statment.

The consequence of this is that new basic blocks can be inserted into a `BBCode`. This is
distinct from `IRCode`, in which to create a new basic block, one must insert additional
statments which you know will create a new basic block -- this is generally quite an
unreliable process, while inserting a new `BBlock` into `BBCode` is entirely predictable.
Furthermore, inserting a new `BBlock` does not change the `ID` associated to the other
blocks, meaning that you can safely assume that references from existing basic block
terminators / phi nodes to other blocks will not be modified by inserting a new basic block.

Additionally, since each statment in each basic block has its own unique `ID`, new
statments can be inserted without changing references between other blocks. `IRCode` also
has some support for this via its `new_nodes` field, but eventually all statements will be
renamed upon `compact!`ing the `IRCode`, meaning that the name of any given statement will
eventually change.

Finally, note that the basic blocks in a `BBCode` support the custom `Switch` statement.
This statement is not valid in `IRCode`, and is therefore lowered into a collection of
`GotoIfNot`s and `GotoNode`s when a `BBCode` is converted back into an `IRCode`.
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

Base.copy(ir::BBCode) = BBCode(ir, copy(ir.blocks))

function predecessors(blk::BBlock, ir::BBCode)
    tmp = map(b -> is_successor(b, blk.id, is_next(b, blk, ir)) ? [b.id] : ID[], ir.blocks)
    return reduce(vcat, tmp)
end

function is_next(block::BBlock, other_block::BBlock, ir::BBCode)
    return location(block, ir) + 1 == location(other_block, ir)
end

location(block::BBlock, ir::BBCode) = findfirst(b -> b.id == block.id, ir.blocks)

is_successor(b::BBlock, id::ID, is_next::Bool) = is_successor(terminator(b), id, is_next)
is_successor(::Nothing, ::ID, is_next::Bool) = is_next
is_successor(x::IDGotoNode, id::ID, ::Bool) = x.label == id
is_successor(x::IDGotoIfNot, id::ID, is_next::Bool) = is_next || x.dest == id
is_successor(::ReturnNode, ::ID, ::Bool) = false
is_successor(x::Switch, id::ID, ::Bool) = any(==(id), x.dests) || id == x.fallthrough_dest

find_block_ind(ir::BBCode, id::ID) = findfirst(b -> b.id == id, ir.blocks)

"""
    collect_stmts(ir::BBCode)::Vector{Tuple{ID, Any}}

Produce a `Vector{Any}` containing all of the statements in `ir`. These are returned in
order, so it is safe to assume that element `n` refers to the `nth` element of the `IRCode`
associated to `ir`. 
"""
function collect_stmts(ir::BBCode)::Vector{Tuple{ID, Any}}
    return reduce(vcat, map(blk -> blk.stmts, ir.blocks))
end

"""
    id_to_line_map(ir::BBCode)

Produces a `Dict` mapping from each `ID` associated with a line in `ir` to its line number.
This is isomorphic to mapping to its `SSAValue` in `IRCode`. Terminators do not have `ID`s
associated to them, so not every line in the original `IRCode` is mapped to.
"""
function id_to_line_map(ir::BBCode)
    lines = collect_stmts(ir)
    lines_and_line_numbers = collect(zip(lines, eachindex(lines)))
    ids_and_line_numbers = map(x -> (x[1][1], x[2]), lines_and_line_numbers)
    return Dict(ids_and_line_numbers)
end

#
# Converting from IRCode to BBCode
#

"""
    BBCode(ir::IRCode)

Convert an `ir` into a `BBCode`. Creates a completely independent data structure, so
mutating the `BBCode` returned will not mutate `ir`.

All `PhiNode`s, `GotoIfNot`s, and `GotoNode`s will be replaced with the `IDPhiNode`s,
`IDGotoIfNot`s, and `IDGotoNode`s respectively.

See `IRCode` for conversion back to `IRCode`.

Note that `IRCode(BBCode(ir))` should be equal to the identity function.
"""
function BBCode(ir::IRCode)

    # Produce a new set of statements with `IDs` rather than `SSAValues` and block numbers.
    ssa_ids, stmts = _ssas_to_ids(ir.stmts.inst)
    block_ids, stmts = _block_nums_to_ids(stmts, ir.cfg)

    # Chop up the new statements into `BBlocks`, according to the `CFG` in `ir`.
    blocks = map(zip(ir.cfg.blocks, block_ids)) do (bb, id)
        return BBlock(id, collect(zip(ssa_ids[bb.stmts], stmts[bb.stmts])))
    end
    return BBCode(ir, blocks)
end

concatenate_ids(bb_code::BBCode) = reduce(vcat, map(concatenate_ids, bb_code.blocks))
concatenate_stmts(bb_code::BBCode) = reduce(vcat, map(concatenate_stmts, bb_code.blocks))

# Maps from positional names (SSAValues for nodes, Integers for basic blocks) to IDs.
const SSAToIdDict = Dict{SSAValue, ID}
const BlockNumToIdDict = Dict{Integer, ID}

# Assigns an ID to each line in `stmts`, and replaces each instance of an `SSAValue` in each
# line with the corresponding `ID`. For example, a call statement of the form
# `Expr(:call, :f, %4)` is be replaced with `Expr(:call, :f, id_assigned_to_%4)`.
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
_ssa_to_ids(d::SSAToIdDict, x::PiNode) = PiNode(get(d, x.val, x.val), get(d, x.typ, x.typ))
_ssa_to_ids(d::SSAToIdDict, x::QuoteNode) = x
_ssa_to_ids(d::SSAToIdDict, x) = x
function _ssa_to_ids(d::SSAToIdDict, x::PhiNode)
    new_values = Vector{Any}(undef, length(x.values))
    for n in eachindex(x.values)
        if isassigned(x.values, n)
            new_values[n] = get(d, x.values[n], x.values[n])
        end
    end
    return PhiNode(x.edges, new_values)
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
        t = terminator(block)
        if t isa Switch

            # Create new block without the `Switch`.
            push!(new_blocks, BBlock(block.id, block.stmts[1:end-1]))

            # Create new blocks for each `GotoIfNot` from the `Switch`.
            foreach(t.conds, t.dests) do cond, dest
                blk = BBlock(ID(), Tuple{ID, Any}[(ID(), IDGotoIfNot(cond, dest))])
                push!(new_blocks, blk)
            end

            # Create a new block for the fallthrough dest.
            blk = BBlock(ID(), Tuple{ID, Any}[(ID(), IDGotoNode(t.fallthrough_dest))])
            push!(new_blocks, blk)
        else
            push!(new_blocks, block)
        end
    end
    return BBCode(bb_code, new_blocks)
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
_to_ssas(d::Dict, x::PiNode) = PiNode(get(d, x.val, x.val), get(d, x.typ, x.typ))
_to_ssas(d::Dict, x::QuoteNode) = x
_to_ssas(d::Dict, x) = x
function _to_ssas(d::Dict, x::IDPhiNode)
    new_values = Vector{Any}(undef, length(x.values))
    for n in eachindex(x.values)
        if isassigned(x.values, n)
            new_values[n] = get(d, x.values[n], x.values[n])
        end
    end
    return PhiNode(map(e -> Int32(getindex(d, e).id), x.edges), new_values)
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

# If the `dest` field of a `GotoIfNot` node points towards the next block, replace it with
# a `GotoNode`.
function remove_double_edges(ir::BBCode)
    new_blks = map(enumerate(ir.blocks)) do (n, blk)
        t = terminator(blk)
        if t isa IDGotoIfNot && t.dest == ir.blocks[n+1].id
            t_id = blk.stmts[end][1]
            return BBlock(blk.id, vcat(blk.stmts[1:end-1], (t_id, IDGotoNode(t.dest))))
        else
            return blk
        end
    end
    return BBCode(ir, new_blks)
end

#=
    _sort_blocks!(ir::BBCode)

Ensure that blocks appear in order of distance-from-entry-point, where distance the
distance from block b to the entry point is defined to be the minimum number of basic
blocks that must be passed through in order to reach b.

For reasons unknown (to me, Will), the compiler / optimiser needs this for inference to
succeed. Since we do quite a lot of re-ordering on the reverse-pass of AD, this is a problem
there.

WARNING: use with care. Only use if you are confident that arbitrary re-ordering of basic
blocks in `ir` is valid. Notably, this does not hold if you have any `IDGotoIfNot` nodes in
`ir`.
=#
function _sort_blocks!(ir::BBCode)

    node_ints = collect(eachindex(ir.blocks))
    id_to_int = Dict(zip(map(blk -> blk.id, ir.blocks), node_ints))

    direct_predecessors = map(ir.blocks) do blk
        return map(b -> Edge(id_to_int[b], id_to_int[blk.id]), predecessors(blk, ir))
    end
    g = SimpleDiGraph(reduce(vcat, direct_predecessors))

    d = dijkstra_shortest_paths(g, id_to_int[ir.blocks[1].id]).dists
    I = sortperm(d)
    ir.blocks .= ir.blocks[I]
    return ir
end