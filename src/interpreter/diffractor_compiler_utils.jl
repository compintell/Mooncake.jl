#! format: off

# Utilities that should probably go into CC
using Core.Compiler: IRCode, CFG, BasicBlock

function Base.push!(cfg::CFG, bb::BasicBlock)
    @assert cfg.blocks[end].stmts.stop+1 == bb.stmts.start
    push!(cfg.blocks, bb)
    push!(cfg.index, bb.stmts.start)
end

if VERSION < v"1.11.0-DEV.258"
    Base.getindex(ir::IRCode, ssa::SSAValue) = CC.getindex(ir, ssa)
end

if VERSION < v"1.12.0-DEV.1268"
    if isdefined(CC, :Future)
        Base.isready(future::CC.Future) = CC.isready(future)
        Base.getindex(future::CC.Future) = CC.getindex(future)
        Base.setindex!(future::CC.Future, value) = CC.setindex!(future, value)
    end

    Base.iterate(c::CC.IncrementalCompact, args...) = CC.iterate(c, args...)
    Base.iterate(p::CC.Pair, args...) = CC.iterate(p, args...)
    Base.iterate(urs::CC.UseRefIterator, args...) = CC.iterate(urs, args...)
    Base.iterate(x::CC.BBIdxIter, args...) = CC.iterate(x, args...)
    Base.getindex(urs::CC.UseRefIterator, args...) = CC.getindex(urs, args...)
    Base.getindex(urs::CC.UseRef, args...) = CC.getindex(urs, args...)
    Base.getindex(c::CC.IncrementalCompact, args...) = CC.getindex(c, args...)
    Base.setindex!(c::CC.IncrementalCompact, args...) = CC.setindex!(c, args...)
    Base.setindex!(urs::CC.UseRef, args...) = CC.setindex!(urs, args...)

    Base.copy(ir::IRCode) = CC.copy(ir)

    CC.BasicBlock(x::UnitRange) =
        BasicBlock(StmtRange(first(x), last(x)))
    CC.BasicBlock(x::UnitRange, preds::Vector{Int}, succs::Vector{Int}) =
        BasicBlock(StmtRange(first(x), last(x)), preds, succs)
    Base.length(c::CC.NewNodeStream) = CC.length(c)
    Base.setindex!(i::CC.Instruction, args...) = CC.setindex!(i, args...)
    Base.size(x::CC.UnitRange) = CC.size(x)

    CC.get(a::Dict, b, c) = Base.get(a,b,c)
    CC.haskey(a::Dict, b) = Base.haskey(a, b)
    CC.setindex!(a::Dict, b, c) = setindex!(a, b, c)
end

CC.NewInstruction(@nospecialize node) =
    NewInstruction(node, Any, CC.NoCallInfo(), nothing, CC.IR_FLAG_REFINED)

Base.setproperty!(x::CC.Instruction, f::Symbol, v) = CC.setindex!(x, v, f)

Base.getproperty(x::CC.Instruction, f::Symbol) = CC.getindex(x, f)

function Base.setindex!(ir::IRCode, ni::NewInstruction, i::Int)
    stmt = ir.stmts[i]
    stmt.inst = ni.stmt
    stmt.type = ni.type
    stmt.flag = something(ni.flag, 0)  # fixes 1.9?
    @static if VERSION ≥ v"1.12.0-DEV.173"
        stmt.line = something(ni.line, CC.NoLineUpdate)
    else
        stmt.line = something(ni.line, 0)
    end
    return ni
end

Base.lastindex(x::CC.InstructionStream) = CC.length(x)

function replace_call!(ir, idx::SSAValue, new_call)
    ir[idx][:inst] = new_call
    ir[idx][:type] = Any
    ir[idx][:info] = CC.NoCallInfo()
    ir[idx][:flag] = CC.IR_FLAG_REFINED
end

#! format: on
