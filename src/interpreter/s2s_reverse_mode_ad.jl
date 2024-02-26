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
    if !(line in keys(m.m))
        m.m[line] = maximum(keys(m.m)) + 1
    end
    return m.m[line]
end

#=
    ADInfo(terminator_block_id::ID, line_map::LineToADDataMap)

This data structure is used to hold global information which gets passed around, in
particular to `make_ad_stmts!`.

- `terminator_block_id`: the ID of the block inserted to provide a unique exit point on the
    forwards-pass
- `line_map`: a `LineToADDataMap`.
=#
struct ADInfo
    terminator_block_id::ID
    line_map::LineToADDataMap
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
    if Meta.isexpr(stmt, :call)

    elseif Meta.isexpr(stmt, :invoke)

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

@inline function __throw_undef_if_not(slotname::Symbol, cond_codual::CoDual)
    primal(cond_codual) || throw(UndefVarError(slotname))
    return nothing
end

"""
    build_rrule(::Type{C}, sig::Type{<:Tuple}) where {C}

Returns and `OpaqueClosure` which is an `rrule!!` for `sig` in context `C`.
"""
function build_rrule(::Type{C}, sig::Type{<:Tuple}) where {C}

    # Grab code associated to the primal.
    interp = TapedInterpreter(C)
    ir, Treturn = lookup_ir(interp, sig)

    # Normalise the IR.
    is_vararg, spnames = is_vararg_sig_and_sparam_names(sig)
    ir = normalise!(ir, spnames)


end

