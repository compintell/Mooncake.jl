#=
    SharedDataPairs()

A data structure used to manage the captured data in the `OpaqueClosures` which implement
the bulk of the forwards- and reverse-passes of AD. An entry `(id, data)` at element `n`
of the `pairs` field of this data structure means that `data` will be available at register
`id` during the forwards- and reverse-passes of `AD`.

This is achieved by storing all of the data in the `pairs` field in the captured tuple which
is passed to an `OpaqueClosure`, and extracting this data into registers associated to the
corresponding `ID`s.
=#
struct SharedDataPairs
    pairs::Vector{Tuple{ID, Any}}
    SharedDataPairs() = new(Tuple{ID, Any}[])
end

#=
    add_data!(p::SharedDataPairs, data)::ID

Puts `data` into `p`, and returns the `id` associated to it. This `id` should be assumed to
be available during the forwards- and reverse-passes of AD, and it should further be assumed
that the value associated to this `id` is always `data`.
=#
function add_data!(p::SharedDataPairs, data)::ID
    id = ID()
    push!(p.pairs, (id, data))
    return id
end

#=
    shared_data_tuple(p::SharedDataPairs)::Tuple

Create the tuple that will constitute the captured variables in the forwards- and reverse-
pass `OpaqueClosure`s.

For example, if `p.pairs` is
```julia
[(ID(5), 5.0), (ID(3), "hello")]
```
then the output of this function is
```julia
(5.0, "hello")
```
=#
shared_data_tuple(p::SharedDataPairs)::Tuple = tuple(map(last, p.pairs)...)

#=
    shared_data_stmts(p::SharedDataPairs)::Vector{IDInstPair}

Produce a sequence of id-statment pairs which will extract the data from
`shared_data_tuple(p)` such that the correct value is associated to the correct `ID`.

For example, if `p.pairs` is
```julia
[(ID(5), 5.0), (ID(3), "hello")]
```
then the output of this function is
```julia
IDInstPair[
    (ID(5), new_inst(:(getfield(_1, 1)))),
    (ID(3), new_inst(:(getfield(_1, 2)))),
]
```
=#
function shared_data_stmts(p::SharedDataPairs)::Vector{IDInstPair}
    return map(enumerate(p.pairs)) do (n, p)
        return (p[1], new_inst(Expr(:call, getfield, Argument(1), n)))
    end
end

#=
The block stack is the stack used to keep track of which basic blocks are visited on the
forwards pass, and therefore which blocks need to be visited on the reverse pass. There is
one block stack per derived rule.
By using Int32, we assume that there aren't more than `typemax(Int32)` unique basic blocks
in a given function, which ought to be reasonable.
=#
const BlockStack = Stack{Int32}

#=
    ADInfo

This data structure is used to hold "global" information associated to a particular call to
`build_rrule`. It is used as a means of communication between `make_ad_stmts!` and the
codegen which produces the forwards- and reverse-passes.

- `interp`: a `MooncakeInterpreter`.
- `block_stack_id`: the ID associated to the block stack -- the stack which keeps track of
    which blocks we visited during the forwards-pass, and which is used on the reverse-pass
    to determine which blocks to visit.
- `block_stack`: the block stack. Can always be found at `block_stack_id` in the forwards-
    and reverse-passes.
- `entry_id`: ID associated to the block inserted at the start of execution in the the
    forwards-pass, and the end of execution in the pullback.
- `shared_data_pairs`: the `SharedDataPairs` used to define the captured variables passed
    to both the forwards- and reverse-passes.
- `arg_types`: a map from `Argument` to its static type.
- `ssa_insts`: a map from `ID` associated to lines to the primal `NewInstruction`. This
    contains the line of code, its static / inferred type, and some other detailss. See
    `Core.Compiler.NewInstruction` for a full list of fields.
- `arg_rdata_ref_ids`: the dict mapping from arguments to the `ID` which creates and
    initialises the `Ref` which contains the reverse data associated to that argument.
    Recall that the heap allocations associated to this `Ref` are always optimised away in
    the final programme.
- `ssa_rdata_ref_ids`: the same as `arg_rdata_ref_ids`, but for each `ID` associated to an
    ssa rather than each argument.
- `debug_mode`: if `true`, run in "debug mode" -- wraps all rule calls in `DebugRRule`. This
    is applied recursively, so that debug mode is also switched on in derived rules.
- `is_used_dict`: for each `ID` associated to a line of code, is `false` if line is not used
    anywhere in any other line of code.
- `lazy_zero_rdata_ref_id`: for any arguments whose type doesn't permit the construction of
    a zero-valued rdata directly from the type alone (e.g. a struct with an abstractly-
    typed field), we need to have a zero-valued rdata available on the reverse-pass so that
    this zero-valued rdata can be returned if the argument (or a part of it) is never used
    during the forwards-pass and consequently doesn't obtain a value on the reverse-pass.
    To achieve this, we construct a `LazyZeroRData` for each of the arguments on the
    forwards-pass, and make use of it on the reverse-pass. This field is the ID that will be
    associated to this information.
=#
struct ADInfo
    interp::MooncakeInterpreter
    block_stack_id::ID
    block_stack::BlockStack
    entry_id::ID
    shared_data_pairs::SharedDataPairs
    arg_types::Dict{Argument, Any}
    ssa_insts::Dict{ID, NewInstruction}
    arg_rdata_ref_ids::Dict{Argument, ID}
    ssa_rdata_ref_ids::Dict{ID, ID}
    debug_mode::Bool
    is_used_dict::Dict{ID, Bool}
    lazy_zero_rdata_ref_id::ID
end

# The constructor that you should use for ADInfo if you don't have a BBCode lying around.
# See the definition of the ADInfo struct for info on the arguments.
function ADInfo(
    interp::MooncakeInterpreter,
    arg_types::Dict{Argument, Any},
    ssa_insts::Dict{ID, NewInstruction},
    is_used_dict::Dict{ID, Bool},
    debug_mode::Bool,
    zero_lazy_rdata_ref::Ref{<:Tuple},
)
    shared_data_pairs = SharedDataPairs()
    block_stack = BlockStack()
    return ADInfo(
        interp,
        add_data!(shared_data_pairs, block_stack),
        block_stack,
        ID(),
        shared_data_pairs,
        arg_types,
        ssa_insts,
        Dict((k, ID()) for k in keys(arg_types)),
        Dict((k, ID()) for k in keys(ssa_insts)),
        debug_mode,
        is_used_dict,
        add_data!(shared_data_pairs, zero_lazy_rdata_ref),
    )
end

# The constructor you should use for ADInfo if you _do_ have a BBCode lying around. See the
# ADInfo struct for information regarding `interp` and `debug_mode`.
function ADInfo(interp::MooncakeInterpreter, ir::BBCode, debug_mode::Bool)
    arg_types = Dict{Argument, Any}(
        map(((n, t),) -> (Argument(n) => _type(t)), enumerate(ir.argtypes))
    )
    stmts = collect_stmts(ir)
    ssa_insts = Dict{ID, NewInstruction}(stmts)
    is_used_dict = characterise_used_ids(stmts)
    zero_lazy_rdata_ref = Ref{Tuple{map(lazy_zero_rdata_type ∘ _type, ir.argtypes)...}}()
    return ADInfo(interp, arg_types, ssa_insts, is_used_dict, debug_mode, zero_lazy_rdata_ref)
end

# Shortcut for `add_data!(info.shared_data_pairs, data)`.
add_data!(info::ADInfo, data)::ID = add_data!(info.shared_data_pairs, data)

# Returns `x` if it is a singleton, or the `ID` of the ssa which will contain it on the
# forwards- and reverse-passes. The reason for this is that if something is a singleton, it
# can be placed directly in the IR.
function add_data_if_not_singleton!(p::Union{ADInfo, SharedDataPairs}, x)
    return Base.issingletontype(_typeof(x)) ? x : add_data!(p, x)
end

# Returns `true` if `id` is used by any of the lines in the ir, false otherwise.
is_used(info::ADInfo, id::ID)::Bool = info.is_used_dict[id]

# Returns the static / inferred type associated to `x`.
get_primal_type(info::ADInfo, x::Argument) = info.arg_types[x]
get_primal_type(info::ADInfo, x::ID) = _type(info.ssa_insts[x].type)
get_primal_type(::ADInfo, x::QuoteNode) = _typeof(x.value)
get_primal_type(::ADInfo, x) = _typeof(x)
function get_primal_type(::ADInfo, x::GlobalRef)
    return isconst(x) ? _typeof(getglobal(x.mod, x.name)) : x.binding.ty
end

# Returns the `ID` associated to the line in the reverse pass which will contain the
# reverse data for `x`. If `x` is not an `Argument` or `ID`, then `nothing` is returned.
get_rev_data_id(info::ADInfo, x::Argument) = info.arg_rdata_ref_ids[x]
get_rev_data_id(info::ADInfo, x::ID) = info.ssa_rdata_ref_ids[x]
get_rev_data_id(::ADInfo, ::Any) = nothing

# Create the statements which initialise the reverse-data `Ref`s.
function reverse_data_ref_stmts(info::ADInfo)
    arg_stmts = [(id, __ref(_type(info.arg_types[k]))) for (k, id) in info.arg_rdata_ref_ids]
    ssa_stmts = [(id, __ref(_type(info.ssa_insts[k].type))) for (k, id) in info.ssa_rdata_ref_ids]
    return vcat(arg_stmts, ssa_stmts)
end

# Helper for reverse_data_ref_stmts.
__ref(P) = new_inst(Expr(:call, __make_ref, P))

# Helper for reverse_data_ref_stmts.
@inline @generated function __make_ref(p::Type{P}) where {P}
    _P = @isdefined(P) ? P : _typeof(p)
    R = zero_like_rdata_type(_P)
    return :(Ref{$R}(Mooncake.zero_like_rdata_from_type($_P)))
end

# This specialised method is necessary to ensure that `__make_ref` works properly for
# `DataType`s with unbound type parameters. See `TestResources.typevar_tester` for an
# example. The above method requires that `P` be a type in which all parameters are fully-
# bound. Strange errors occur if this property does not hold.
@inline __make_ref(::Type{<:Type}) = Ref{NoRData}(NoRData())

@inline __make_ref(::Type{Union{}}) = nothing

# Returns the number of arguments that the primal function has.
num_args(info::ADInfo) = length(info.arg_types)

# This struct is used to ensure that `ZeroRData`s, which are used as placeholder zero
# elements whenever an actual instance of a zero rdata for a particular primal type cannot
# be constructed without also having an instance of said type, never reach rules.
# On the pullback, we increment the cotangent dy by an amount equal to zero. This ensures
# that if it is a `ZeroRData`, we instead get an actual zero of the correct type. If it is
# not a zero rdata, the computation _should_ be elided via inlining + constant prop.
struct RRuleZeroWrapper{Trule}
    rule::Trule
end

_copy(x::P) where {P<:RRuleZeroWrapper} = P(_copy(x.rule))

struct RRuleWrapperPb{Tpb!!, Tl}
    pb!!::Tpb!!
    l::Tl
end

(rule::RRuleWrapperPb)(dy) = rule.pb!!(increment!!(dy, instantiate(rule.l)))

@inline function (rule::RRuleZeroWrapper{R})(f::F, args::Vararg{CoDual, N}) where {R, F, N}
    y, pb!! = rule.rule(f, args...)
    l = lazy_zero_rdata(primal(y))
    return y::CoDual, (pb!! isa NoPullback ? pb!! : RRuleWrapperPb(pb!!, l))
end

#=
    ADStmtInfo

Data structure which contains the result of `make_ad_stmts!`. Fields are
- `line`: the ID associated to the primal line from which this is derived
- `comms_id`: an `ID` from one of the lines in `fwds`, whose value will be made
    available on the reverse-pass in the same `ID`. Nothing is asserted about _how_ this
    value is made available on the reverse-pass of AD, so this package is free to do this in
    whichever way is most efficient, in particular to group these communication `ID` on a
    per-block basis.
- `fwds`: the instructions which run the forwards-pass of AD
- `rvs`: the instructions which run the reverse-pass of AD / the pullback
=#
struct ADStmtInfo
    line::ID
    comms_id::Union{ID, Nothing}
    fwds::Vector{IDInstPair}
    rvs::Vector{IDInstPair}
end

# Convenient constructor for `ADStmtInfo`. If either `fwds` or `rvs` is not a vector,
# `__vec` promotes it to a single-element `Vector`.
function ad_stmt_info(line::ID, comms_id::Union{ID, Nothing}, fwds, rvs)
    if !(comms_id === nothing || in(comms_id, map(first, __vec(line, fwds))))
        throw(ArgumentError("comms_id not found in IDs of `fwds` instructions."))
    end
    return ADStmtInfo(line, comms_id, __vec(line, fwds), __vec(line, rvs))
end

__vec(line::ID, x::Any) = __vec(line, new_inst(x))
__vec(line::ID, x::NewInstruction) = IDInstPair[(line, x)]
__vec(line::ID, x::Vector{Tuple{ID, Any}}) = throw(error("boooo"))
__vec(line::ID, x::Vector{IDInstPair}) = x

# Return the element of `fwds` whose `ID` is the communcation `ID`. Returns `Nothing` if
# `comms_id` is `nothing`.
function comms_channel(info::ADStmtInfo)
    info.comms_id === nothing && return nothing
    return only(filter(x -> x[1] == info.comms_id, info.fwds))
end

#=
    make_ad_stmts!(inst::NewInstruction, line::ID, info::ADInfo)::ADStmtInfo

Every line in the primal code is associated to one or more lines in the forwards-pass of AD,
and one or more lines in the pullback. This function has method specific to every
node type in the Julia SSAIR.

Translates the instruction `inst`, associated to `line` in the primal, into a specification
of what should happen for this instruction in the forwards- and reverse-passes of AD, and
what data should be shared between the forwards- and reverse-passes. Returns this in the
form of an `ADStmtInfo`.

`info` is a data structure containing various bits of global information that certain types
of nodes need access to.
=#
function make_ad_stmts! end

# `nothing` as a statement in Julia IR indicates the presence of a line which will later be
# removed. We emit a no-op on both the forwards- and reverse-passes. No shared data.
function make_ad_stmts!(::Nothing, line::ID, ::ADInfo)
    return ad_stmt_info(line, nothing, nothing, nothing)
end

# `ReturnNode`s have a single field, `val`, for which there are three cases to consider:
#
# 1. `val` is undefined: this `ReturnNode` is unreachable. Consequently, we'll never hit the
#   associated statements on the forwards-pass or pullback. We just return the original
#   statement on the forwards-pass, and `nothing` on the reverse-pass.
# 2. `val isa Union{Argument, ID}`: this is an active piece of data. Consequently, we know
#   that it will be an `CoDual` already, and can just return it. Therefore `stmt`
#   is returned as the forwards-pass (with any `Argument`s incremented). On the reverse-pass
#   the associated rdata ref should be incremented with the rdata passed to the pullback,
#   which lives in argument 2.
# 3. `val` is defined, but not a `Union{Argument, ID}`: in this case we're returning a
#   constant -- build a constant CoDual and return that. There is nothing to do on the
#   reverse pass.
function make_ad_stmts!(stmt::ReturnNode, line::ID, info::ADInfo)
    if !is_reachable_return_node(stmt)
        return ad_stmt_info(line, nothing, inc_args(stmt), nothing)
    end
    if is_active(stmt.val)
        rdata_id = get_rev_data_id(info, stmt.val)
        rvs = new_inst(Expr(:call, increment_ref!, rdata_id, Argument(2)))
        return ad_stmt_info(line, nothing, inc_args(stmt), rvs)
    else
        fwds = ReturnNode(const_codual(stmt.val, info))
        return ad_stmt_info(line, nothing, fwds, nothing)
    end
end

# Identity forwards-pass, no-op reverse. No shared data.
function make_ad_stmts!(stmt::IDGotoNode, line::ID, ::ADInfo)
    return ad_stmt_info(line, nothing, inc_args(stmt), nothing)
end

# Identity forwards-pass, no-op reverse. No shared data.
function make_ad_stmts!(stmt::IDGotoIfNot, line::ID, ::ADInfo)
    stmt = inc_args(stmt)

    # If cond is not going to be wrapped in a `CoDual`, so just return the stmt.
    is_active(stmt.cond) || return ad_stmt_info(line, nothing, stmt, nothing)

    # stmt.cond is active, so primal must be extracted from `CoDual`.
    cond_id = ID()
    fwds = [
        (cond_id, new_inst(Expr(:call, primal, stmt.cond))),
        (line, new_inst(IDGotoIfNot(cond_id, stmt.dest), Any)),
    ]
    return ad_stmt_info(line, nothing, fwds, nothing)
end

# Identity forwards-pass, no-op reverse. No shared data.
function make_ad_stmts!(stmt::IDPhiNode, line::ID, info::ADInfo)
    vals = stmt.values
    new_vals = Vector{Any}(undef, length(vals))
    for n in eachindex(vals)
        isassigned(vals, n) || continue
        new_vals[n] = is_active(vals[n]) ? __inc(vals[n]) : const_codual(vals[n], info)
    end

    # It turns out to be really very important to do type inference correctly for PhiNodes.
    # For some reason, type inference really doesn't like it when you encounter mutually-
    # dependent PhiNodes whose types are unknown and for which you set the flag to
    # CC.IR_FLAG_REFINED. To avoid this we directly tell the compiler what the type is.
    new_type = fcodual_type(get_primal_type(info, line))
    _inst = new_inst(IDPhiNode(stmt.edges, new_vals), new_type, info.ssa_insts[line].flag)
    return ad_stmt_info(line, nothing, _inst, nothing)
end

function make_ad_stmts!(stmt::PiNode, line::ID, info::ADInfo)

    # PiNodes of the form `π (nothing, Union{})` have started appearing in 1.11. These nodes
    # appear in unreachable sections of code, and appear to serve no purpose. Consequently,
    # we mark them for removal (replace them with `nothing`). We do not currently have a
    # unit test for this, but integration testing seems to catch it in multiple places.
    stmt == PiNode(nothing, Union{}) && return ad_stmt_info(line, nothing, stmt, nothing)

    if is_active(stmt.val)
        # Get the primal type of this line, and the rdata refs for the `val` of this
        # `PiNode` and this line itself.
        P = get_primal_type(info, line)
        val_rdata_ref_id = get_rev_data_id(info, stmt.val)
        output_rdata_ref_id = get_rev_data_id(info, line)
        fwds = PiNode(__inc(stmt.val), fcodual_type(_type(stmt.typ)))
        rvs = Expr(:call, __pi_rvs!, P, val_rdata_ref_id, output_rdata_ref_id)
    else
        # If the value of the PiNode is a constant / QuoteNode etc, then there is nothing to
        # do on the reverse-pass.
        fwds = PiNode(const_codual(stmt.val, info), fcodual_type(_type(stmt.typ)))
        rvs = nothing
    end

    return ad_stmt_info(line, nothing, fwds, rvs)
end

@inline function __pi_rvs!(::Type{P}, val_rdata_ref::Ref, output_rdata_ref::Ref) where {P}
    increment_ref!(val_rdata_ref, __deref_and_zero(P, output_rdata_ref))
    return nothing
end

# Constant GlobalRefs are handled. See const_codual. Non-constant GlobalRefs are handled by
# assuming that they are constant, and creating a CoDual with the value. We then check at
# run-time that the value has not changed.
function make_ad_stmts!(stmt::GlobalRef, line::ID, info::ADInfo)
    isconst(stmt) && return const_ad_stmt(stmt, line, info)

    x = const_codual(getglobal(stmt.mod, stmt.name), info)
    globalref_id = ID()
    fwds = [
        (globalref_id, new_inst(stmt)),
        (line, new_inst(Expr(:call, __verify_const, globalref_id, x))),
    ]
    return ad_stmt_info(line, nothing, fwds, nothing)
end

# Helper used by `make_ad_stmts! ` for `GlobalRef`. Noinline to avoid IR bloat.
@noinline function __verify_const(global_ref, stored_value)
    @assert global_ref == primal(stored_value)
    return uninit_fcodual(global_ref)
end

# QuoteNodes are constant.
make_ad_stmts!(stmt::QuoteNode, line::ID, info::ADInfo) = const_ad_stmt(stmt, line, info)

# Literal constant.
make_ad_stmts!(stmt, line::ID, info::ADInfo) = const_ad_stmt(stmt, line, info)

# `make_ad_stmts!` for constants.
function const_ad_stmt(stmt, line::ID, info::ADInfo)
    x = const_codual(stmt, info)
    return ad_stmt_info(line, nothing, x isa ID ? Expr(:call, identity, x) : x, nothing)
end

# Build a `CoDual` from `stmt`, with zero / uninitialised fdata. If the resulting CoDual is
# a bits type, then it is returned. If it is not, then the CoDual is put into shared data,
# and the ID associated to it in the forwards- and reverse-passes returned.
function const_codual(stmt, info::ADInfo)
    x = uninit_fcodual(get_const_primal_value(stmt))
    return isbitstype(_typeof(x)) ? x : add_data!(info, x)
end

# Get the value associated to `x`. For `GlobalRef`s, verify that `x` is indeed a constant.
function get_const_primal_value(x::GlobalRef)
    isconst(x) || unhandled_feature("Non-constant GlobalRef not supported: $x")
    return getglobal(x.mod, x.name)
end
get_const_primal_value(x::QuoteNode) = x.value
get_const_primal_value(x) = x

# Mooncake does not yet handle `PhiCNode`s. Throw an error if one is encountered.
function make_ad_stmts!(stmt::Core.PhiCNode, ::ID, ::ADInfo)
    unhandled_feature("Encountered PhiCNode: $stmt")
end

# Mooncake does not yet handle `UpsilonNode`s. Throw an error if one is encountered.
function make_ad_stmts!(stmt::Core.UpsilonNode, ::ID, ::ADInfo)
    unhandled_feature("Encountered UpsilonNode: $stmt")
end

# There are quite a number of possible `Expr`s that can be encountered. Each case has its
# own comment, explaining what is going on.
function make_ad_stmts!(stmt::Expr, line::ID, info::ADInfo)
    is_invoke = Meta.isexpr(stmt, :invoke)
    if Meta.isexpr(stmt, :call) || is_invoke

        # Find the types of all arguments to this call / invoke.
        args = ((is_invoke ? stmt.args[2:end] : stmt.args)..., )
        arg_types = map(arg -> get_primal_type(info, arg), args)

        # Special case: if the result of a call to getfield is un-used, then leave the
        # primal statment alone (just increment arguments as usual). This was causing
        # performance problems in a couple of situations where the field being requested is
        # not known at compile time. `getfield` cannot be dead-code eliminated, because it
        # can throw an error if the requested field does not exist. Everything _other_ than
        # the boundscheck is eliminated in LLVM codegen, so it's important that AD doesn't
        # get in the way of this.
        #
        # This might need to be generalised to more things than just `getfield`, but at the
        # time of writing this comment, it's unclear whether or not this is the case.
        if !is_used(info, line) && get_const_primal_value(args[1]) == getfield
            fwds = new_inst(Expr(:call, __fwds_pass_no_ad!, map(__inc, args)...))
            return ad_stmt_info(line, nothing, fwds, nothing)
        end

        # Construct signature, and determine how the rrule is to be computed.
        sig = Tuple{arg_types...}
        raw_rule = if is_primitive(context_type(info.interp), sig)
            rrule!! # intrinsic / builtin / thing we provably have rule for
        elseif is_invoke
            mi = stmt.args[1]::Core.MethodInstance
            LazyDerivedRule(mi, info.debug_mode) # Static dispatch
        else
            DynamicDerivedRule(info.debug_mode)  # Dynamic dispatch
        end

        # Wrap the raw rule in a struct which ensures that any `ZeroRData`s are stripped
        # away before the raw_rule is called. Only do this if we cannot prove that the
        # output of `can_produce_zero_rdata_from_type(P)`, where `P` is the type of the
        # value returned by this line.
        tmp = can_produce_zero_rdata_from_type(get_primal_type(info, line))
        zero_wrapped_rule = tmp ? raw_rule : RRuleZeroWrapper(raw_rule)

        # If debug mode has been requested, use a debug rule.
        rule = info.debug_mode ? DebugRRule(zero_wrapped_rule) : zero_wrapped_rule

        # If the rule is `rrule!!` (i.e. `sig` is primitive), then don't bother putting
        # the rule into shared data, because it's safe to put it directly into the code.
        rule_ref = add_data_if_not_singleton!(info, rule)

        # If the type of the pullback is a singleton type, then there is no need to store it
        # in the shared data, it can be interpolated directly into the generated IR.
        T_pb!! = pullback_type(_typeof(rule), arg_types)

        #
        # Write forwards-pass. These statements are written out manually, as writing them
        # out in a function would prevent inlining in some (all?) type-unstable situations.
        #

        # Make arguments to rrule call. Things which are not already CoDual must be made so.
        codual_arg_ids = map(_ -> ID(), args)
        __codual_args = map(arg -> Expr(:call, __make_codual, __inc(arg)), args)
        codual_args = IDInstPair[
            (id, new_inst(arg)) for (id, arg) in zip(codual_arg_ids, __codual_args)
        ]

        # Make call to rule.
        rule_call_id = ID()
        rule_call = Expr(:call, rule_ref, codual_arg_ids...)

        # Extract the output-codual from the returned tuple.
        raw_output_id = ID()
        raw_output = Expr(:call, getfield, rule_call_id, 1)

        # Extract the pullback from the returned tuple. Specialise on the case that the
        # pullback is provably a singleton type.
        if Base.issingletontype(T_pb!!)
            pb = T_pb!!.instance
            pb_stmt = (ID(), new_inst(nothing))
            comms_id = nothing
        else
            pb = ID()
            pb_stmt = (pb, new_inst(Expr(:call, getfield, rule_call_id, 2), T_pb!!))
            comms_id = pb
        end

        # Provide a type assertion to help the compiler out. Doing it this way, rather than
        # directly changing the inferred type of the instruction associated to raw_output,
        # has the advantage of not introducing the possibility of segfaults. It will still
        # be optimised away in situations where the compiler is able to successfully infer
        # the type, so performance in performance-critical situations is unaffected.
        output_id = line
        F = fcodual_type(get_primal_type(info, line))
        output = Expr(:call, Core.typeassert, raw_output_id, F)

        # Create statements associated to forwards-pass.
        fwds = vcat(
            codual_args,
            IDInstPair[
                (rule_call_id, new_inst(rule_call)),
                (raw_output_id, new_inst(raw_output)),
                pb_stmt,
                (output_id, new_inst(output)),
            ],
        )

        # Make statement associated to reverse-pass. If the reverse-pass is provably a
        # NoPullback, then don't bother doing anything at all.
        rvs_pass = if T_pb!! <: NoPullback
            nothing
        else
            Expr(
                :call,
                __run_rvs_pass!,
                get_primal_type(info, line),
                sig,
                pb,
                get_rev_data_id(info, line),
                map(Base.Fix1(get_rev_data_id, info), args)...,
            )
        end
        return ad_stmt_info(line, comms_id, fwds, new_inst(rvs_pass))

    elseif Meta.isexpr(stmt, :boundscheck)
        # For some reason the compiler cannot handle boundscheck statements when we run it
        # again. Consequently, emit `true` to be safe. Ideally we would handle this in a
        # more natural way, but I'm not sure how to do that.
        return ad_stmt_info(line, nothing, zero_fcodual(true), nothing)

    elseif Meta.isexpr(stmt, :code_coverage_effect)
        # Code coverage irrelevant for derived code, and really inflates it in some
        # situations. Since code coverage is usually only requrested during CI, including
        # these effects also creates differences between the code generated when developing
        # and the code generated in CI, which occassionally creates hard-to-debug issues.
        return ad_stmt_info(line, nothing, nothing, nothing)

    elseif Meta.isexpr(stmt, :copyast)
        # Get constant out and shove it in shared storage.
        x = const_codual(stmt.args[1], info)
        return ad_stmt_info(line, nothing, Expr(:call, identity, x), nothing)

    elseif Meta.isexpr(stmt, :loopinfo)
        # Cannot pass loopinfo back through the optimiser for some reason.
        # At the time of writing, I am unclear why this is not possible.
        return ad_stmt_info(line, nothing, nothing, nothing)

    elseif stmt.head in [
        :enter,
        :gc_preserve_begin,
        :gc_preserve_end,
        :leave,
        :pop_exception,
        :throw_undef_if_not,
        :meta,
    ]
        # Expressions which do not require any special treatment.
        return ad_stmt_info(line, nothing, stmt, nothing)

    elseif stmt.head == :(=) && stmt.args[1] isa GlobalRef
        msg = "Encountered assignment to global variable: $(stmt.args[1]). " *
            "Cannot differentiate through assignments to globals. " *
            "Please refactor your code to avoid assigning to a global, for example by " *
            "passing the variable in to the function as an argument."
        unhandled_feature(msg)
    else
        # Encountered an expression that we've not seen before.
        throw(error("Unrecognised expression $stmt"))
    end
end

is_active(::Union{Argument, ID}) = true
is_active(::Any) = false

# Get a bound on the pullback type, given a rule and associated primal types.
function pullback_type(Trule, arg_types)
    T = Core.Compiler.return_type(Tuple{Trule, map(fcodual_type, arg_types)...})
    return (T <: Tuple && T !== Union{} && !(T isa Union)) ? T.parameters[2] : Any
end

# Used by the getfield special-case in call / invoke statments.
@inline function __fwds_pass_no_ad!(f::F, raw_args::Vararg{Any, N}) where {F, N}
    return tuple_splat(__get_primal(f), tuple_map(__get_primal, raw_args))
end

__get_primal(x::CoDual) = primal(x)
__get_primal(x) = x

# Helper used in make_ad_stmts! for call / invoke.
__make_codual(x::P) where {P} = (P <: CoDual ? x : uninit_fcodual(x))::CoDual

# Used in `make_ad_stmts!` method for `Expr(:call, ...)` and `Expr(:invoke, ...)`.
@inline function __run_rvs_pass!(::Type{P}, ::Type{sig}, pb!!, ret_rev_data_ref::Ref, arg_rev_data_refs...) where {P, sig}
    tuple_map(increment_if_ref!, arg_rev_data_refs, pb!!(ret_rev_data_ref[]))
    set_ret_ref_to_zero!!(P, ret_rev_data_ref)
    return nothing
end

@inline increment_if_ref!(ref::Ref, rvs_data) = increment_ref!(ref, rvs_data)
@inline increment_if_ref!(::Ref, ::ZeroRData) = nothing
@inline increment_if_ref!(::Nothing, ::Any) = nothing

@inline increment_ref!(x::Ref, t) = setindex!(x, increment!!(x[], t))
@inline increment_ref!(::Base.RefValue{NoRData}, t) = nothing

# Useful to have this function call for debugging when looking at the generated IRCode.
@inline __pop_pb_stack!(stack) = pop!(stack)

@inline function set_ret_ref_to_zero!!(::Type{P}, r::Ref{R}) where {P, R}
    r[] = zero_like_rdata_from_type(P)
end
@inline set_ret_ref_to_zero!!(::Type{P}, r::Base.RefValue{NoRData}) where {P} = nothing

#
# Runners for generated code. The main job of these functions is to handle the translation
# between differing varargs conventions.
#

struct Pullback{Tprimal, Tpb_oc, Tisva<:Val, Tnvargs<:Val}
    pb_oc::Tpb_oc
    isva::Tisva
    nvargs::Tnvargs
end

function Pullback(
    Tprimal, pb_oc::Tpb_oc, isva::Tisva, nvargs::Tnvargs
) where {Tpb_oc, Tisva, Tnvargs}
    return Pullback{Tprimal, Tpb_oc, Tisva, Tnvargs}(pb_oc, isva, nvargs)
end

@inline (pb::Pullback)(dy) = __flatten_varargs(pb.isva, pb.pb_oc(dy), pb.nvargs)

struct DerivedRule{Tprimal, Tfwds_oc, Tpb_oc, Tisva<:Val, Tnargs<:Val}
    fwds_oc::Tfwds_oc
    pb_oc::Tpb_oc
    isva::Tisva
    nargs::Tnargs
end

function DerivedRule(
    Tprimal, fwds_oc::Tfwds_oc, pb_oc::Tpb_oc, isva::Tisva, nargs::Tnargs
) where {Tfwds_oc, Tpb_oc, Tisva, Tnargs}
    return DerivedRule{Tprimal, Tfwds_oc, Tpb_oc, Tisva, Tnargs}(fwds_oc, pb_oc, isva, nargs)
end

# Extends functionality defined for debug_mode.
function verify_args(::DerivedRule{sig}, ::Tx) where {sig, Tx}
    sig === Tx && return nothing
    msg = "Arguments with sig $Tx do not match signature expected by rule, $sig"
    throw(ArgumentError(msg))
end

_copy(::Nothing) = nothing

function _copy(x::P) where {P<:DerivedRule}
    new_captures = _copy(x.fwds_oc.oc.captures)
    new_fwds_oc = replace_captures(x.fwds_oc, new_captures)
    new_pb_oc = replace_captures(x.pb_oc, new_captures)
    return P(new_fwds_oc, new_pb_oc, x.isva, x.nargs)
end

_copy(x::Symbol) = x

_copy(x::Tuple) = map(_copy, x)

_copy(x::NamedTuple) = map(_copy, x)

_copy(x::Ref{T}) where {T} = isassigned(x) ? Ref{T}(_copy(x[])) : Ref{T}()

_copy(x::Type) = x

_copy(x) = copy(x)

@inline function (fwds::DerivedRule{P, Q, S})(args::Vararg{CoDual, N}) where {P, Q, S, N}
    uf_args = __unflatten_codual_varargs(fwds.isva, args, fwds.nargs)
    pb!! = Pullback(P, fwds.pb_oc, fwds.isva, nvargs(length(args), fwds.nargs))
    return fwds.fwds_oc(uf_args...)::CoDual, pb!!
end

@inline nvargs(n_flat, ::Val{nargs}) where {nargs} = Val(n_flat - nargs + 1)

# If isva, inputs (5.0, (4.0, 3.0)) are transformed into (5.0, 4.0, 3.0).
function __flatten_varargs(::Val{isva}, args, ::Val{nvargs}) where {isva, nvargs}
    isva || return args
    last_el = isa(args[end], NoRData) ? ntuple(n -> NoRData(), nvargs) : args[end]
    return (args[1:end-1]..., last_el...)
end

# If isva and nargs=2, then inputs `(CoDual(5.0, 0.0), CoDual(4.0, 0.0), CoDual(3.0, 0.0))`
# are transformed into `(CoDual(5.0, 0.0), CoDual((5.0, 4.0), (0.0, 0.0)))`.
function __unflatten_codual_varargs(::Val{isva}, args, ::Val{nargs}) where {isva, nargs}
    isva || return args
    group_primal = map(primal, args[nargs:end])
    if fdata_type(tangent_type(_typeof(group_primal))) == NoFData
        grouped_args = zero_fcodual(group_primal)
    else
        grouped_args = CoDual(group_primal, map(tangent, args[nargs:end]))
    end
    return (args[1:nargs-1]..., grouped_args)
end

#
# Rule derivation.
#

_is_primitive(C::Type, mi::Core.MethodInstance) = is_primitive(C, mi.specTypes)
_is_primitive(C::Type, sig::Type) = is_primitive(C, sig)

# Compute the concrete type of the rule that will be returned from `build_rrule`. This is
# important for performance in dynamic dispatch, and to ensure that recursion works
# properly.
function rule_type(interp::MooncakeInterpreter{C}, sig_or_mi; debug_mode) where {C}

    if _is_primitive(C, sig_or_mi)
        return debug_mode ? DebugRRule{typeof(rrule!!)} : typeof(rrule!!)
    end

    ir, _ = lookup_ir(interp, sig_or_mi)
    Treturn = Base.Experimental.compute_ir_rettype(ir)
    isva, _ = is_vararg_and_sparam_names(sig_or_mi)

    arg_types = map(_type, ir.argtypes)
    primal_sig = Tuple{arg_types...}
    arg_fwds_types = Tuple{map(fcodual_type, arg_types)...}
    arg_rvs_types = Tuple{map(rdata_type ∘ tangent_type, arg_types)...}
    rvs_return_type = rdata_type(tangent_type(Treturn))
    return if isconcretetype(Treturn)
        if debug_mode
            DebugRRule{DerivedRule{
                primal_sig,
                MistyClosure{OpaqueClosure{arg_fwds_types, fcodual_type(Treturn)}},
                MistyClosure{OpaqueClosure{Tuple{rvs_return_type}, arg_rvs_types}},
                Val{isva},
                Val{length(ir.argtypes)},
            }}
        else
            DerivedRule{
                primal_sig,
                MistyClosure{OpaqueClosure{arg_fwds_types, fcodual_type(Treturn)}},
                MistyClosure{OpaqueClosure{Tuple{rvs_return_type}, arg_rvs_types}},
                Val{isva},
                Val{length(ir.argtypes)},
            }
        end
    else
        if debug_mode
            DebugRRule{DerivedRule{
                primal_sig,
                MistyClosure{OpaqueClosure{arg_fwds_types, Tfcodual_ret}},
                MistyClosure{OpaqueClosure{Tuple{rvs_return_type}, arg_rvs_types}},
                Val{isva},
                Val{length(ir.argtypes)},
            }} where {Tfcodual_ret <: fcodual_type(Treturn)}
        else
            DerivedRule{
                primal_sig,
                MistyClosure{OpaqueClosure{arg_fwds_types, Tfcodual_ret}},
                MistyClosure{OpaqueClosure{Tuple{rvs_return_type}, arg_rvs_types}},
                Val{isva},
                Val{length(ir.argtypes)},
            } where {Tfcodual_ret <: fcodual_type(Treturn)}
        end
    end
end

"""
    build_rrule(args...; debug_mode=false)

Helper method. Only uses static information from `args`.
"""
function build_rrule(args...; debug_mode=false)
    interp = get_interpreter()
    return build_rrule(interp, _typeof(TestUtils.__get_primals(args)); debug_mode)
end

"""
    build_rrule(sig::Type{<:Tuple})

Equivalent to `build_rrule(Mooncake.get_interpreter(), sig)`.
"""
build_rrule(sig::Type{<:Tuple}) = build_rrule(get_interpreter(), sig)

const MOONCAKE_INFERENCE_LOCK = ReentrantLock()

"""
    build_rrule(interp::MooncakeInterpreter{C}, sig_or_mi; debug_mode=false) where {C}

Returns a `DerivedRule` which is an `rrule!!` for `sig_or_mi` in context `C`. See the
docstring for `rrule!!` for more info.

If `debug_mode` is `true`, then all calls to rules are replaced with calls to `DebugRRule`s.
"""
function build_rrule(
    interp::MooncakeInterpreter{C}, sig_or_mi; debug_mode=false, silence_debug_messages=true
) where {C}

    # To avoid segfaults, ensure that we bail out if the interpreter's world age is greater
    # than the current world age.
    if Base.get_world_counter() > interp.world
        throw(ArgumentError(
            "World age associated to interp is behind current world age. Please " *
            "a new interpreter for the current world age."
        ))
    end

    # If we're compiling in debug mode, let the user know by default.
    if !silence_debug_messages && debug_mode
        @info "Compiling rule for $sig_or_mi in debug mode. Disable for best performance."
    end

    # If we have a hand-coded rule, just use that.
    _is_primitive(C, sig_or_mi) && return (debug_mode ? DebugRRule(rrule!!) : rrule!!)
    lock(MOONCAKE_INFERENCE_LOCK)
    try
        # If we've already derived the OpaqueClosures and info, do not re-derive, just create a
        # copy and pass in new shared data.
        oc_cache_key = ClosureCacheKey(interp.world, (sig_or_mi, debug_mode))
        if haskey(interp.oc_cache, oc_cache_key)
            return _copy(interp.oc_cache[oc_cache_key])
        else

            # Reset id count. This ensures that the IDs generated are the same each time this
            # function runs.
            seed_id!()

            # Grab code associated to the primal.
            ir, _ = lookup_ir(interp, sig_or_mi)
            Treturn = Base.Experimental.compute_ir_rettype(ir)

            # Normalise the IR, and generated BBCode version of it.
            isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
            ir = normalise!(ir, spnames)
            primal_ir = remove_unreachable_blocks!(BBCode(ir))

            # Compute global info.
            info = ADInfo(interp, primal_ir, debug_mode)

            # For each block in the fwds and pullback BBCode, translate all statements. Running this
            # will, in general, push items to `info.shared_data_pairs`.
            ad_stmts_blocks = map(primal_ir.blocks) do primal_blk
                ids = primal_blk.inst_ids
                primal_stmts = map(x -> x.stmt, primal_blk.insts)
                return (primal_blk.id, make_ad_stmts!.(primal_stmts, ids, Ref(info)))
            end

            # Make shared data, and construct BBCode for forwards-pass and pullback.
            fwds_comms_insts, pb_comms_insts = create_comms_insts!(ad_stmts_blocks, info)
            shared_data = shared_data_tuple(info.shared_data_pairs)
            Tshared_data = _typeof(shared_data)

            fwds_ir = forwards_pass_ir(
                primal_ir, ad_stmts_blocks, fwds_comms_insts, info, Tshared_data
            )
            pb_ir = pullback_ir(
                primal_ir, Treturn, ad_stmts_blocks, pb_comms_insts, info, Tshared_data
            )
            optimised_fwds_ir = optimise_ir!(IRCode(fwds_ir); do_inline=true)
            optimised_pb_ir = optimise_ir!(IRCode(pb_ir); do_inline=true)
            # @show sig_or_mi
            # @show Treturn
            # @show debug_mode
            # display(ir)
            # display(IRCode(fwds_ir))
            # display(IRCode(pb_ir))
            # display(optimised_fwds_ir)
            # display(optimised_pb_ir)
            # @show length(stmt(ir.stmts))
            # @show length(stmt(optimised_fwds_ir.stmts))
            # @show length(stmt(optimised_pb_ir.stmts))
            fwds_oc = MistyClosure(optimised_fwds_ir, shared_data...; do_compile=true)
            pb_oc = MistyClosure(optimised_pb_ir, shared_data...; do_compile=true)

            # Compute the signature. Needs careful handling with varargs.
            sig = sig_or_mi isa Core.MethodInstance ? sig_or_mi.specTypes : sig_or_mi
            nargs = num_args(info)
            if isva
                sig = Tuple{sig.parameters[1:nargs-1]..., Tuple{sig.parameters[nargs:end]...}}
            end

            raw_rule = DerivedRule(sig, fwds_oc, pb_oc, Val(isva), Val(num_args(info)))
            rule = debug_mode ? DebugRRule(raw_rule) : raw_rule
            interp.oc_cache[oc_cache_key] = rule
            return rule
        end
    finally
        unlock(MOONCAKE_INFERENCE_LOCK)
    end
end

# Given an `OpaqueClosure` `oc`, create a new `OpaqueClosure` of the same type, but with new
# captured variables. This is needed for efficiency reasons -- if `build_rrule` is called
# repeatedly with the same signature and intepreter, it is important to avoid recompiling
# the `OpaqueClosure`s that it produces multiple times, because it can be quite expensive to
# do so.
@eval function replace_captures(oc::Toc, new_captures) where {Toc<:OpaqueClosure}
    return $(Expr(
        :new, :(Toc), :new_captures, :(oc.world), :(oc.source), :(oc.invoke), :(oc.specptr)
    ))
end

# Wrapper for `MistyClosure`s.
function replace_captures(mc::Tmc, new_captures) where {Tmc<:MistyClosure}
    return Tmc(replace_captures(mc.oc, new_captures), mc.ir)
end

const ADStmts = Vector{Tuple{ID, Vector{ADStmtInfo}}}

#=
    create_comms_insts!(ad_stmts_blocks::ADStmts, info::ADInfo)

This function produces code which can be inserted into the forwards-pass and reverse-pass at
specific locations to implement the promise associated to the `comms_id` field of the
`ADStmtInfo` type -- namely that if you assign a value to `comms_id` on the forwards-pass,
the same value will be available at `comms_id` on the reverse-pass.

For each basic block represented in `ADStmts`:
1. create a stack containing a `Tuple` which can hold all of the values associated to the
    `comms_id`s for each statement. Put this stack in shared data.
2. create instructions which can be inserted at the _end_ of the block generated to perform
    the forwards-pass (in `forwards_pass_ir`) which will put all of the data associated to
    the `comms_id`s into shared data, and
3. create instruction which can be inserted at the _start_ of the block generated to perform
    the reverse-pass (in `pullback_ir`), which will extract all of the data put into
    shared data by the instructions generated by the previous point, and assigned them to
    the `comms_id`s.

Returns two a `Tuple{Vector{IDInstPair}, Vector{IDInstPair}`. The nth element of each
`Vector` corresponds to the instructions to be inserted into the forwards- and reverse
passes resp. for the nth block in `ad_stmts_blocks`.
=#
function create_comms_insts!(ad_stmts_blocks::ADStmts, info::ADInfo)
    insts = map(ad_stmts_blocks) do (_, ad_stmts)

        # Get the communication channel for each stmt which has one.
        comms_channels = filter(!=(nothing), map(comms_channel, ad_stmts))
        comms_ids = map(first, comms_channels)

        # Determine the type of the tuple which will contain these, and create a stack which
        # can hold such tuples. Put this stack in shared data, and get its `ID`.
        # Optimise for the case that `TT` is a singleton type -- the result of this
        # optimisation is to directly insert the stack into the code if `TT` is a singleton
        # type, and avoid adding anything to shared data. One notable case in which this
        # will hold is when comms_ids is empty.
        TT = Tuple{map(x -> x[2].type, comms_channels)...}
        stack = Base.issingletontype(TT) ? SingletonStack{TT}() : Stack{TT}()
        comms_stack_id = add_data_if_not_singleton!(info, stack)

        # Create instructions for forwards-pass to create tuple + push onto comms stack.
        tuple_id = ID()
        fwds_insts = IDInstPair[
            (tuple_id, new_inst(Expr(:call, tuple, comms_ids...))),
            (ID(), new_inst(Expr(:call, push!, comms_stack_id, tuple_id))),
        ]

        # Create instructions for reverse-pass to pop comms stack and extract elements of
        # tuple into comms ids.
        rvs_insts = IDInstPair[
            (tuple_id, new_inst(Expr(:call, pop!, comms_stack_id))),
            map(enumerate(comms_ids)) do (n, id)
                (id, new_inst(Expr(:call, getfield, tuple_id, n)))
            end...,
        ]

        return fwds_insts, rvs_insts
    end
    return map(first, insts), map(last, insts)
end

#=
    forwards_pass_ir(ir::BBCode, ad_stmts_blocks::ADStmts, info::ADInfo, Tshared_data)

Produce the IR associated to the `OpaqueClosure` which runs most of the forwards-pass.
=#
function forwards_pass_ir(
    ir::BBCode, ad_stmts_blocks::ADStmts, fwds_comms_insts, info::ADInfo, Tshared_data
)

    is_unique_pred, pred_is_unique_pred = characterise_unique_predecessor_blocks(ir.blocks)

    # Insert a block at the start which extracts all items from the captures field of the
    # `OpaqueClosure`, which contains all of the data shared between the forwards- and
    # reverse-passes. These are assigned to the `ID`s given by the `SharedDataPairs`.
    # Push the entry id onto the block stack if needed. Create `LazyZeroRData` for each
    # argument, and put it in the `Ref` for use on the reverse-pass.
    sds = shared_data_stmts(info.shared_data_pairs)
    if pred_is_unique_pred[ir.blocks[1].id]
        push_block_stack_insts = IDInstPair[]
    else
        push_block_stack_stmt = Expr(
            :call, __push_blk_stack!, info.block_stack_id, info.entry_id.id
        )
        push_block_stack_insts = [(ID(), new_inst(push_block_stack_stmt))]
    end
    lazy_zero_rdata_stmt = Expr(
        :call,
        __assemble_lazy_zero_rdata,
        info.lazy_zero_rdata_ref_id,
        map(n -> Argument(n + 1), 1:num_args(info))...,
    )
    lazy_zero_rdata_insts = [(ID(), new_inst(lazy_zero_rdata_stmt))]
    entry_stmts = vcat(sds, lazy_zero_rdata_insts, push_block_stack_insts)
    entry_block = BBlock(info.entry_id, entry_stmts)

    # Construct augmented version of each basic block from the primal. For each block:
    # 1. pull the translated basic block statements from ad_stmts_blocks,
    # 2. insert a series of statements to log the contents of the `comms_id`s -- see
    #   the `comms_id` field of `ADStmtInfo`,
    # 3. insert a statement which logs the ID of the current block (if necessary to know
    #   how to perform the reverse-pass),
    # 4. return the BBlock.
    blocks = map(ad_stmts_blocks, fwds_comms_insts) do (block_id, ad_stmts), comms_insts

        # Extract the `fwds` fields from the stmts, and create the block for the fwds pass.
        blk = BBlock(block_id, reduce(vcat, map(x -> x.fwds, ad_stmts)))

        # Insert communcation instructions. See `create_comms_insts!` for an explanation.
        for stack_inst in comms_insts
            insert_before_terminator!(blk, stack_inst[1], stack_inst[2])
        end

        # Log the ID of the current basic block. This is needed to know which basic block to
        # jump to during the reverse-pass if the current block is not the unique predecessor
        # of each of its successors (in which case there is no need to log that control
        # passed through this block as opposed to any other).
        if !is_unique_pred[block_id]
            ins_stmt = Expr(:call, __push_blk_stack!, info.block_stack_id, block_id.id)
            insert_before_terminator!(blk, ID(), new_inst(ins_stmt))
        end

        return blk
    end

    # Create and return the `BBCode` for the forwards-pass.
    arg_types = vcat(Tshared_data, map(fcodual_type ∘ _type, ir.argtypes))
    ir = BBCode(vcat(entry_block, blocks), arg_types, ir.sptypes, ir.linetable, ir.meta)
    return remove_unreachable_blocks!(ir)
end

# Going via this function, rather than just calling push!, makes it very straightforward to
# figure out much time is spent pushing to the block stack when profiling AD.
@inline __push_blk_stack!(block_stack::BlockStack, id::Int32) = push!(block_stack, id)

@inline function __assemble_lazy_zero_rdata(r::Ref{T}, args::Vararg{CoDual, N}) where {T<:Tuple, N}
    r[] = __make_tuples(T, args)
    return nothing
end

@generated function __make_tuples(::Type{T}, args::Tuple) where {T}
    lazy_exprs = map(eachindex(T.parameters)) do n
        return :(lazy_zero_rdata($(T.parameters[n]), primal(args[$n])))
    end
    return Expr(:call, tuple, lazy_exprs...)
end

#=
    pullback_ir(ir::BBCode, Tret, ad_stmts_blocks::ADStmts, info::ADInfo, Tshared_data)

Produce the IR associated to the `OpaqueClosure` which runs most of the pullback.
=#
function pullback_ir(
    ir::BBCode, Tret, ad_stmts_blocks::ADStmts, pb_comms_insts, info::ADInfo, Tshared_data
)

    # Compute the argument types associated to the reverse-pass.
    arg_types = vcat(Tshared_data, rdata_type(tangent_type(Tret)))

    # Compute the blocks which return in the primal.
    primal_exit_blocks_inds = findall(is_reachable_return_node ∘ terminator, ir.blocks)

    #
    # Short-circuit for non-terminating primals -- applies to a tiny fraction of primals:
    #

    # If there are no blocks which successfully return in the primal, then the primal never
    # terminates without throwing, meaning that if AD hits this function, it definitely
    # won't succeed on the forwards-pass. As such, the reverse-pass can just be a no-op.
    if isempty(primal_exit_blocks_inds)
        blocks = [BBlock(ID(), [(ID(), new_inst(ReturnNode(nothing)))])]
        return BBCode(blocks, arg_types, ir.sptypes, ir.linetable, ir.meta)
    end

    #
    # Standard path pullback generation -- applies to 99% of primals:
    #

    # Create entry block which:
    # 1. extracts items from shared data to the correct IDs,
    # 2. creates `Ref`s (which will be optimised away later) to hold rdata for all ssas,
    # 3. create switch statement to block which terminated the forwards pass. If there is
    #   only a single block in the primal containing a reachable ReturnNode, then there is
    #   no need to pop the block stack.
    data_stmts = shared_data_stmts(info.shared_data_pairs)
    rev_data_ref_stmts = reverse_data_ref_stmts(info)
    exit_blocks_ids = map(n -> ir.blocks[n].id, primal_exit_blocks_inds)
    switch_stmts = make_switch_stmts(exit_blocks_ids, length(exit_blocks_ids) == 1, info)
    entry_block = BBlock(ID(), vcat(data_stmts, rev_data_ref_stmts, switch_stmts))

    # For each basic block in the primal:
    # 1. if the block is reachable on the reverse-pass, the bulk of its statements are the
    #   translated basic block statements, in reverse.
    # 2. if, on the other hand, the block is provably not reachable on the reverse-pass,
    #   return a block with nothing in it. At present we only assert that a block is not
    #   reachable if it ends with an unreachable `Core.ReturnNode`.
    # 3. if we need to pop the predecessor stack, pop it. We don't need to pop it if there
    #   is only a single predecessor to this block, and said predecessor is a _unique_
    #   _predecessor_ (see characterise_unique_predecessor_blocks for more info), as its
    #   ID is uniquely determined, and nothing will have been put on to the block stack
    #   during the forwards-pass (see how the output of
    #   characterise_unique_predecessor_blocks is used in forwards_pass_ir).
    # 4. if the block began with one or more PhiNodes, then handle their rdata.
    # 5. jump to the predecessor block.
    ps = compute_all_predecessors(ir)
    _, pred_is_unique_pred = characterise_unique_predecessor_blocks(ir.blocks)
    main_blocks = map(
        ad_stmts_blocks, enumerate(ir.blocks), pb_comms_insts
    ) do (blk_id, ad_stmts), (n, blk), comms_insts

        # Short-circuit if we know that this block cannot be reached on the reverse-pass.
        if is_unreachable_return_node(terminator(blk))
            return BBlock(blk_id, [(ID(), new_inst(nothing))])
        end

        # Extract reverse-stmts from ad_stmts.
        rvs_ad_stmts = reduce(vcat, [x.rvs for x in reverse(ad_stmts)])

        # Conclude the block.
        pred_ids = vcat(ps[blk.id], n == 1 ? [info.entry_id] : ID[])
        tmp = pred_is_unique_pred[blk_id]
        additional_stmts, new_blocks = conclude_rvs_block(blk, pred_ids, tmp, info)

        # Combine all blocks and return. See `create_comms_insts!` for more info regarding
        # `comms_insts`.
        rvs_block = BBlock(blk_id, vcat(comms_insts, rvs_ad_stmts, additional_stmts))
        return vcat(rvs_block, new_blocks)
    end
    main_blocks = reduce(vcat, main_blocks)

    # Create an exit block. Dereferences reverse-data for arguments, increments a zero rdata
    # against it to ensure that it is of the correct type, and returns it.
    arg_rdata_ref_ids = map(n -> info.arg_rdata_ref_ids[Argument(n)], 1:num_args(info))

    # De-reference the lazy zero rdata.
    lazy_zero_rdata_tuple_id = ID()
    lazy_zero_rdata_tuple = new_inst(Expr(:call, getindex, info.lazy_zero_rdata_ref_id))

    # For each argument, dereference its rdata, and increment said rdata against the zero
    # rdata element.
    final_ids = Vector{ID}()
    rdata_extraction_stmts = map(eachindex(arg_rdata_ref_ids)) do n

        # De-reference the nth rdata.
        rdata_id = ID()
        rdata = new_inst(Expr(:call, getindex, arg_rdata_ref_ids[n]))

        # Get the nth lazy zero rdata.
        lazy_zero_rdata_id = ID()
        lazy_zero_rdata = new_inst(Expr(:call, getfield, lazy_zero_rdata_tuple_id, n))

        # Instantiate the nth zero rdata.
        zero_rdata_id = ID()
        zero_rdata = new_inst(Expr(:call, instantiate, lazy_zero_rdata_id))

        # Increment the rdata.
        final_rdata_id = ID()
        final_rdata = new_inst(Expr(:call, increment!!, rdata_id, zero_rdata_id))

        # Log the ID of the rdata to return.
        push!(final_ids, final_rdata_id)

        return IDInstPair[
            (rdata_id, rdata),
            (lazy_zero_rdata_id, lazy_zero_rdata),
            (zero_rdata_id, zero_rdata),
            (final_rdata_id, final_rdata),
        ]
    end

    # Construct a tuple containing all of the rdata.
    deref_id = ID()
    deref = new_inst(Expr(:call, tuple, final_ids...))

    ret = new_inst(ReturnNode(deref_id))
    exit_block = BBlock(
        info.entry_id,
        vcat(
            (lazy_zero_rdata_tuple_id, lazy_zero_rdata_tuple),
            rdata_extraction_stmts...,
            [(deref_id, deref), (ID(), ret)],
        ),
    )

    # Create and return `BBCode` for the pullback. Sort the blocks and remove any blocks
    # which are unreachable, in the sense that they have no predecessors (except the entry
    # block). This ought not to be necessary, but _appears_ to be necessary in order to
    # avoid annoying the Julia compiler.
    blks = vcat(entry_block, main_blocks, exit_block)
    pb_ir = BBCode(blks, arg_types, ir.sptypes, ir.linetable, ir.meta)
    return remove_unreachable_blocks!(_sort_blocks!(pb_ir))
end

#=
    conclude_rvs_block(
        blk::BBlock, pred_ids::Vector{ID}, pred_is_unique_pred::Bool, info::ADInfo
    )

Generates code which is inserted at the end of each counterpart block in the reverse-pass.
Handles phi nodes, and choosing the correct next block to switch to.
=#
function conclude_rvs_block(
    blk::BBlock, pred_ids::Vector{ID}, pred_is_unique_pred::Bool, info::ADInfo
)
    # Get the PhiNodes and their IDs.
    phi_ids, phis = phi_nodes(blk)

    # If there are no PhiNodes in this block, switch directly to the predecessor.
    if length(phi_ids) == 0
        return make_switch_stmts(pred_ids, pred_is_unique_pred, info), BBlock[]
    end

    # Create statements which extract + zero the rdata refs associated to them.
    rdata_ids = map(_ -> ID(), phi_ids)
    deref_stmts = map(phi_ids, rdata_ids) do phi_id, deref_id
        P = get_primal_type(info, phi_id)
        r = get_rev_data_id(info, phi_id)
        return (deref_id, new_inst(Expr(:call, __deref_and_zero, P, r)))
    end

    # For each predecessor, create a `BBlock` which processes its corresponding edge in
    # each of the `PhiNode`s.
    new_blocks = map(pred_ids) do pred_id
        values = Any[__get_value(pred_id, p.stmt) for p in phis]
        return rvs_phi_block(pred_id, rdata_ids, values, info)
    end
    new_pred_ids = map(blk -> blk.id, new_blocks)
    switch = make_switch_stmts(pred_ids, new_pred_ids, pred_is_unique_pred, info)
    return vcat(deref_stmts, switch), new_blocks
end

# Helper functionality for conclude_rvs_block.
function __get_value(edge::ID, x::IDPhiNode)
    edge in x.edges || return nothing
    n = only(findall(==(edge), x.edges))
    return isassigned(x.values, n) ? x.values[n] : nothing
end

# Helper, used in conclude_rvs_block.
@inline function __deref_and_zero(::Type{P}, x::Ref) where {P}
    t = x[]
    x[] = Mooncake.zero_like_rdata_from_type(P)
    return t
end

#=
    rvs_phi_block(pred_id::ID, rdata_ids::Vector{ID}, values::Vector{Any}, info::ADInfo)

Produces a `BBlock` which runs the reverse-pass for the edge associated to `pred_id` in a
collection of `IDPhiNode`s, and then goes to the block associated to `pred_id`.

For example, suppose that we encounter the following collection of `PhiNode`s at the start
of some block:
```julia
%6 = φ (#2 => _1, #3 => %5)
%7 = φ (#2 => 5., #3 => _2)
```
Let the tangent refs associated to `%6`, `%7`, and `_1`` be denoted `t%6`, `t%7`, and `t_1`
resp., and let `pred_id` be `#2`, then this function will produce a basic block of the form
```julia
increment_ref!(t_1, t%6)
nothing
goto #2
```
The call to `increment_ref!` appears because `_1` is the value associated to`%6` when the
primal code comes from `#2`. Similarly, the `goto #2` statement appears because we came from
`#2` on the forwards-pass. There is no `increment_ref!` associated to `%7` because `5.` is a
constant. We emit a `nothing` statement, which the compiler will happily optimise away later
on.

The same ideas apply if `pred_id` were `#3`. The block would end with `#3`, and there would
be two `increment_ref!` calls because both `%5` and `_2` are not constants.
=#
function rvs_phi_block(pred_id::ID, rdata_ids::Vector{ID}, values::Vector{Any}, info::ADInfo)
    @assert length(rdata_ids) == length(values)
    inc_stmts = map(rdata_ids, values) do id, val
        stmt = Expr(:call, increment_if_ref!, get_rev_data_id(info, val), id)
        return (ID(), new_inst(stmt))
    end
    goto_stmt = (ID(), new_inst(IDGotoNode(pred_id)))
    return BBlock(ID(), vcat(inc_stmts, goto_stmt))
end

#=
    make_switch_stmts(
        pred_ids::Vector{ID}, target_ids::Vector{ID}, pred_is_unique_pred::Bool, info::ADInfo
    )

`preds_ids` comprises the `ID`s associated to all possible predecessor blocks to the primal
block under consideration. Suppose its value is `[ID(1), ID(2), ID(3)]`, then
`make_switch_stmts` emits code along the lines of

```julia
prev_block = pop!(block_stack)
not_pred_was_1 = !(prev_block == ID(1))
not_pred_was_2 = !(prev_block == ID(2))
switch(
    not_pred_was_1 => ID(1),
    not_pred_was_2 => ID(2),
    ID(3)
)
```

In words: `make_switch_stmts` emits code which jumps to whichever block preceded the current
block during the forwards-pass.
=#
function make_switch_stmts(
    pred_ids::Vector{ID}, target_ids::Vector{ID}, pred_is_unique_pred::Bool, info::ADInfo
)
    # If there are no predecessors, then we can't possibly have hit this block. This can
    # happen when all of the statements in a block have been eliminated, but the Julia
    # optimiser has not removed the block entirely from the `IRCode`. This often presents as
    # a block containing only a single `nothing` statement.
    # Consequently, we just direct this block back towards the entry node. This is safe, as
    # this block will never get hit, and ensures that the block is safe under re-ordering.
    isempty(pred_ids) && return [(ID(), new_inst(IDGotoNode(info.entry_id)))]

    # Get the predecessor that we actually had in the primal.
    prev_blk_id = ID()
    if pred_is_unique_pred
        prev_blk = new_inst(QuoteNode(only(pred_ids)))
    else
        prev_blk = new_inst(Expr(:call, __pop_blk_stack!, info.block_stack_id))
    end

    # Compare predecessor from primal with all possible predecessors.
    conds = map(pred_ids[1:end-1]) do id
        return (ID(), new_inst(Expr(:call, __switch_case, id.id, prev_blk_id)))
    end

    # Switch statement to change to the predecessor.
    switch_stmt = Switch(Any[c[1] for c in conds], target_ids[1:end-1], target_ids[end])
    switch = (ID(), new_inst(switch_stmt))

    return vcat((prev_blk_id, prev_blk), conds, switch)
end

function make_switch_stmts(pred_ids::Vector{ID}, pred_is_unique_pred::Bool, info::ADInfo)
    return make_switch_stmts(pred_ids, pred_ids, pred_is_unique_pred, info)
end

# Going via this function, rather than just calling pop! directly, makes it easy to figure
# out how much time is spent popping the block stack when profiling performance.
@inline __pop_blk_stack!(block_stack::BlockStack) = pop!(block_stack)

# Helper function emitted by `make_switch_stmts`.
__switch_case(id::Int32, predecessor_id::Int32) = !(id === predecessor_id)

#=
    DynamicDerivedRule(interp::MooncakeInterpreter, debug_mode::Bool)

For internal use only.

A callable data structure which, when invoked, calls an rrule specific to the dynamic types
of its arguments. Stores rules in an internal cache to avoid re-deriving.

This is used to implement dynamic dispatch.
=#
struct DynamicDerivedRule{V}
    cache::V
    debug_mode::Bool
end

DynamicDerivedRule(debug_mode::Bool) = DynamicDerivedRule(Dict{Any, Any}(), debug_mode)

_copy(x::P) where {P<:DynamicDerivedRule} = P(Dict{Any, Any}(), x.debug_mode)

function (dynamic_rule::DynamicDerivedRule)(args::Vararg{Any, N}) where {N}
    sig = Tuple{map(_typeof ∘ primal, args)...}
    rule = get(dynamic_rule.cache, sig, nothing)
    if rule === nothing
        rule = build_rrule(get_interpreter(), sig; debug_mode=dynamic_rule.debug_mode)
        dynamic_rule.cache[sig] = rule
    end
    return rule(args...)
end

#=
    LazyDerivedRule(interp, mi::Core.MethodInstance, debug_mode::Bool)

For internal use only.

A type-stable wrapper around a `DerivedRule`, which only instantiates the `DerivedRule`
when it is first called. This is useful, as it means that if a rule does not get run, it
does not have to be derived.

If `debug_mode` is `true`, then the rule constructed will be a `DebugRRule`. This is useful
when debugging, but should usually be switched off for production code as it (in general)
incurs some runtime overhead.

Note: the signature of the primal for which this is a rule is stored in the type. The only
reason to keep this around is for debugging -- it is very helpful to have this type visible
in the stack trace when something goes wrong, as it allows you to trivially determine which
bit of your code is the culprit.
=#
mutable struct LazyDerivedRule{primal_sig, Trule}
    debug_mode::Bool
    mi::Core.MethodInstance
    rule::Trule
    function LazyDerivedRule(mi::Core.MethodInstance, debug_mode::Bool)
        interp = get_interpreter()
        return new{mi.specTypes, rule_type(interp, mi; debug_mode)}(debug_mode, mi)
    end
    function LazyDerivedRule{Tprimal_sig, Trule}(
        mi::Core.MethodInstance, debug_mode::Bool
    ) where {Tprimal_sig, Trule}
        return new{Tprimal_sig, Trule}(debug_mode, mi)
    end
end

_copy(x::P) where {P<:LazyDerivedRule} = P(x.mi, x.debug_mode)

@inline function (rule::LazyDerivedRule)(args::Vararg{Any, N}) where {N}
    isdefined(rule, :rule) || _build_rule!(rule, args)
    return rule.rule(args...)
end

struct BadRuleTypeException <: Exception
    mi::Core.MethodInstance
    sig::Type
    actual_rule_type::Type
    expected_rule_type::Type
end

function Base.showerror(io::IO, err::BadRuleTypeException)
    println(io, "BadRuleTypeException:")
    println(io)
    println(io, "Rule is of type:")
    println(io, err.actual_rule_type)
    println(io)
    println(io, "However, expected rule to be of type:")
    println(io, err.expected_rule_type)
    println(io)
    println(io, "This error occured for $(err.mi) with signature:")
    println(io, err.sig)
    println(io)
    msg = "Usually this error is indicative of something having gone wrong in the " *
        "compilation of the rule in question. Look at the error message for the error " *
        "which caused this error (below) for more details. If the error below does not " *
        "immediately give you enough information to debug what is going on, consider " *
        "building the rule for the signature above, and inspecting the IR."
    println(io, msg)
end

@noinline function _build_rule!(rule::LazyDerivedRule{sig, Trule}, args) where {sig, Trule}
    derived_rule = build_rrule(get_interpreter(), rule.mi; debug_mode=rule.debug_mode)
    if derived_rule isa Trule
        rule.rule = derived_rule
    else
        @warn "Unable to put rule in rule field. A `BadRuleTypeException` should be thrown."
        err = BadRuleTypeException(rule.mi, sig, typeof(derived_rule), Trule)
        try
            derived_rule(args...)
        catch
            throw(err)
        end
        @warn "`BadRuleTypException was _not_ thrown. Throwing now."
        throw(err)
    end
end
