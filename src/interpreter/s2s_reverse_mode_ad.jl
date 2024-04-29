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
    shared_data_stmts(p::SharedDataPairs)::Vector{Tuple{ID, NewInstruction}}

Produce a sequence of id-statment pairs which will extract the data from
`shared_data_tuple(p)` such that the correct value is associated to the correct `ID`.

For example, if `p.pairs` is
```julia
[(ID(5), 5.0), (ID(3), "hello")]
```
then the output of this function is
```julia
Tuple{ID, NewInstruction}[
    (ID(5), new_inst(:(getfield(_1, 1)))),
    (ID(3), new_inst(:(getfield(_1, 2)))),
]
```
=#
function shared_data_stmts(p::SharedDataPairs)::Vector{Tuple{ID, NewInstruction}}
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

- `interp`: a `TapirInterpreter`.
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
- `safety_on`: if `true`, run in "safe mode" -- wraps all rule calls in `SafeRRule`. This is
    applied recursively, so that safe mode is also switched on in derived rules.
- `is_used_dict`: for each `ID` associated to a line of code, is `false` if line is not used
    anywhere in any other line of code.
=#
struct ADInfo
    interp::PInterp
    block_stack_id::ID
    block_stack::BlockStack
    entry_id::ID
    shared_data_pairs::SharedDataPairs
    arg_types::Dict{Argument, Any}
    ssa_insts::Dict{ID, NewInstruction}
    arg_rdata_ref_ids::Dict{Argument, ID}
    ssa_rdata_ref_ids::Dict{ID, ID}
    safety_on::Bool
    is_used_dict::Dict{ID, Bool}
end

# The constructor that you should use for ADInfo if you don't have a BBCode lying around.
# See the definition of the ADInfo struct for info on the arguments.
function ADInfo(
    interp::PInterp,
    arg_types::Dict{Argument, Any},
    ssa_insts::Dict{ID, NewInstruction},
    is_used_dict::Dict{ID, Bool},
    safety_on::Bool,
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
        safety_on,
        is_used_dict,
    )
end

# The constructor you should use for ADInfo if you _do_ have a BBCode lying around. See the
# ADInfo struct for information regarding `interp` and `safety_on`.
function ADInfo(interp::PInterp, ir::BBCode, safety_on::Bool)
    arg_types = Dict{Argument, Any}(
        map(((n, t),) -> (Argument(n) => _type(t)), enumerate(ir.argtypes))
    )
    stmts = collect_stmts(ir)
    ssa_insts = Dict{ID, NewInstruction}(stmts)
    is_used_dict = characterise_used_ids(stmts)
    return ADInfo(interp, arg_types, ssa_insts, is_used_dict, safety_on)
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
@inline @generated function __make_ref(::Type{P}) where {P}
    R = zero_like_rdata_type(P)
    return :(Ref{$R}(Tapir.zero_like_rdata_from_type(P)))
end

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

struct RRuleWrapperPb{Tpb!!, Tl}
    pb!!::Tpb!!
    l::Tl
end

(rule::RRuleWrapperPb)(dy) = rule.pb!!(increment!!(dy, instantiate(rule.l)))

@inline function (rule::RRuleZeroWrapper{R})(f::F, args::Vararg{CoDual, N}) where {R, F, N}
    y, pb!! = rule.rule(f, args...)
    l = LazyZeroRData(primal(y))
    return y::CoDual, (pb!! isa NoPullback ? pb!! : RRuleWrapperPb(pb!!, l))
end

#=
    ADStmtInfo

Data structure which contains the result of `make_ad_stmts!`. Fields are
- `line`: the ID associated to the primal line from which this is derived
- `fwds`: the instructions which run the forwards-pass of AD
- `rvs`: the instructions which run the reverse-pass of AD / the pullback
=#
struct ADStmtInfo
    line::ID
    fwds::Vector{Tuple{ID, NewInstruction}}
    rvs::Vector{Tuple{ID, NewInstruction}}
end

# Convenient constructor for `ADStmtInfo`. If either `fwds` or `rvs` is not a vector,
# `__vec` promotes it to a single-element `Vector`.
ad_stmt_info(line::ID, fwds, rvs) = ADStmtInfo(line, __vec(line, fwds), __vec(line, rvs))

__vec(line::ID, x::Any) = __vec(line, new_inst(x))
__vec(line::ID, x::NewInstruction) = Tuple{ID, NewInstruction}[(line, x)]
__vec(line::ID, x::Vector{Tuple{ID, Any}}) = throw(error("boooo"))
__vec(line::ID, x::Vector{Tuple{ID, NewInstruction}}) = x

#=
    make_ad_stmts(inst::NewInstruction, line::ID, info::ADInfo)::ADStmtInfo

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
make_ad_stmts!(::Nothing, line::ID, ::ADInfo) = ad_stmt_info(line, nothing, nothing)

# `ReturnNode`s have a single field, `val`, for which there are three cases to consider:
#
# 1. `val` is undefined: this `ReturnNode` is unreachable. Consequently, we'll never hit the
#   associated statements on the forwards-pass of pullback. We just return the original
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
    is_reachable_return_node(stmt) || return ad_stmt_info(line, inc_args(stmt), nothing)
    if is_active(stmt.val)
        rdata_id = get_rev_data_id(info, stmt.val)
        rvs = new_inst(Expr(:call, increment_ref!, rdata_id, Argument(2)))
        return ad_stmt_info(line, inc_args(stmt), rvs)
    else
        return ad_stmt_info(line, ReturnNode(const_codual(stmt.val, info)), nothing)
    end
end

# Identity forwards-pass, no-op reverse. No shared data.
function make_ad_stmts!(stmt::IDGotoNode, line::ID, ::ADInfo)
    return ad_stmt_info(line, inc_args(stmt), nothing)
end

# Identity forwards-pass, no-op reverse. No shared data.
function make_ad_stmts!(stmt::IDGotoIfNot, line::ID, ::ADInfo)
    stmt = inc_args(stmt)
    if stmt.cond isa Union{Argument, ID}
        # If cond refers to a register, then the primal must be extracted.
        cond_id = ID()
        fwds = [
            (cond_id, new_inst(Expr(:call, primal, stmt.cond))),
            (line, new_inst(IDGotoIfNot(cond_id, stmt.dest), Any)),
        ]
        return ad_stmt_info(line, fwds, nothing)
    else
        # If something other than a register, then there is nothing to do.
        return ad_stmt_info(line, stmt, nothing)
    end
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
    new_type = fwds_codual_type(get_primal_type(info, line))
    _inst = new_inst(IDPhiNode(stmt.edges, new_vals), new_type, info.ssa_insts[line].flag)
    # _inst = new_inst(IDPhiNode(stmt.edges, new_vals))
    return ad_stmt_info(line, _inst, nothing)
end

function make_ad_stmts!(stmt::PiNode, line::ID, info::ADInfo)
    isa(stmt.val, Union{Argument, ID}) || unhandled_feature("PiNode: $stmt")

    # Get the primal type of this line, and the rdata refs for the `val` of this `PiNode`
    # and this line itself.
    P = get_primal_type(info, line)
    val_rdata_ref_id = get_rev_data_id(info, stmt.val)
    output_rdata_ref_id = get_rev_data_id(info, line)

    # Assemble the above lines and construct reverse-pass.
    return ad_stmt_info(
        line,
        PiNode(stmt.val, fwds_codual_type(_type(stmt.typ))),
        Expr(:call, __pi_rvs!, P, val_rdata_ref_id, output_rdata_ref_id),
    )
end

@inline function __pi_rvs!(::Type{P}, val_rdata_ref::Ref, output_rdata_ref::Ref) where {P}
    increment_ref!(val_rdata_ref, __deref_and_zero(P, output_rdata_ref))
    return nothing
end

# Constant GlobalRefs are handled. See const_codual. Non-constant
# GlobalRefs are handled by assuming that they are constant, and creating a CoDual with
# the value. We then check at run-time that the value has not changed.
function make_ad_stmts!(stmt::GlobalRef, line::ID, info::ADInfo)
    if isconst(stmt)
        return const_ad_stmt(stmt, line, info)
    else
        x = const_codual(getglobal(stmt.mod, stmt.name), info)
        gref_id = ID()
        fwds = [
            (gref_id, new_inst(stmt)),
            (line, new_inst(Expr(:call, __verify_const, gref_id, x))),
        ]
        return ad_stmt_info(line, fwds, nothing)
    end
end

# Helper used by `make_ad_stmts! ` for `GlobalRef`.
@noinline function __verify_const(global_ref, stored_value)
    @assert global_ref == primal(stored_value)
    return uninit_fcodual(global_ref)
end

# QuoteNodes are constant. See const_codual for details.
make_ad_stmts!(stmt::QuoteNode, line::ID, info::ADInfo) = const_ad_stmt(stmt, line, info)

# Literal constant. See const_codual for details.
make_ad_stmts!(stmt, line::ID, info::ADInfo) = const_ad_stmt(stmt, line, info)

# `make_ad_stmts!` for constants.
function const_ad_stmt(stmt, line::ID, info::ADInfo)
    x = const_codual(stmt, info)
    return ad_stmt_info(line, x isa ID ? Expr(:call, identity, x) : x, nothing)
end

# Build a `CoDual` from `stmt`, which will be checked to ensure that its value
# is constant. If the resulting CoDual is a bits type, then it is returned. If it is not,
# then the CoDual is put into shared data, and the ID associated to it in the forwards-
# and reverse-passes returned.
function const_codual(stmt, info::ADInfo)
    x = build_const_codual(stmt)
    return isbitstype(_typeof(x)) ? x : add_data!(info, x)
end

# Create a `CoDual` containing the values associated to `stmt`, a zero forwards data.
build_const_codual(stmt) = uninit_fcodual(get_const_primal_value(stmt))

# Get the value associated to `x`. For `GlobalRef`s, verify that `x` is indeed a constant,
# and error if it is not.
function get_const_primal_value(x::GlobalRef)
    isconst(x) || unhandled_feature("Non-constant GlobalRef not supported: $x")
    return getglobal(x.mod, x.name)
end
get_const_primal_value(x::QuoteNode) = x.value
get_const_primal_value(x) = x

# Tapir does not yet handle `PhiCNode`s. Throw an error if one is encountered.
function make_ad_stmts!(stmt::Core.PhiCNode, ::ID, ::ADInfo)
    unhandled_feature("Encountered PhiCNode: $stmt")
end

# Tapir does not yet handle `UpsilonNode`s. Throw an error if one is encountered.
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

        # If this function is side-effect free, and its value is unused, then just leave the
        # call alone, and do nothing of the reverse-pass. This functionality ought to be
        # generalised to more things.
        if !is_used(info, line) && get_const_primal_value(args[1]) == getfield
            fwds = new_inst(Expr(:call, __fwds_pass_no_ad!, map(__inc, args)...))
            return ad_stmt_info(line, fwds, nothing)
        end

        # Construct signature, and determine how the rrule is to be computed.
        sig = Tuple{arg_types...}
        raw_rule = if is_primitive(context_type(info.interp), sig)
            rrule!! # intrinsic / builtin / thing we provably have rule for
        elseif is_invoke
            LazyDerivedRule(info.interp, sig, info.safety_on) # Static dispatch
        else
            DynamicDerivedRule(info.interp, info.safety_on)  # Dynamic dispatch
        end

        # Wrap the raw rule in a struct which ensures that any `ZeroRData`s are stripped
        # away before the raw_rule is called.
        zero_safe_rule = RRuleZeroWrapper(raw_rule)

        # If we need to run safety checks, wrap the rule in a correctness checker.
        rule = info.safety_on ? SafeRRule(zero_safe_rule) : zero_safe_rule

        # If the rule is `rrule!!` (i.e. `sig` is primitive), then don't bother putting
        # the rule into shared data, because it's safe to put it directly into the code.
        rule_ref = add_data_if_not_singleton!(info, rule)

        # If the type of the pullback is a singleton type, then there is no need to store it
        # in the shared data, it can be interpolated directly into the generated IR.
        T_pb!! = pullback_type(_typeof(rule), arg_types)
        pb_stack_id = add_data_if_not_singleton!(info, build_pb_stack(T_pb!!))

        #
        # Write forwards-pass. These statements are written out manually, as writing them
        # out in a function would prevent inlining in type-unstable situations.
        #

        # Make arguments to rrule call.
        codual_arg_ids = map(_ -> ID(), args)
        __codual_args = map(arg -> Expr(:call, __make_codual, __inc(arg)), args)
        codual_args = Tuple{ID, NewInstruction}[
            (id, new_inst(arg)) for (id, arg) in zip(codual_arg_ids, __codual_args)
        ]

        # Make call to rule.
        rule_call_id = ID()
        rule_call = Expr(:call, rule_ref, codual_arg_ids...)

        # Extract output.
        raw_output_id = ID()
        raw_output = Expr(:call, getfield, rule_call_id, 1)

        # Extract pullback.
        pb_id = ID()
        pb = Expr(:call, getfield, rule_call_id, 2)

        # Push the pullback stack.
        push_pb_stack_id = ID()
        push_pb_stack = Expr(:call, __push_pb_stack!, pb_stack_id, pb_id)

        # Impose a type assertion to ensure the CoDual returned by the rule is correct.
        output_id = line
        F = fwds_codual_type(get_primal_type(info, line))
        output = Expr(:call, Core.typeassert, raw_output_id, F)

        # Create statements associated to forwards-pass.
        fwds = vcat(
            codual_args,
            Tuple{ID, NewInstruction}[
                (rule_call_id, new_inst(rule_call)),
                (raw_output_id, new_inst(raw_output)),
                (pb_id, new_inst(pb)),
                (push_pb_stack_id, new_inst(push_pb_stack)),
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
                __rvs_pass!,
                get_primal_type(info, line),
                pb_stack_id,
                get_rev_data_id(info, line),
                map(Base.Fix1(get_rev_data_id, info), args)...,
            )
        end
        return ad_stmt_info(line, fwds, new_inst(rvs_pass))

    elseif Meta.isexpr(stmt, :boundscheck)
        # For some reason the compiler cannot handle boundscheck statements when we run it
        # again. Consequently, emit `true` to be safe. Ideally we would handle this in a
        # more natural way, but I'm not sure how to do that.
        return ad_stmt_info(line, zero_fcodual(true), nothing)

    elseif Meta.isexpr(stmt, :code_coverage_effect)
        # Code coverage irrelevant for derived code.
        return ad_stmt_info(line, nothing, nothing)

    elseif Meta.isexpr(stmt, :copyast)
        # Get constant out and shove it in shared storage.
        x = const_codual(stmt.args[1], info)
        return ad_stmt_info(line, Expr(:call, identity, x), nothing)

    elseif Meta.isexpr(stmt, :loopinfo)
        # Cannot pass loopinfo back through the optimiser for some reason.
        # At the time of writing, I am unclear why this is not possible.
        return ad_stmt_info(line, nothing, nothing)

    elseif stmt.head in [
        :enter,
        :gc_preserve_begin,
        :gc_preserve_end,
        :leave,
        :pop_exception,
        :throw_undef_if_not,
    ]
        # Expressions which do not require any special treatment.
        return ad_stmt_info(line, stmt, nothing)
    else
        # Encountered an expression that we've not seen before.
        throw(error("Unrecognised expression $stmt"))
    end
end

is_active(::Union{Argument, ID}) = true
is_active(::Any) = false

# Get a bound on the pullback type, given a rule and associated primal types.
function pullback_type(Trule, arg_types)
    T = Core.Compiler.return_type(Tuple{Trule, map(fwds_codual_type, arg_types)...})
    return (T <: Tuple && T !== Union{} && !(T isa Union)) ? T.parameters[2] : Any
end

# Build a stack to contain the pullback. Specialises on whether the pullback is a singleton,
# and whether we get to know the concrete type of the pullback or not.
function build_pb_stack(T_pb!!)
    return Base.issingletontype(T_pb!!) ? SingletonStack{T_pb!!}() : Stack{T_pb!!}()
end

@inline function __fwds_pass_no_ad!(f::F, raw_args::Vararg{Any, N}) where {F, N}
    return tuple_splat(__get_primal(f), tuple_map(__get_primal, raw_args))
end

__get_primal(x::CoDual) = primal(x)
__get_primal(x) = x

__make_codual(x::P) where {P} = (P <: CoDual ? x : uninit_fcodual(x))::CoDual

# Useful to have this function call for debugging when looking at the generated IRCode.
@inline __push_pb_stack!(stack, pb!!) = push!(stack, pb!!)

# Used in `make_ad_stmts!` method for `Expr(:call, ...)` and `Expr(:invoke, ...)`.
@inline function __rvs_pass!(P, pb_stack, ret_rev_data_ref, arg_rev_data_refs...)::Nothing
    __run_rvs_pass!(P, __pop_pb_stack!(pb_stack), ret_rev_data_ref, arg_rev_data_refs...)
end

# If `NoPullback` is the pullback, then there is nothing to do. Moreover, since the
# reverse-data accumulated in the `ret_rev_data_ref` is never used, we don't even need to
# bother reseting it's value to zero.
@inline __run_rvs_pass!(::Any, ::NoPullback, ::Ref, arg_rev_data_refs...) = nothing

@inline function __run_rvs_pass!(P, pb!!, ret_rev_data_ref::Ref, arg_rev_data_refs...)
    tuple_map(increment_if_ref!, arg_rev_data_refs, pb!!(ret_rev_data_ref[]))
    set_ret_ref_to_zero!!(P, ret_rev_data_ref)
    return nothing
end

@inline increment_if_ref!(ref::Ref, rvs_data) = increment_ref!(ref, rvs_data)
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
# Runners for generated code.
#

struct Pullback{Tpb_oc, Tisva<:Val, Tnvargs<:Val}
    pb_oc::Tpb_oc
    isva::Tisva
    nvargs::Tnvargs
end

@inline (pb::Pullback)(dy) = __flatten_varargs(pb.isva, pb.pb_oc(dy), pb.nvargs)

struct DerivedRule{Tfwds_oc, Tpb_oc, Tisva<:Val, Tnargs<:Val}
    fwds_oc::Tfwds_oc
    pb_oc::Tpb_oc
    isva::Tisva
    nargs::Tnargs
end

@inline function (fwds::DerivedRule{P, Q, S})(args::Vararg{CoDual, N}) where {P, Q, S, N}
    uf_args = __unflatten_codual_varargs(fwds.isva, args, fwds.nargs)
    pb!! = Pullback(fwds.pb_oc, fwds.isva, nvargs(length(args), fwds.nargs))
    return fwds.fwds_oc(uf_args...)::CoDual, pb!!
end

@inline nvargs(n_flat, ::Val{nargs}) where {nargs} = Val(n_flat - nargs + 1)

# Compute the concrete type of the rule that will be returned from `build_rrule`. This is
# important for performance in dynamic dispatch, and to ensure that recursion works
# properly.
function rule_type(interp::TapirInterpreter{C}, ::Type{sig}) where {C, sig}
    is_primitive(C, sig) && return typeof(rrule!!)

    ir, _ = lookup_ir(interp, sig)
    Treturn = Base.Experimental.compute_ir_rettype(ir)
    isva, _ = is_vararg_sig_and_sparam_names(sig)

    arg_types = map(_type, ir.argtypes)
    arg_fwds_types = Tuple{map(fwds_codual_type, arg_types)...}
    arg_rvs_types = Tuple{map(rdata_type ∘ tangent_type, arg_types)...}
    fwds_return_codual = fwds_codual_type(Treturn)
    rvs_return_type = rdata_type(tangent_type(Treturn))
    if isconcretetype(fwds_return_codual)
        return DerivedRule{
            Core.OpaqueClosure{arg_fwds_types, fwds_return_codual},
            Core.OpaqueClosure{Tuple{rvs_return_type}, arg_rvs_types},
            Val{isva},
            Val{length(ir.argtypes)},
        }
    else
        return DerivedRule{
            Core.OpaqueClosure{arg_fwds_types, P} where {P<:fwds_return_codual},
            Core.OpaqueClosure{Tuple{rvs_return_type}, arg_rvs_types},
            Val{isva},
            Val{length(ir.argtypes)},
        }
    end
end

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

"""
    build_rrule(args...)

Helper method. Only uses static information from `args`.
"""
build_rrule(args...) = build_rrule(PInterp(), _typeof(TestUtils.__get_primals(args)))

"""
    build_rrule(interp::PInterp{C}, sig::Type{<:Tuple}; safety_on=false) where {C}

Returns a `DerivedRule` which is an `rrule!!` for `sig` in context `C`. See the docstring
for `rrule!!` for more info.

If `safety_on` is `true`, then all calls to rules are replaced with calls to `SafeRRule`s.
"""
function build_rrule(interp::PInterp{C}, sig::Type{<:Tuple}; safety_on=false) where {C}

    # Reset id count. This ensures that everything in this function is deterministic.
    seed_id!()

    # If we have a hand-coded rule, just use that.
    is_primitive(C, sig) && return (safety_on ? SafeRRule(rrule!!) : rrule!!)

    # Grab code associated to the primal.
    ir, _ = lookup_ir(interp, sig)
    Treturn = Base.Experimental.compute_ir_rettype(ir)

    # Normalise the IR, and generated BBCode version of it.
    isva, spnames = is_vararg_sig_and_sparam_names(sig)
    ir = normalise!(ir, spnames)
    primal_ir = BBCode(ir)

    # Compute global info.
    info = ADInfo(interp, primal_ir, safety_on)

    # For each block in the fwds and pullback BBCode, translate all statements.
    ad_stmts_blocks = map(primal_ir.blocks) do primal_blk
        ids = primal_blk.inst_ids
        primal_stmts = map(x -> x.stmt, primal_blk.insts)
        return (primal_blk.id, make_ad_stmts!.(primal_stmts, ids, Ref(info)))
    end

    # Make shared data, and construct BBCode for forwards-pass and pullback.
    shared_data = shared_data_tuple(info.shared_data_pairs)

    # If we've already derived the OpaqueClosures and info, do not re-derive, just create a
    # copy and pass in new shared data.
    if !haskey(interp.oc_cache, (sig, safety_on))
        fwds_ir = forwards_pass_ir(primal_ir, ad_stmts_blocks, info, _typeof(shared_data))
        pb_ir = pullback_ir(primal_ir, Treturn, ad_stmts_blocks, info, _typeof(shared_data))
        # @show sig, safety_on
        # display(ir)
        # display(IRCode(fwds_ir))
        # display(IRCode(pb_ir))
        optimised_fwds_ir = optimise_ir!(IRCode(fwds_ir); do_inline=true)
        optimised_pb_ir = optimise_ir!(IRCode(pb_ir); do_inline=true)
        # @show length(ir.stmts.inst)
        # @show length(optimised_fwds_ir.stmts.inst)
        # @show length(optimised_pb_ir.stmts.inst)
        # display(optimised_fwds_ir)
        # display(optimised_pb_ir)
        fwds_oc = OpaqueClosure(optimised_fwds_ir, shared_data...; do_compile=true)
        pb_oc = OpaqueClosure(optimised_pb_ir, shared_data...; do_compile=true)
        interp.oc_cache[(sig, safety_on)] = (fwds_oc, pb_oc)
    else
        existing_fwds_oc, existing_pb_oc = interp.oc_cache[(sig, safety_on)]
        fwds_oc = replace_captures(existing_fwds_oc, shared_data)
        pb_oc = replace_captures(existing_pb_oc, shared_data)
    end

    raw_rule = rule_type(interp, sig)(fwds_oc, pb_oc, Val(isva), Val(num_args(info)))
    return safety_on ? SafeRRule(raw_rule) : raw_rule
end

# Given an `OpaqueClosure` `oc`, create a new `OpaqueClosure` of the same type, but with new
# captured variables. This is needed for efficiency reasons -- if `build_rrule` is called
# repeatedly with the same signature and intepreter, it is important to avoid recompiling
# the `OpaqueClosure`s that it produces multiple times, because it can be quite expensive to
# do so.
@eval function replace_captures(oc::Toc, new_captures) where {Toc<:Core.OpaqueClosure}
    return $(Expr(
        :new, :(Toc), :new_captures, :(oc.world), :(oc.source), :(oc.invoke), :(oc.specptr)
    ))
end

const ADStmts = Vector{Tuple{ID, Vector{ADStmtInfo}}}

#=
    forwards_pass_ir(ir::BBCode, ad_stmts_blocks::ADStmts, info::ADInfo, Tshared_data)

Produce the IR associated to the `OpaqueClosure` which runs most of the forwards-pass.
=#
function forwards_pass_ir(ir::BBCode, ad_stmts_blocks::ADStmts, info::ADInfo, Tshared_data)

    is_unique_pred, pred_is_unique_pred = characterise_unique_predecessor_blocks(ir.blocks)

    # Insert a block at the start which extracts all items from the captures field of the
    # `OpaqueClosure`, which contains all of the data shared between the forwards- and
    # reverse-passes. These are assigned to the `ID`s given by the `SharedDataPairs`.
    # Additionally, push the entry id onto the block stack.
    sds = shared_data_stmts(info.shared_data_pairs)
    if pred_is_unique_pred[ir.blocks[1].id]
        entry_stmts = sds
    else
        push_block_stack_stmt = Expr(
            :call, __push_blk_stack!, info.block_stack_id, info.entry_id.id
        )
        entry_stmts = vcat(sds, (ID(), new_inst(push_block_stack_stmt)))
    end
    entry_block = BBlock(info.entry_id, entry_stmts)

    # Construct augmented version of each basic block from the primal. For each block:
    # 1. pull the translated basic block statements from ad_stmts_blocks.
    # 2. insert a statement which logs the ID of the current block to the block stack.
    # 3. construct and return a BBlock.
    blocks = map(ad_stmts_blocks) do (block_id, ad_stmts)
        fwds_stmts = reduce(vcat, map(x -> x.fwds, ad_stmts))
        if !is_unique_pred[block_id]
            ins_loc = length(fwds_stmts) + (isa(fwds_stmts[end][2].stmt, Terminator) ? 0 : 1)
            ins_stmt = Expr(:call, __push_blk_stack!, info.block_stack_id, block_id.id)
            ins_inst = (ID(), new_inst(ins_stmt))
            insert!(fwds_stmts, ins_loc, ins_inst)
        end
        return BBlock(block_id, fwds_stmts)
    end

    # Create and return the `BBCode` for the forwards-pass.
    arg_types = vcat(Tshared_data, map(fwds_codual_type ∘ _type, ir.argtypes))
    return BBCode(vcat(entry_block, blocks), arg_types, ir.sptypes, ir.linetable, ir.meta)
end

@inline __push_blk_stack!(block_stack::BlockStack, id::Int32) = push!(block_stack, id)

#=
    pullback_ir(ir::BBCode, Tret, ad_stmts_blocks::ADStmts, info::ADInfo, Tshared_data)

Produce the IR associated to the `OpaqueClosure` which runs most of the pullback.
=#
function pullback_ir(ir::BBCode, Tret, ad_stmts_blocks::ADStmts, info::ADInfo, Tshared_data)

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
    # Standard path pullback generation -- applied to 99% of primals:
    #

    # Create entry block, which pops the block_stack, creates + initialised the reverse-data
    # Refs, and switches to reverse-pass counterpart to whichever block we were in at the
    # end of the forwards-pass.
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
    #   reachable if it ends with an unreachable return node.
    # 3. if we need to pop the predecessor stack, pop it. We don't need to pop it if there
    #   is only a single predecessor to this block, and said predecessor is a _unique_
    #   _predecessor_ (see characterise_unique_predecessor_blocks for more info), as its
    #   ID is uniquely determined, and nothing will have been put on to the block stack
    #   during the forwards-pass (see how the output of
    #   characterise_unique_predecessor_blocks is used in forwards_pass_ir).
    # 4. if the block began with one or more PhiNodes, then handle their tangents.
    # 5. jump to the predecessor block
    ps = compute_all_predecessors(ir)
    _, pred_is_unique_pred = characterise_unique_predecessor_blocks(ir.blocks)
    main_blocks = map(ad_stmts_blocks, enumerate(ir.blocks)) do (blk_id, ad_stmts), (n, blk)
        if is_unreachable_return_node(terminator(blk))
            rvs_stmts = [(ID(), new_inst(nothing))]
        else
            rvs_stmts = reduce(vcat, [x.rvs for x in reverse(ad_stmts)])
        end
        pred_ids = vcat(ps[blk.id], n == 1 ? [info.entry_id] : ID[])
        tmp = pred_is_unique_pred[blk_id]
        additional_stmts, new_blocks = finalise_rvs_block(blk, pred_ids, tmp, info)
        rvs_block = BBlock(blk_id, vcat(rvs_stmts, additional_stmts))
        return vcat(rvs_block, new_blocks)
    end
    main_blocks = vcat(main_blocks...)

    # Create an exit block. Dereferences reverse-data for arguments and returns it.
    arg_rdata_ref_ids = map(n -> info.arg_rdata_ref_ids[Argument(n)], 1:num_args(info))
    deref_id = ID()
    deref = new_inst(Expr(:call, __deref_arg_rev_data_refs, arg_rdata_ref_ids...))
    ret = new_inst(ReturnNode(deref_id))
    exit_block = BBlock(info.entry_id, [(deref_id, deref), (ID(), ret)])

    # Create and return `BBCode` for the pullback.
    blks = vcat(entry_block, main_blocks, exit_block)
    return _sort_blocks!(BBCode(blks, arg_types, ir.sptypes, ir.linetable, ir.meta))
end

#=

=#
function finalise_rvs_block(
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

# Helper functionality for finalise_rvs_block # REDO THIS DOCUMENTATION BEFORE MERGING!!!
function __get_value(edge::ID, x::IDPhiNode)
    edge in x.edges || return nothing
    n = only(findall(==(edge), x.edges))
    return isassigned(x.values, n) ? x.values[n] : nothing
end

# Helper, used in finalise_rvs_block... SWITCH OUT FINALISE_RVS_BLOCK TO BE A MORE HELPFUL
# FUNCTION NAME.
@inline function __deref_and_zero(::Type{P}, x::Ref) where {P}
    t = x[]
    x[] = Tapir.zero_like_rdata_from_type(P)
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

@inline __pop_blk_stack!(block_stack::BlockStack) = pop!(block_stack)

# Helper function emitted by `make_switch_stmts`.
__switch_case(id::Int32, predecessor_id::Int32) = !(id === predecessor_id)

# Helper function used by `pullback_ir`.
@inline __deref_arg_rev_data_refs(arg_rev_data_refs...) = map(getindex, arg_rev_data_refs)

#=
    DynamicDerivedRule(interp::TapirInterpreter)

For internal use only.

A callable data structure which, when invoked, calls an rrule specific to the dynamic types
of its arguments. Stores rules in an internal cache to avoid re-deriving.

This is used to implement dynamic dispatch.
=#
struct DynamicDerivedRule{T, V}
    interp::T
    cache::V
    safety_on::Bool
end

function DynamicDerivedRule(interp::TapirInterpreter, safety_on::Bool)
    return DynamicDerivedRule(interp, Dict{Any, Any}(), safety_on)
end

function (dynamic_rule::DynamicDerivedRule)(args::Vararg{Any, N}) where {N}
    sig = Tuple{tuple_map(_typeof, tuple_map(primal, args))...}
    is_primitive(context_type(dynamic_rule.interp), sig) && return rrule!!(args...)
    rule = get(dynamic_rule.cache, sig, nothing)
    if rule === nothing
        rule = build_rrule(dynamic_rule.interp, sig; safety_on=dynamic_rule.safety_on)
        dynamic_rule.cache[sig] = rule
    end
    return rule(args...)
end

#=
    LazyDerivedRule(interp, sig, safety_on::Bool)

For internal use only.

A type-stable wrapper around a `DerivedRule`, which only instantiates the `DerivedRule`
when it is first called. This is useful, as it means that if a rule does not get run, it
does not have to be derived.

If `safety_on` is `true`, then the rule constructed will be a `SafeRRule`. This is useful
when debugging, but should usually be switched off for production code as it (in general)
incurs some runtime overhead.
=#
mutable struct LazyDerivedRule{sig, Tinterp<:TapirInterpreter, Trule}
    interp::Tinterp
    safety_on::Bool
    rule::Trule
    function LazyDerivedRule(interp::A, ::Type{sig}, safety_on::Bool) where {A, sig}
        rt = safety_on ? SafeRRule{rule_type(interp, sig)} : rule_type(interp, sig)
        return new{sig, A, rt}(interp, safety_on)
    end
end

function (rule::LazyDerivedRule{sig})(args::Vararg{Any, N}) where {N, sig}
    if !isdefined(rule, :rule)
        rule.rule = build_rrule(rule.interp, sig; safety_on=rule.safety_on)
    end
    return rule.rule(args...)
end
