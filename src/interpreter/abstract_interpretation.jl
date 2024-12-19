# AbstractInterpretation -- this is an instance of a Julia AbstractInterpreter. We use it
# in conjunction with the contexts above to decide what should be inlined and what should
# not be inlined. Similar strategies are employed by Enzyme and Diffractor.

# The most important bit of this code is `inlining_policy` -- the rest is copy + pasted
# boiler plate, largely taken from https://github.com/JuliaLang/julia/blob/2fe4190b3d26b4eee52b2b1b1054ddd6e38a941e/test/compiler/newinterp.jl#L11
#
# Credit: much of the code in here is copied over from the main Julia repo, and from
# Enzyme.jl, which has a very similar set of concerns to Mooncake in terms of avoiding
# inlining primitive functions.
#

struct ClosureCacheKey
    world_age::UInt
    key::Any
end

struct MooncakeCache
    dict::IdDict{Core.MethodInstance,Core.CodeInstance}
end

MooncakeCache() = MooncakeCache(IdDict{Core.MethodInstance,Core.CodeInstance}())

# The method table used by `Mooncake.@mooncake_overlay`.
Base.Experimental.@MethodTable mooncake_method_table

struct MooncakeInterpreter{C} <: CC.AbstractInterpreter
    meta # additional information
    world::UInt
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
    inf_cache::Vector{CC.InferenceResult}
    code_cache::MooncakeCache
    oc_cache::Dict{ClosureCacheKey,Any}
    inline_primitives::Bool
    function MooncakeInterpreter(
        ::Type{C};
        meta=nothing,
        world::UInt=Base.get_world_counter(),
        inf_params::CC.InferenceParams=CC.InferenceParams(),
        opt_params::CC.OptimizationParams=CC.OptimizationParams(),
        inf_cache::Vector{CC.InferenceResult}=CC.InferenceResult[],
        code_cache::MooncakeCache=MooncakeCache(),
        oc_cache::Dict{ClosureCacheKey,Any}=Dict{ClosureCacheKey,Any}(),
        inline_primitives::Bool=false,
    ) where {C}
        return new{C}(meta, world, inf_params, opt_params, inf_cache, code_cache, oc_cache, inline_primitives)
    end
end

# Don't print out the IRCode object, because this tends to pollute the REPL. Just make it
# clear that this is a MistyClosure, which contains an OpaqueClosure.
function Base.show(io::IO, mime::MIME"text/plain", mc::MooncakeInterpreter)
    return _show_interp(io, mime, mc)
end
Base.show(io::IO, mc::MooncakeInterpreter) = _show_interp(io, MIME"text/plain"(), mc)

function _show_interp(io::IO, ::MIME"text/plain", ::MooncakeInterpreter)
    return print(io, "MooncakeInterpreter()")
end

MooncakeInterpreter() = MooncakeInterpreter(DefaultCtx)

context_type(::MooncakeInterpreter{C}) where {C} = C

CC.InferenceParams(interp::MooncakeInterpreter) = interp.inf_params
CC.OptimizationParams(interp::MooncakeInterpreter) = interp.opt_params
CC.get_inference_cache(interp::MooncakeInterpreter) = interp.inf_cache
function CC.code_cache(interp::MooncakeInterpreter)
    return CC.WorldView(interp.code_cache, CC.WorldRange(interp.world))
end
function CC.get(wvc::CC.WorldView{MooncakeCache}, mi::Core.MethodInstance, default)
    return get(wvc.cache.dict, mi, default)
end
function CC.getindex(wvc::CC.WorldView{MooncakeCache}, mi::Core.MethodInstance)
    return getindex(wvc.cache.dict, mi)
end
function CC.haskey(wvc::CC.WorldView{MooncakeCache}, mi::Core.MethodInstance)
    return haskey(wvc.cache.dict, mi)
end
function CC.setindex!(
    wvc::CC.WorldView{MooncakeCache}, ci::Core.CodeInstance, mi::Core.MethodInstance
)
    return setindex!(wvc.cache.dict, ci, mi)
end
function CC.method_table(interp::MooncakeInterpreter)
    return CC.OverlayMethodTable(interp.world, mooncake_method_table)
end

@static if VERSION < v"1.11.0"
    CC.get_world_counter(interp::MooncakeInterpreter) = interp.world
    get_inference_world(interp::CC.AbstractInterpreter) = CC.get_world_counter(interp)
else
    CC.get_inference_world(interp::MooncakeInterpreter) = interp.world
    CC.cache_owner(::MooncakeInterpreter) = nothing
    get_inference_world(interp::CC.AbstractInterpreter) = CC.get_inference_world(interp)
end

_type(x::Type) = x
_type(x::CC.Const) = _typeof(x.val)
_type(x::CC.PartialStruct) = x.typ
_type(x::CC.Conditional) = Union{_type(x.thentype),_type(x.elsetype)}
_type(::CC.PartialTypeVar) = TypeVar

struct NoInlineCallInfo <: CC.CallInfo
    info::CC.CallInfo # wrapped call
    tt::Any # signature
end

CC.nsplit_impl(info::NoInlineCallInfo) = CC.nsplit(info.info)
CC.getsplit_impl(info::NoInlineCallInfo, idx::Int) = CC.getsplit(info.info, idx)
CC.getresult_impl(info::NoInlineCallInfo, idx::Int) = CC.getresult(info.info, idx)

function Core.Compiler.abstract_call_gf_by_type(
    interp::MooncakeInterpreter{C},
    @nospecialize(f),
    arginfo::CC.ArgInfo,
    si::CC.StmtInfo,
    @nospecialize(atype),
    sv::CC.AbsIntState,
    max_methods::Int,
) where {C}
    ret = @invoke CC.abstract_call_gf_by_type(
        interp::CC.AbstractInterpreter,
        f::Any,
        arginfo::CC.ArgInfo,
        si::CC.StmtInfo,
        atype::Any,
        sv::CC.AbsIntState,
        max_methods::Int,
    )
    callinfo = ret.info
    if !interp.inline_primitives && Mooncake.is_primitive(C, atype)
        callinfo = NoInlineCallInfo(callinfo, atype)
    end
    rt = ret.rt
    effects = ret.effects
    a = arginfo.argtypes
    if length(a) == 2 && a[1] == Core.Const(tangent_type) && a[2] isa Core.Const
        rt = Core.Const(tangent_type(a[2].val))
        effects = CC.EFFECTS_TOTAL
    end
    @static if VERSION ≥ v"1.11-"
        return CC.CallMeta(rt, ret.exct, effects, callinfo)
    else
        return CC.CallMeta(rt, effects, callinfo)
    end
end

@static if VERSION < v"1.11-"
    function CC.inlining_policy(
        interp::MooncakeInterpreter{C},
        @nospecialize(src),
        @nospecialize(info::CC.CallInfo),
        stmt_flag::UInt8,
        mi::Core.MethodInstance,
        argtypes::Vector{Any},
    ) where {C}

        # Do not inline away primitives.
        info isa NoInlineCallInfo && return nothing

        # If not a primitive, AD doesn't care about it. Use the usual inlining strategy.
        return @invoke CC.inlining_policy(
            interp::CC.AbstractInterpreter,
            src::Any,
            info::CC.CallInfo,
            stmt_flag::UInt8,
            mi::Core.MethodInstance,
            argtypes::Vector{Any},
        )
    end

else # 1.11 and up.
    function CC.inlining_policy(
        interp::MooncakeInterpreter,
        @nospecialize(src),
        @nospecialize(info::CC.CallInfo),
        stmt_flag::UInt32,
    )
        # Do not inline away primitives.
        info isa NoInlineCallInfo && return nothing

        # If not a primitive, AD doesn't care about it. Use the usual inlining strategy.
        return @invoke CC.inlining_policy(
            interp::CC.AbstractInterpreter, src::Any, info::CC.CallInfo, stmt_flag::UInt32
        )
    end
end

"""
    const GLOBAL_INTERPRETER

Globally cached interpreter. Should only be accessed via `get_interpreter`.
"""
const GLOBAL_INTERPRETER = Ref(bootstrap!(MooncakeInterpreter()))

"""
    const GLOBAL_INLINING_INTERPRETER

Globally cached interpreter which inline away AD primitives.
"""
const GLOBAL_INLINING_INTERPRETER = Ref(bootstrap!(MooncakeInterpreter(DefaultCtx; inline_primitives=true)))

"""
    get_interpreter()

Returns a `MooncakeInterpreter` appropriate for the current world age. Will use a cached
interpreter if one already exists for the current world age, otherwise creates a new one.

This should be prefered over constructing a `MooncakeInterpreter` directly.
"""
function get_interpreter(; inline_primitives=false)
    if inline_primitives
        if GLOBAL_INLINING_INTERPRETER[].world != Base.get_world_counter()
            interp = bootstrap!(MooncakeInterpreter(DefaultCtx; inline_primitives))
            GLOBAL_INLINING_INTERPRETER[] = interp
        end
        return GLOBAL_INLINING_INTERPRETER[]
    else
        if GLOBAL_INTERPRETER[].world != Base.get_world_counter()
            GLOBAL_INTERPRETER[] = bootstrap!(MooncakeInterpreter())
        end
        return GLOBAL_INTERPRETER[]
    end
end
