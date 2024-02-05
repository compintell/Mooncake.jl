# AbstractInterpretation -- this is an instance of a Julia AbstractInterpreter. We use it
# in conjunction with the contexts above to decide what should be inlined and what should
# not be inlined. Similar strategies are employed by Enzyme and Diffractor.

# The most important bit of this code is `inlining_policy` -- the rest is copy + pasted
# boiler plate, largely taken from https://github.com/JuliaLang/julia/blob/2fe4190b3d26b4eee52b2b1b1054ddd6e38a941e/test/compiler/newinterp.jl#L11

struct TICache
    dict::IdDict{Core.MethodInstance, Core.CodeInstance}
end

TICache() = TICache(IdDict{Core.MethodInstance, Core.CodeInstance}())

struct TapedInterpreter{C} <: CC.AbstractInterpreter
    ctx::C
    meta # additional information
    world::UInt
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
    inf_cache::Vector{CC.InferenceResult}
    code_cache::TICache
    in_f_cache::Dict{Any, Any}
    in_f_rrule_cache::Dict{Any, Any}
    function TapedInterpreter(
        ctx::C=DefaultCtx();
        meta=nothing,
        world::UInt=Base.get_world_counter(),
        inf_params::CC.InferenceParams=CC.InferenceParams(),
        opt_params::CC.OptimizationParams=CC.OptimizationParams(),
        inf_cache::Vector{CC.InferenceResult}=CC.InferenceResult[], 
        code_cache::TICache=TICache(),
        in_f_cache::Dict{Any, Any}=Dict(),
        in_f_rrule_cache::Dict{Any, Any}=Dict(),
    ) where {C}
        return new{C}(
            ctx, meta, world, inf_params, opt_params, inf_cache, code_cache, in_f_cache, in_f_rrule_cache
        )
    end
end

const TInterp = TapedInterpreter

CC.InferenceParams(interp::TInterp) = interp.inf_params
CC.OptimizationParams(interp::TInterp) = interp.opt_params
CC.get_world_counter(interp::TInterp) = interp.world
CC.get_inference_cache(interp::TInterp) = interp.inf_cache
function CC.code_cache(interp::TInterp)
    return CC.WorldView(interp.code_cache, CC.WorldRange(interp.world))
end
function CC.get(wvc::CC.WorldView{TICache}, mi::Core.MethodInstance, default)
    return get(wvc.cache.dict, mi, default)
end
function CC.getindex(wvc::CC.WorldView{TICache}, mi::Core.MethodInstance)
    return getindex(wvc.cache.dict, mi)
end
CC.haskey(wvc::CC.WorldView{TICache}, mi::Core.MethodInstance) = haskey(wvc.cache.dict, mi)
function CC.setindex!(
    wvc::CC.WorldView{TICache}, ci::Core.CodeInstance, mi::Core.MethodInstance
)
    return setindex!(wvc.cache.dict, ci, mi)
end

_type(x) = x
_type(x::CC.Const) = Core.Typeof(x.val)
_type(x::CC.PartialStruct) = x.typ
_type(x::CC.Conditional) = Union{x.thentype, x.elsetype}

function CC.inlining_policy(
    interp::TapedInterpreter{C},
    @nospecialize(src),
    @nospecialize(info::CC.CallInfo),
    stmt_flag::UInt8,
    mi::Core.MethodInstance,
    argtypes::Vector{Any},
) where {C}

    # Do not inline away primitives.
    argtype_tuple = Tuple{map(_type, argtypes)...}
    is_primitive(C, argtype_tuple) && return nothing

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
