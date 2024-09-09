# AbstractInterpretation -- this is an instance of a Julia AbstractInterpreter. We use it
# in conjunction with the contexts above to decide what should be inlined and what should
# not be inlined. Similar strategies are employed by Enzyme and Diffractor.

# The most important bit of this code is `inlining_policy` -- the rest is copy + pasted
# boiler plate, largely taken from https://github.com/JuliaLang/julia/blob/2fe4190b3d26b4eee52b2b1b1054ddd6e38a941e/test/compiler/newinterp.jl#L11


struct ClosureCacheKey
    world_age::UInt
    key::Any
end

struct TICache
    dict::IdDict{Core.MethodInstance, Core.CodeInstance}
end

TICache() = TICache(IdDict{Core.MethodInstance, Core.CodeInstance}())

struct TapirInterpreter{C} <: CC.AbstractInterpreter
    meta # additional information
    world::UInt
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
    inf_cache::Vector{CC.InferenceResult}
    code_cache::TICache
    oc_cache::Dict{ClosureCacheKey, Any}
    function TapirInterpreter(
        ::Type{C};
        meta=nothing,
        world::UInt=Base.get_world_counter(),
        inf_params::CC.InferenceParams=CC.InferenceParams(),
        opt_params::CC.OptimizationParams=CC.OptimizationParams(),
        inf_cache::Vector{CC.InferenceResult}=CC.InferenceResult[], 
        code_cache::TICache=TICache(),
        oc_cache::Dict{ClosureCacheKey, Any}=Dict{ClosureCacheKey, Any}(),
    ) where {C}
        return new{C}(meta, world, inf_params, opt_params, inf_cache, code_cache, oc_cache)
    end
end

# Don't print out the IRCode object, because this tends to pollute the REPL. Just make it
# clear that this is a MistyClosure, which contains an OpaqueClosure.
Base.show(io::IO, mime::MIME"text/plain", mc::TapirInterpreter) = _show_tapir_interp(io, mime, mc)
Base.show(io::IO, mc::TapirInterpreter) = _show_tapir_interp(io, MIME"text/plain"(), mc)

function _show_tapir_interp(io::IO, mime::MIME"text/plain", mc::TapirInterpreter)
    print(io, "TapirInterpreter()")
end

TapirInterpreter() = TapirInterpreter(DefaultCtx)

# Globally cached interpreter. Should only be accessed via `get_tapir_interpreter`.
const GLOBAL_INTERPRETER = Ref(TapirInterpreter())

"""
    get_tapir_interpreter()

Returns a `TapirInterpreter` appropriate for the current world age. Will use a cached
interpreter if one already exists for the current world age, otherwise creates a new one.
This is a very conservative approach to caching the interpreter, which reflects the
approach taken the the closure cache.
"""
function get_tapir_interpreter()
    if GLOBAL_INTERPRETER[].world != Base.get_world_counter()
        @info "Refreshing the global interpreter"
        GLOBAL_INTERPRETER[] = TapirInterpreter()
    end
    return GLOBAL_INTERPRETER[]
end

CC.InferenceParams(interp::TapirInterpreter) = interp.inf_params
CC.OptimizationParams(interp::TapirInterpreter) = interp.opt_params
CC.get_world_counter(interp::TapirInterpreter) = interp.world
CC.get_inference_cache(interp::TapirInterpreter) = interp.inf_cache
function CC.code_cache(interp::TapirInterpreter)
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
_type(x::CC.Const) = _typeof(x.val)
_type(x::CC.PartialStruct) = x.typ
_type(x::CC.Conditional) = Union{_type(x.thentype), _type(x.elsetype)}

function CC.inlining_policy(
    interp::TapirInterpreter{C},
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

context_type(::TapirInterpreter{C}) where {C} = C
