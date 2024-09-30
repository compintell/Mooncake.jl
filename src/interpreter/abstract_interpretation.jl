# AbstractInterpretation -- this is an instance of a Julia AbstractInterpreter. We use it
# in conjunction with the contexts above to decide what should be inlined and what should
# not be inlined. Similar strategies are employed by Enzyme and Diffractor.

# The most important bit of this code is `inlining_policy` -- the rest is copy + pasted
# boiler plate, largely taken from https://github.com/JuliaLang/julia/blob/2fe4190b3d26b4eee52b2b1b1054ddd6e38a941e/test/compiler/newinterp.jl#L11

struct ClosureCacheKey
    world_age::UInt
    key::Any
end

struct MooncakeCache
    dict::IdDict{Core.MethodInstance, Core.CodeInstance}
end

MooncakeCache() = MooncakeCache(IdDict{Core.MethodInstance, Core.CodeInstance}())

Base.Experimental.@MethodTable mooncake_method_table

struct MooncakeInterpreter{C} <: CC.AbstractInterpreter
    meta # additional information
    world::UInt
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
    inf_cache::Vector{CC.InferenceResult}
    code_cache::MooncakeCache
    oc_cache::Dict{ClosureCacheKey, Any}
    method_table_to_overlay::CC.MethodTable
    function MooncakeInterpreter(
        ::Type{C};
        meta=nothing,
        world::UInt=Base.get_world_counter(),
        inf_params::CC.InferenceParams=CC.InferenceParams(),
        opt_params::CC.OptimizationParams=CC.OptimizationParams(),
        inf_cache::Vector{CC.InferenceResult}=CC.InferenceResult[], 
        code_cache::MooncakeCache=MooncakeCache(),
        oc_cache::Dict{ClosureCacheKey, Any}=Dict{ClosureCacheKey, Any}(),
        method_table_to_overlay::CC.MethodTable=mooncake_method_table,
    ) where {C}
        return new{C}(
            meta,
            world,
            inf_params,
            opt_params,
            inf_cache,
            code_cache,
            oc_cache,
            method_table_to_overlay,
        )
    end
end

# Don't print out the IRCode object, because this tends to pollute the REPL. Just make it
# clear that this is a MistyClosure, which contains an OpaqueClosure.
function Base.show(io::IO, mime::MIME"text/plain", mc::MooncakeInterpreter)
    return _show_interp(io, mime, mc)
end
Base.show(io::IO, mc::MooncakeInterpreter) = _show_interp(io, MIME"text/plain"(), mc)

function _show_interp(io::IO, ::MIME"text/plain", ::MooncakeInterpreter)
    print(io, "MooncakeInterpreter()")
end

MooncakeInterpreter() = MooncakeInterpreter(DefaultCtx)

# Globally cached interpreter. Should only be accessed via `get_interpreter`.
const GLOBAL_INTERPRETER = Ref(MooncakeInterpreter())

"""
    get_interpreter()

Returns a `MooncakeInterpreter` appropriate for the current world age. Will use a cached
interpreter if one already exists for the current world age, otherwise creates a new one.

This should be prefered over constructing a `MooncakeInterpreter` directly.
"""
function get_interpreter()
    if GLOBAL_INTERPRETER[].world != Base.get_world_counter()
        GLOBAL_INTERPRETER[] = MooncakeInterpreter()
    end
    return GLOBAL_INTERPRETER[]
end

CC.InferenceParams(interp::MooncakeInterpreter) = interp.inf_params
CC.OptimizationParams(interp::MooncakeInterpreter) = interp.opt_params
CC.get_world_counter(interp::MooncakeInterpreter) = interp.world
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
    return CC.OverlayMethodTable(interp.world, interp.method_table_to_overlay)
end

_type(x) = x
_type(x::CC.Const) = _typeof(x.val)
_type(x::CC.PartialStruct) = x.typ
_type(x::CC.Conditional) = Union{_type(x.thentype), _type(x.elsetype)}

function CC.inlining_policy(
    interp::MooncakeInterpreter{C},
    @nospecialize(src),
    @nospecialize(info::CC.CallInfo),
    stmt_flag::UInt8,
    mi::Core.MethodInstance,
    argtypes::Vector{Any},
) where {C}

    # Do not inline away primitives.
    argtype_tuple = Tuple{map(_type, argtypes)...}
    if is_primitive(C, argtype_tuple)
        return nothing
    end

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

context_type(::MooncakeInterpreter{C}) where {C} = C
