# The purpose of the code in this file is to provide a work-around for the Julia compiler
# bug discussed in https://github.com/compintell/Mooncake.jl/issues/319 . You do not need to
# understand it in order to understand Mooncake. I (Will) would recommend against spending
# any time at all reading / understanding this file unless you are actively working on this
# issue, and find it useful.
#
# As soon as patch versions of Julia (both LTS and Release) are made which contain a fix for
# the bug underlying the above issue, this code can and should be removed, and the minimum
# versions of Julia which Mooncake supports bumped.
#
# The only place in which this code seeps into Mooncake.jl code is in Mooncake.optimise_ir!,
# located in src/interpreter/ir_utils.jl . In particular, we replace the `local_interp`
# variable with the `AbstractInterpreter`.
#
# The work around:
# We define a new `AbstractInterpreter` which wraps around the `Compiler.NativeInterpreter`.
# This makes it possible to add methods to various functions in `Compiler`, thereby enabling
# us to insert the bug fixes.

struct BugPatchInterpreter <: CC.AbstractInterpreter
    interp::CC.NativeInterpreter
    BugPatchInterpreter() = new(CC.NativeInterpreter())
end

CC.InferenceParams(ip::BugPatchInterpreter) = CC.InferenceParams(ip.interp)
CC.OptimizationParams(ip::BugPatchInterpreter) = CC.OptimizationParams(ip.interp)
CC.get_inference_cache(ip::BugPatchInterpreter) = CC.get_inference_cache(ip.interp)
CC.code_cache(ip::BugPatchInterpreter) = CC.code_cache(ip.interp)
function CC.get(wvc::CC.WorldView{BugPatchInterpreter}, mi::Core.MethodInstance, default)
    return get(wvc.cache.dict, mi, default)
end
function CC.getindex(wvc::CC.WorldView{BugPatchInterpreter}, mi::Core.MethodInstance)
    return getindex(wvc.cache.dict, mi)
end
function CC.haskey(wvc::CC.WorldView{BugPatchInterpreter}, mi::Core.MethodInstance)
    return haskey(wvc.cache.dict, mi)
end
function CC.setindex!(
    wvc::CC.WorldView{BugPatchInterpreter}, ci::Core.CodeInstance, mi::Core.MethodInstance
)
    return setindex!(wvc.cache.dict, ci, mi)
end
CC.method_table(ip::BugPatchInterpreter) = CC.method_table(ip.interp)

@static if VERSION < v"1.11.0"
    CC.get_world_counter(ip::BugPatchInterpreter) = CC.get_world_counter(ip.interp)
    get_inference_world(ip::CC.AbstractInterpreter) = CC.get_world_counter(ip)
else
    CC.get_inference_world(ip::BugPatchInterpreter) = CC.get_inference_world(ip.interp)
    CC.cache_owner(ip::BugPatchInterpreter) = ip.interp
    get_inference_world(interp::CC.AbstractInterpreter) = CC.get_inference_world(interp)
end

