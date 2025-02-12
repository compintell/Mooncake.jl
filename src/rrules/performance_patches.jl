# All of the code here purely exists to work around current performance limitations of
# Mooncake.jl. In order to prevent this from getting out of hand, there are several
# conventions to which we adhere when writing these rules:
#
# 1. for each rule, a comment is added containing a link to the issue or issues that are
#   believed to describe the deficiencies of Mooncake.jl which cause the rule to be needed.
# 2. the number of concrete types for which the signature is valid is finite, and all are
#   tested. For example, `Array{<:IEEEFloat}` is a permissible type. The only exception to
#   this is the dimension of an `Array` argument. For example, it is fine to write rules for
#   `Array{Float64}`, despite the fact that this technically includes `Array{Float64,1}`,
#   `Array{Float64,2}`, `Array{Float64,3}`, etc.
#   `Diagonal{<:IEEEFloat}` is not, on the other hand, permissible. This is because we do
#   not know what the type of its `diag` field is, and it _could_ be any `AbstractVector`.
#   Something more precise like `Diagonal{P, Vector{P}} where {P<:IEEEFloat}` is fine.
#   This convention ensures that we are confident the rules here confident a strict
#   improvement over what we currently have, and prevents the addition of flakey rules which
#   cause robustness or correctness problems.

# Performance issue: https://github.com/compintell/Mooncake.jl/issues/156
# Complicated implementation involving low-level machinery needed due to
# https://github.com/compintell/Mooncake.jl/issues/238
# @is_primitive(
#     DefaultCtx,
#     Tuple{
#         typeof(Base._mapreduce),
#         typeof(identity),
#         typeof(Base.add_sum),
#         Base.IndexLinear,
#         Array{<:IEEEFloat},
#     },
# )
# function rrule!!(
#     ::CoDual{typeof(Base._mapreduce)},
#     ::CoDual{typeof(identity)},
#     ::CoDual{typeof(Base.add_sum)},
#     ::CoDual{Base.IndexLinear},
#     x::CoDual{<:Array{P}}
# ) where {P<:IEEEFloat}
#     dx = x.dx
#     function sum_pb!!(dz::P)
#         dx .+= dz
#         return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
#     end
#     return zero_fcodual(sum(x.x)), sum_pb!!
# end

@is_primitive(DefaultCtx, Tuple{typeof(sum),Array{<:IEEEFloat}})
function rrule!!(::CoDual{typeof(sum)}, x::CoDual{<:Array{P}}) where {P<:IEEEFloat}
    dx = x.dx
    function sum_pb!!(dz::P)
        dx .+= dz
        return NoRData(), NoRData()
    end
    return zero_fcodual(sum(x.x)), sum_pb!!
end

@is_primitive(DefaultCtx, Tuple{typeof(sum),typeof(abs2),Array{<:IEEEFloat}})
function rrule!!(
    ::CoDual{typeof(sum)}, ::CoDual{typeof(abs2)}, x::CoDual{<:Array{P}}
) where {P<:IEEEFloat}
    function sum_abs2_pb!!(dz::P)
        x.dx .+= 2 .* x.x .* dz
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(sum(abs2, x.x)), sum_abs2_pb!!
end

# # Performance issue: https://github.com/compintell/Mooncake.jl/issues/156
# # Complicated implementation involving low-level machinery needed due to
# # https://github.com/compintell/Mooncake.jl/issues/238
# @is_primitive(
#     DefaultCtx,
#     Tuple{
#         typeof(Base._mapreduce),
#         typeof(abs2),
#         typeof(Base.add_sum),
#         Base.IndexLinear,
#         Array{<:IEEEFloat},
#     },
# )
# function rrule!!(
#     ::CoDual{typeof(Base._mapreduce)},
#     ::CoDual{typeof(abs2)},
#     ::CoDual{typeof(Base.add_sum)},
#     ::CoDual{Base.IndexLinear},
#     x::CoDual{<:Array{P}}
# ) where {P<:IEEEFloat}
#     dx = x.dx
#     function sum_pb!!(dz::P)
#         x.dx .+= 2 .* dz .* x.x
#         return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
#     end
#     return zero_fcodual(sum(abs2, x.x)), sum_pb!!
# end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:performance_patches})
    rng = rng_ctor(123)
    sizes = [(11,), (11, 3)]
    precisions = [Float64, Float32, Float16]
    test_cases = vcat(
        (false, :stability_and_allocs, nothing, sum, randn(Float32, 10)),
        (false, :stability_and_allocs, nothing, sum, randn(Float64, 10)),
        (false, :stability_and_allocs, nothing, sum, randn(Float32, 10, 10)),
        (false, :stability_and_allocs, nothing, sum, randn(Float64, 10, 10)),
        (false, :stability_and_allocs, nothing, sum, abs2, randn(Float32, 10)),
        (false, :stability_and_allocs, nothing, sum, abs2, randn(Float64, 10)),
        (false, :stability_and_allocs, nothing, sum, abs2, randn(Float32, 10, 10)),
        (false, :stability_and_allocs, nothing, sum, abs2, randn(Float64, 10, 10)),

        # # sum(x), sum(abs2, x)
        # map_prod(sizes, precisions, [identity, abs2]) do (sz, P, f)
        #     flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
        #     x = randn(rng, P, sz...)
        #     args = (Base._mapreduce, f, Base.add_sum, Base.IndexLinear(), x)
        #     return (flags..., args...)
        # end,
    )
    memory = Any[]
    return test_cases, memory
end

generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:performance_patches}) = Any[], Any[]
