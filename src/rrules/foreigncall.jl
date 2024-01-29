# Fallback rule for foreigncall which gives an interpretable error message.
struct MissingForeigncallRuleError <: Exception
    msg::String
end

Base.showerror(io::IO, err::MissingForeigncallRuleError) = print(io, err.msg)

# Fallback foreigncall rrule. This is a sufficiently common special case, that it's worth
# creating an informative error message, so that users have some chance of knowing why
# they're not able to differentiate a piece of code.
function rrule!!(::CoDual{<:Tforeigncall}, args...)
    throw(MissingForeigncallRuleError(
        "No rrule!! available for foreigncall with primal argument types " *
        "$(typeof(map(primal, args))). " *
        "This problem has most likely arisen because there is a ccall somewhere in the " *
        "function you are trying to differentiate, for which an rrule!! has not been " *
        "explicitly written." *
        "You have three options: write an rrule!! for this foreigncall, write an rrule!! " *
        "for a Julia function that calls this foreigncall, or re-write your code to " *
        "avoid this foreigncall entirely. " *
        "If you believe that this error has arisen for some other reason than the above, " *
        "or the above does not help you to workaround this problem, please open an issue."
    ))
end

_get_arg_type(::Type{Val{T}}) where {T} = T

"""
    function _foreigncall_(
        ::Val{name}, ::Val{RT}, AT::Tuple, ::Val{nreq}, ::Val{calling_convention}, x...
    ) where {name, RT, nreq, calling_convention}

:foreigncall nodes get translated into calls to this function.
For example,
```julia
Expr(:foreigncall, :foo, Tout, (A, B), nreq, :ccall, args...)
```
becomes
```julia
_foreigncall_(Val(:foo), Val(Tout), (Val(A), Val(B)), Val(nreq), Val(:ccall), args...)
```
Please consult the Julia documentation for more information on how foreigncall nodes work,
and consult this package's tests for examples.

Credit: Umlaut.jl has the original implementation of this function. This is largely copied
over from there.
"""
@generated function _foreigncall_(
    ::Val{name}, ::Val{RT}, AT::Tuple, ::Val{nreq}, ::Val{calling_convention}, x::Vararg{Any, N}
) where {name, RT, nreq, calling_convention, N}
    return Expr(
        :foreigncall,
        QuoteNode(name),
        :($(RT)),
        Expr(:call, :(Core.svec), map(_get_arg_type, AT.parameters)...),
        :($nreq),
        QuoteNode(calling_convention),
        map(n -> :(x[$n]), 1:length(x))...,
    )
end

@generated function _eval(::typeof(_foreigncall_), x::Vararg{Any, N}) where {N}
    return Expr(:call, :_foreigncall_, map(n -> :(getfield(x, $n)), 1:N)...)
end

@is_primitive MinimalCtx Tuple{typeof(_foreigncall_), Vararg}

#
# Rules to handle / avoid foreigncall nodes
#

@is_primitive MinimalCtx Tuple{typeof(Base.allocatedinline), Type}
function rrule!!(::CoDual{typeof(Base.allocatedinline)}, T::CoDual{<:Type})
    return CoDual(Base.allocatedinline(primal(T)), NoTangent()), NoPullback()
end

@is_primitive MinimalCtx Tuple{Type{<:Array{T, N}}, typeof(undef), Vararg} where {T, N}
function rrule!!(
    ::CoDual{Type{Array{T, N}}}, ::CoDual{typeof(undef)}, m::Vararg{CoDual}
) where {T, N}
    _m = map(primal, m)
    p = Array{T, N}(undef, _m...)
    t = Array{tangent_type(T), N}(undef, _m...)
    if isassigned(t, 1)
        for n in eachindex(t)
            @inbounds t[n] = zero_tangent(p[n])
        end
    end
    return CoDual(p, t), NoPullback()
end

@is_primitive MinimalCtx Tuple{Type{<:Array{T, N}}, typeof(undef), NTuple{N}} where {T, N}
function rrule!!(
    ::CoDual{<:Type{<:Array{T, N}}}, ::CoDual{typeof(undef)}, m::CoDual{NTuple{N}},
) where {T, N}
    _m = primal(m)
    p = Array{T, N}(undef, _m)
    t = Array{tangent_type(T), N}(undef, _m)
    if isassigned(t, 1)
        for n in eachindex(t)
            @inbounds t[n] = zero_tangent(p[n])
        end
    end
    return CoDual(p, t), NoPullback()
end

@is_primitive MinimalCtx Tuple{typeof(copy), Array}
function rrule!!(::CoDual{typeof(copy)}, a::CoDual{<:Array})
    y = CoDual(copy(primal(a)), copy(tangent(a)))
    copy_pullback!!(dy, df, dx) = df, increment!!(dx, dy)
    return y, copy_pullback!!
end

@is_primitive MinimalCtx Tuple{typeof(fill!), Array{<:Union{UInt8, Int8}}, Integer}
function rrule!!(
    ::CoDual{typeof(fill!)},
    a::CoDual{<:Union{Array{UInt8}, Array{Int8}}, <:Array{NoTangent}},
    x::CoDual{<:Integer},
)
    old_value = copy(primal(a))
    fill!(primal(a), primal(x))
    function fill!_pullback!!(dy, df, da, dx)
        primal(a) .= old_value
        return df, da, dx
    end
    return a, fill!_pullback!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._growbeg!), Vector, Integer}
function rrule!!(
    ::CoDual{typeof(Base._growbeg!)}, _a::CoDual{<:Vector{T}}, _delta::CoDual{<:Integer},
) where {T}
    d = primal(_delta)
    a = primal(_a)
    Base._growbeg!(a, d)
    Base._growbeg!(tangent(_a), d)
    function _growbeg!_pb!!(_, df, da, ddelta)
        Base._deletebeg!(a, d)
        Base._deletebeg!(da, d)
        return df, da, ddelta
    end
    return zero_codual(nothing), _growbeg!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._growend!), Vector, Integer}
function rrule!!(
    ::CoDual{typeof(Base._growend!)}, _a::CoDual{<:Vector}, _delta::CoDual{<:Integer},
)
    d = primal(_delta)
    a = primal(_a)
    Base._growend!(a, d)
    Base._growend!(tangent(_a), d)
    function _growend!_pullback!!(dy, df, da, ddelta)
        Base._deleteend!(a, d)
        Base._deleteend!(da, d)
        return df, da, ddelta
    end
    return zero_codual(nothing), _growend!_pullback!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._growat!), Vector, Integer, Integer}
function rrule!!(
    ::CoDual{typeof(Base._growat!)},
    _a::CoDual{<:Vector},
    _i::CoDual{<:Integer},
    _delta::CoDual{<:Integer},
)
    # Extract data.
    a, i, delta = map(primal, (_a, _i, _delta))

    # Run the primal.
    Base._growat!(a, i, delta)
    Base._growat!(tangent(_a), i, delta)

    function _growat!_pb!!(_, df, da, di, ddelta)
        deleteat!(a, i:i+delta-1)
        deleteat!(da, i:i+delta-1)
        return df, da, di, ddelta
    end
    return zero_codual(nothing), _growat!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._deletebeg!), Vector, Integer}
function rrule!!(
    ::CoDual{typeof(Base._deletebeg!)}, _a::CoDual{<:Vector}, _delta::CoDual{<:Integer},
)
    delta = primal(_delta)
    a = primal(_a)

    a_beg = a[1:delta]
    da_beg = tangent(_a)[1:delta]

    Base._deletebeg!(a, delta)
    Base._deletebeg!(tangent(_a), delta)

    function _deletebeg!_pb!!(_, df, da, ddelta)
        splice!(a, 1:0, a_beg)
        splice!(da, 1:0, da_beg)
        return df, da, ddelta
    end
    return zero_codual(nothing), _deletebeg!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._deleteend!), Vector, Integer}
function rrule!!(
    ::CoDual{typeof(Base._deleteend!)}, _a::CoDual{<:Vector}, _delta::CoDual{<:Integer}
)
    # Extract data.
    a = primal(_a)
    delta = primal(_delta)

    # Store the section to be cut for later.
    primal_tail = a[end-delta+1:end]
    tangent_tail = tangent(_a)[end-delta+1:end]

    # Cut the end off the primal and tangent.
    Base._deleteend!(a, delta)
    Base._deleteend!(tangent(_a), delta)

    function _deleteend!_pb!!(_, df, da, ddelta)

        Base._growend!(a, delta)
        a[end-delta+1:end] .= primal_tail

        Base._growend!(da, delta)
        da[end-delta+1:end] .= tangent_tail

        return df, da, ddelta
    end
    return zero_codual(nothing), _deleteend!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._deleteat!), Vector, Integer, Integer}
function rrule!!(
    ::CoDual{typeof(Base._deleteat!)},
    _a::CoDual{<:Vector},
    _i::CoDual{<:Integer},
    _delta::CoDual{<:Integer},
)
    # Extract data.
    a, i, delta = map(primal, (_a, _i, _delta))

    # Store the cut section for later.
    primal_mem = a[i:i+delta-1]
    tangent_mem = tangent(_a)[i:i+delta-1]

    # Run the primal.
    Base._deleteat!(a, i, delta)
    Base._deleteat!(tangent(_a), i, delta)

    function _deleteat!_pb!!(_, df, da, di, ddelta)
        splice!(a, i:i-1, primal_mem)
        splice!(da, i:i-1, tangent_mem)
        return df, da, di, ddelta
    end

    return zero_codual(nothing), _deleteat!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(sizehint!), Vector, Integer}
function rrule!!(::CoDual{typeof(sizehint!)}, x::CoDual{<:Vector}, sz::CoDual{<:Integer})
    sizehint!(primal(x), primal(sz))
    sizehint!(tangent(x), primal(sz))
    return x, NoPullback()
end

@is_primitive MinimalCtx Tuple{typeof(objectid), Any}
function rrule!!(::CoDual{typeof(objectid)}, @nospecialize(x))
    return CoDual(objectid(primal(x)), NoTangent()), NoPullback()
end

@is_primitive MinimalCtx Tuple{typeof(pointer_from_objref), Any}
function rrule!!(::CoDual{typeof(pointer_from_objref)}, x)
    y = CoDual(
        pointer_from_objref(primal(x)),
        bitcast(Ptr{tangent_type(Nothing)}, pointer_from_objref(tangent(x))),
    )
    return y, NoPullback()
end

@is_primitive MinimalCtx Tuple{typeof(CC.return_type), Vararg}
function rrule!!(::CoDual{typeof(Core.Compiler.return_type)}, args...)
    return zero_codual(Core.Compiler.return_type(map(primal, args)...)), NoPullback()
end

# unsafe_copyto! is the only function in Julia that appears to rely on a ccall to `memmove`.
# Since we can't differentiate `memmove` (due to a lack of type information), it is
# necessary to work with `unsafe_copyto!` instead.
@is_primitive MinimalCtx Tuple{typeof(unsafe_copyto!), Ptr{T}, Ptr{T}, Any} where {T}
function rrule!!(
    ::CoDual{typeof(unsafe_copyto!)}, dest::CoDual{Ptr{T}}, src::CoDual{Ptr{T}}, n::CoDual
) where {T}
    _n = primal(n)

    # Record values that will be overwritten.
    dest_copy = Vector{T}(undef, _n)
    ddest_copy = Vector{T}(undef, _n)
    unsafe_copyto!(pointer(dest_copy), primal(dest), _n)
    unsafe_copyto!(pointer(ddest_copy), tangent(dest), _n)

    # Run primal computation.
    unsafe_copyto!(primal(dest), primal(src), _n)
    unsafe_copyto!(tangent(dest), tangent(src), _n)

    function unsafe_copyto!_pb!!(_, df, ddest, dsrc, dn)

        # Increment dsrc.
        dsrc = _increment_pointer!(dsrc, ddest, _n)

        # Restore initial state.
        unsafe_copyto!(primal(dest), pointer(dest_copy), _n)
        unsafe_copyto!(tangent(dest), pointer(ddest_copy), _n)

        return df, ddest, dsrc, dn
    end
    return dest, unsafe_copyto!_pb!!
end

# same structure as the previous method, just without the pointers.
@is_primitive MinimalCtx Tuple{typeof(unsafe_copyto!), Array{T}, Any, Array{T}, Any, Any} where {T}
function rrule!!(
    ::CoDual{typeof(unsafe_copyto!)},
    dest::CoDual{<:Array{T}},
    doffs::CoDual,
    src::CoDual{<:Array{T}},
    soffs::CoDual,
    n::CoDual,
) where {T}
    _n = primal(n)

    # Record values that will be overwritten.
    _doffs = primal(doffs)
    dest_idx = _doffs:_doffs + _n - 1
    _soffs = primal(soffs)
    dest_copy = primal(dest)[dest_idx]
    ddest_copy = tangent(dest)[dest_idx]

    # Run primal computation.
    unsafe_copyto!(primal(dest), _doffs, primal(src), _soffs, _n)
    unsafe_copyto!(tangent(dest), _doffs, tangent(src), _soffs, _n)

    function unsafe_copyto_pb!!(_, df, ddest, ddoffs, dsrc, dsoffs, dn)

        # Increment dsrc.
        src_idx = _soffs:_soffs + _n - 1
        dsrc[src_idx] .= increment!!.(view(dsrc, src_idx), view(ddest, dest_idx))

        # Restore initial state.
        primal(dest)[dest_idx] .= dest_copy
        tangent(dest)[dest_idx] .= ddest_copy

        return df, ddest, ddoffs, dsrc, dsoffs, dn
    end

    return dest, unsafe_copyto_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base.unsafe_pointer_to_objref), Ptr}
function rrule!!(::CoDual{typeof(Base.unsafe_pointer_to_objref)}, x::CoDual{<:Ptr})
    y = CoDual(unsafe_pointer_to_objref(primal(x)), unsafe_pointer_to_objref(tangent(x)))
    return y, NoPullback()
end

@is_primitive MinimalCtx Tuple{typeof(Threads.threadid)}
rrule!!(::CoDual{typeof(Threads.threadid)}) = zero_codual(Threads.threadid()), NoPullback()

@is_primitive MinimalCtx Tuple{typeof(typeintersect), Any, Any}
function rrule!!(::CoDual{typeof(typeintersect)}, @nospecialize(a), @nospecialize(b))
    y = typeintersect(primal(a), primal(b))
    return CoDual(y, zero_tangent(y)), NoPullback()
end

function _increment_pointer!(x::Ptr{T}, y::Ptr{T}, N::Integer) where {T}
    increment!!(unsafe_wrap(Vector{T}, x, N), unsafe_wrap(Vector{T}, y, N))
    return x
end



#
# general foreigncall nodes
#

function rrule!!(
    ::CoDual{<:Tforeigncall},
    ::CoDual{Val{:jl_array_ptr}},
    ::CoDual{Val{Ptr{T}}},
    ::CoDual{Tuple{Val{Any}}},
    ::CoDual, # nreq
    ::CoDual, # calling convention
    a::CoDual{<:Array{T}, <:Array{V}}
) where {T, V}
    y = CoDual(
        ccall(:jl_array_ptr, Ptr{T}, (Any, ), primal(a)),
        ccall(:jl_array_ptr, Ptr{V}, (Any, ), tangent(a)),
    )
    return y, NoPullback()
end

# function rrule!!(
#     ::CoDual{<:Tforeigncall},
#     ::CoDual{Val{:jl_value_ptr}},
#     ::CoDual{Val{Ptr{Cvoid}}},
#     ::CoDual,
#     ::CoDual, # nreq
#     ::CoDual, # calling convention
#     a::CoDual
# )
#     y = CoDual(
#         ccall(:jl_value_ptr, Ptr{Cvoid}, (Any, ), primal(a)),
#         ccall(:jl_value_ptr, Ptr{NoTangent}, (Any, ), tangent(a)),
#     )
#     return y, NoPullback()
# end

function rrule!!(
    ::CoDual{<:Tforeigncall},
    ::CoDual{Val{:jl_reshape_array}},
    ::CoDual{Val{Array{P, M}}},
    ::CoDual{Tuple{Val{Any}, Val{Any}, Val{Any}}},
    ::CoDual, # nreq
    ::CoDual, # calling convention
    x::CoDual{Type{Array{P, M}}},
    a::CoDual{Array{P, N}, Array{T, N}},
    dims::CoDual,
) where {P, T, M, N}
    d = primal(dims)
    y = CoDual(
        ccall(:jl_reshape_array, Array{P, M}, (Any, Any, Any), Array{P, M}, primal(a), d),
        ccall(:jl_reshape_array, Array{T, M}, (Any, Any, Any), Array{T, M}, tangent(a), d),
    )
    return y, NoPullback()
end

function rrule!!(
    ::CoDual{<:Tforeigncall},
    ::CoDual{Val{:jl_array_isassigned}},
    ::CoDual, # return type is Int32
    ::CoDual, # arg types are (Any, UInt64)
    ::CoDual, # nreq
    ::CoDual, # calling convention
    a::CoDual{<:Array},
    ii::CoDual{UInt},
    args...,
)
    y = ccall(:jl_array_isassigned, Cint, (Any, UInt), primal(a), primal(ii))
    return zero_codual(y), NoPullback()
end

@is_primitive MinimalCtx Tuple{typeof(deepcopy), Any}
function rrule!!(::CoDual{typeof(deepcopy)}, x::CoDual)
    deepcopy_pb!!(dy, df, dx) = df, increment!!(dx, dy)
    return deepcopy(x), deepcopy_pb!!
end

@is_primitive MinimalCtx Tuple{Type{UnionAll}, TypeVar, Any}
@is_primitive MinimalCtx Tuple{Type{UnionAll}, TypeVar, Type}
function rrule!!(::CoDual{<:Type{UnionAll}}, x::CoDual{<:TypeVar}, y::CoDual{<:Type})
    return CoDual(UnionAll(primal(x), primal(y)), NoTangent()), NoPullback()
end

function rrule!!(
    ::CoDual{typeof(_foreigncall_)}, ::CoDual{Val{:jl_string_ptr}}, args::Vararg{CoDual, N}
) where {N}
    x = tuple_map(primal, args)
    return uninit_codual(_foreigncall_(Val(:jl_string_ptr), x...)), NoPullback()
end

@is_primitive MinimalCtx Tuple{typeof(hash), Union{String, SubString{String}}, UInt}
function rrule!!(
    ::CoDual{typeof(hash)}, s::CoDual{P}, h::CoDual{UInt}
) where {P<:Union{String, SubString{String}}}
    return zero_codual(hash(primal(s), primal(h))), NoPullback()
end

function unexepcted_foreigncall_error(name)
    throw(error(
        "AD has hit a :($name) ccall. This should not happen. " *
        "Please open an issue with a minimal working example in order to reproduce. ",
        "This is true unless you have intentionally written a ccall to :$(name), ",
        "in which case you must write a :foreigncall rule. It may not be possible ",
        "to implement a :foreigncall rule if too much type information has been lost ",
        "in which case your only recourse is to write a rule for whichever Julia ",
        "function calls this one (and retains enough type information).",
    ))
end

for name in [
    :(:jl_alloc_array_1d), :(:jl_alloc_array_2d), :(:jl_alloc_array_3d), :(:jl_new_array),
    :(:jl_array_grow_end), :(:jl_array_del_end), :(:jl_array_copy), :(:jl_object_id),
    :(:jl_type_intersection), :(:memset), :(:jl_get_tls_world_age), :(:memmove),
    :(:jl_array_sizehint), :(:jl_array_del_at), :(:jl_array_grow_at), :(:jl_array_del_beg),
    :(:jl_array_grow_beg), :(:jl_value_ptr), :(:jl_type_unionall), :(:jl_threadid),
    :(:memhash_seed), :(:memhash32_seed),
]
    @eval function _foreigncall_(
        ::Val{$name}, ::Val{RT}, AT::Tuple, ::Val{nreq}, ::Val{calling_convention}, x...,
    ) where {RT, nreq, calling_convention}
        unexepcted_foreigncall_error($name)
    end
    @eval function rrule!!(::CoDual{<:Tforeigncall}, ::CoDual{Val{$name}}, args...)
        unexepcted_foreigncall_error($name)
    end
end


function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:foreigncall})
    _x = Ref(5.0)
    _dx = randn_tangent(Xoshiro(123456), _x)

    _a, _da = randn(5), randn(5)
    _b, _db = randn(4), randn(4)
    ptr_a, ptr_da = pointer(_a), pointer(_da)
    ptr_b, ptr_db = pointer(_b), pointer(_db)

    test_cases = Any[
        (false, :stability, nothing, Base.allocatedinline, Float64),
        (false, :stability, nothing, Base.allocatedinline, Vector{Float64}),
        (true, :stability, nothing, Array{Float64, 1}, undef, 5),
        (true, :stability, nothing, Array{Float64, 2}, undef, 5, 4),
        (true, :stability, nothing, Array{Float64, 3}, undef, 5, 4, 3),
        (true, :stability, nothing, Array{Float64, 4}, undef, 5, 4, 3, 2),
        (true, :stability, nothing, Array{Float64, 5}, undef, 5, 4, 3, 2, 1),
        (true, :stability, nothing, Array{Float64, 4}, undef, (2, 3, 4, 5)),
        (true, :stability, nothing, Array{Float64, 5}, undef, (2, 3, 4, 5, 6)),
        (true, :stability, nothing, Base._growbeg!, randn(5), 3),
        (true, :stability, nothing, Base._growend!, randn(5), 3),
        (true, :stability, nothing, Base._growat!, randn(5), 2, 2),
        (false, :stability, nothing, Base._deletebeg!, randn(5), 0),
        (false, :stability, nothing, Base._deletebeg!, randn(5), 2),
        (false, :stability, nothing, Base._deletebeg!, randn(5), 5),
        (false, :stability, nothing, Base._deleteend!, randn(5), 2),
        (false, :stability, nothing, Base._deleteend!, randn(5), 5),
        (false, :stability, nothing, Base._deleteend!, randn(5), 0),
        (false, :stability, nothing, Base._deleteat!, randn(5), 2, 2),
        (false, :stability, nothing, Base._deleteat!, randn(5), 1, 5),
        (false, :stability, nothing, Base._deleteat!, randn(5), 5, 1),
        (false, :stability, nothing, sizehint!, randn(5), 10),
        (false, :stability, nothing, copy, randn(5, 4)),
        (false, :stability, nothing, fill!, rand(Int8, 5), Int8(2)),
        (false, :stability, nothing, fill!, rand(UInt8, 5), UInt8(2)),
        (false, :stability, nothing, objectid, 5.0),
        (true, :stability, nothing, objectid, randn(5)),
        (true, :stability, nothing, pointer_from_objref, _x),
        (
            true,
            :none, # primal is unstable
            (lb=0.1, ub=100),
            unsafe_pointer_to_objref,
            CoDual(
                pointer_from_objref(_x),
                bitcast(Ptr{tangent_type(Nothing)}, pointer_from_objref(_dx)),
            ),
        ),
        (false, :none, nothing, Core.Compiler.return_type, sin, Tuple{Float64}),
        (
            false, :none, (lb=0.1, ub=100.0),
            Core.Compiler.return_type, Tuple{typeof(sin), Float64},
        ),
        (false, :stability, nothing, Threads.threadid),
        (false, :stability, nothing, typeintersect, Float64, Int),
        (
            true, :stability, nothing,
            unsafe_copyto!, CoDual(ptr_a, ptr_da), CoDual(ptr_b, ptr_db), 4,
        ),
        (false, :stability, nothing, unsafe_copyto!, randn(4), 2, randn(3), 1, 2),
        (
            false, :stability, nothing,
            unsafe_copyto!, [rand(3) for _ in 1:5], 2, [rand(4) for _ in 1:4], 1, 3,
        ),
        (false, :stability, nothing, deepcopy, 5.0),
        (false, :stability, nothing, deepcopy, randn(5)),
        (false, :none, nothing, deepcopy, TestResources.MutableFoo(5.0, randn(5))),
        (false, :none, nothing, deepcopy, TestResources.StructFoo(5.0, randn(5))),
        (false, :stability, nothing, deepcopy, (5.0, randn(5))),
        (false, :stability, nothing, deepcopy, (a=5.0, b=randn(5))),
        (false, :none, nothing, UnionAll, TypeVar(:a), Real),
        (false, :none, nothing, hash, "5", UInt(3)),
    ]
    memory = Any[_x, _dx, _a, _da, _b, _db]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:foreigncall})

    _x = Ref(5.0)

    function unsafe_copyto_tester(x::Vector{T}, y::Vector{T}, n::Int) where {T}
        GC.@preserve x y unsafe_copyto!(pointer(x), pointer(y), n)
        return x
    end

    test_cases = [
        Any[false, :none, nothing, reshape, randn(5, 4), (4, 5)],
        Any[false, :none, nothing, reshape, randn(5, 4), (2, 10)],
        Any[false, :none, nothing, reshape, randn(5, 4), (10, 2)],
        Any[false, :none, nothing, reshape, randn(5, 4), (5, 4, 1)],
        Any[false, :none, nothing, reshape, randn(5, 4), (2, 10, 1)],
        Any[false, :none, nothing, unsafe_copyto_tester, randn(5), randn(3), 2],
        Any[false, :none, nothing, unsafe_copyto_tester, randn(5), randn(6), 4],
        [
            false,
            :none,
            nothing,
            unsafe_copyto_tester,
            [randn(3) for _ in 1:5],
            [randn(4) for _ in 1:6],
            4,
        ],
        Any[false, :none, nothing, x -> unsafe_pointer_to_objref(pointer_from_objref(x)), _x],
        Any[false, :none, nothing, isassigned, randn(5), 4],
        Any[false, :none, nothing, x -> (Base._growbeg!(x, 2); x[1:2] .= 2.0), randn(5)],
    ]
    memory = Any[_x]
    return test_cases, memory
end
