# Fallback rule for foreigncall which gives an interpretable error message.
struct MissingForeigncallRuleError <: Exception
    msg::String
end

Base.showerror(io::IO, err::MissingForeigncallRuleError) = print(io, err.msg)

# Fallback foreigncall rrule. This is a sufficiently common special case, that it's worth
# creating an informative error message, so that users have some chance of knowing why
# they're not able to differentiate a piece of code.
function rrule!!(::CoDual{typeof(_foreigncall_)}, args...)
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

@is_primitive MinimalCtx Tuple{typeof(_foreigncall_), Vararg}

#
# Rules to handle / avoid foreigncall nodes
#

@is_primitive MinimalCtx Tuple{typeof(Base.allocatedinline), Type}
function rrule!!(f::CoDual{typeof(Base.allocatedinline)}, T::CoDual{<:Type})
    return zero_fcodual(Base.allocatedinline(primal(T))), NoPullback(f, T)
end

@is_primitive MinimalCtx Tuple{Type{<:Array{T, N}}, typeof(undef), Vararg} where {T, N}
function rrule!!(
    f::CoDual{Type{Array{T, N}}}, u::CoDual{typeof(undef)}, m::Vararg{CoDual}
) where {T, N}
    return zero_fcodual(Array{T, N}(undef, map(primal, m)...)), NoPullback(f, u, m...)
end

function rrule!!(
    f::CoDual{Type{Array{T, 0}}}, u::CoDual{typeof(undef)}, m::CoDual{Tuple{}}
) where {T}
    return zero_fcodual(Array{T, 0}(undef)), NoPullback(f, u, m)
end

@is_primitive MinimalCtx Tuple{Type{<:Array{T, N}}, typeof(undef), NTuple{N}} where {T, N}
function rrule!!(
    ::CoDual{<:Type{<:Array{T, N}}}, ::CoDual{typeof(undef)}, m::CoDual{NTuple{N}},
) where {T, N}
    return rrule!!(zero_fcodual(Array{T, N}), zero_fcodual(undef), m)
end

@is_primitive MinimalCtx Tuple{typeof(copy), Array}
function rrule!!(::CoDual{typeof(copy)}, a::CoDual{<:Array})
    dx = tangent(a)
    dy = copy(dx)
    y = CoDual(copy(primal(a)), dy)
    function copy_pullback!!(::NoRData)
        increment!!(dx, dy)
        return NoRData(), NoRData()
    end
    return y, copy_pullback!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._deletebeg!), Vector, Integer}
function rrule!!(
    ::CoDual{typeof(Base._deletebeg!)}, _a::CoDual{<:Vector}, _delta::CoDual{<:Integer},
)
    delta = primal(_delta)
    a = primal(_a)
    da = tangent(_a)

    a_beg = a[1:delta]
    da_beg = da[1:delta]

    Base._deletebeg!(a, delta)
    Base._deletebeg!(da, delta)

    function _deletebeg!_pb!!(::NoRData)
        splice!(a, 1:0, a_beg)
        splice!(da, 1:0, da_beg)
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(nothing), _deletebeg!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._deleteend!), Vector, Integer}
function rrule!!(
    ::CoDual{typeof(Base._deleteend!)}, _a::CoDual{<:Vector}, _delta::CoDual{<:Integer}
)
    # Extract data.
    a = primal(_a)
    da = tangent(_a)
    delta = primal(_delta)

    # Store the section to be cut for later.
    primal_tail = a[end-delta+1:end]
    tangent_tail = da[end-delta+1:end]

    # Cut the end off the primal and tangent.
    Base._deleteend!(a, delta)
    Base._deleteend!(da, delta)

    function _deleteend!_pb!!(::NoRData)

        Base._growend!(a, delta)
        a[end-delta+1:end] .= primal_tail

        Base._growend!(da, delta)
        da[end-delta+1:end] .= tangent_tail

        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(nothing), _deleteend!_pb!!
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
    da = tangent(_a)

    # Store the cut section for later.
    primal_mem = a[i:i+delta-1]
    tangent_mem = da[i:i+delta-1]

    # Run the primal.
    Base._deleteat!(a, i, delta)
    Base._deleteat!(da, i, delta)

    function _deleteat!_pb!!(::NoRData)
        splice!(a, i:i-1, primal_mem)
        splice!(da, i:i-1, tangent_mem)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(nothing), _deleteat!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._growbeg!), Vector, Integer}
function rrule!!(
    ::CoDual{typeof(Base._growbeg!)}, _a::CoDual{<:Vector{T}}, _delta::CoDual{<:Integer},
) where {T}
    d = primal(_delta)
    a = primal(_a)
    da = tangent(_a)
    Base._growbeg!(a, d)
    Base._growbeg!(da, d)
    function _growbeg!_pb!!(::NoRData)
        Base._deletebeg!(a, d)
        Base._deletebeg!(da, d)
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(nothing), _growbeg!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._growend!), Vector, Integer}
function rrule!!(
    ::CoDual{typeof(Base._growend!)}, _a::CoDual{<:Vector}, _delta::CoDual{<:Integer},
)
    d = primal(_delta)
    a = primal(_a)
    da = tangent(_a)
    Base._growend!(a, d)
    Base._growend!(da, d)
    function _growend!_pullback!!(::NoRData)
        Base._deleteend!(a, d)
        Base._deleteend!(da, d)
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(nothing), _growend!_pullback!!
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
    da = tangent(_a)

    # Run the primal.
    Base._growat!(a, i, delta)
    Base._growat!(da, i, delta)

    function _growat!_pb!!(::NoRData)
        deleteat!(a, i:i+delta-1)
        deleteat!(da, i:i+delta-1)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(nothing), _growat!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(fill!), Array{<:Union{UInt8, Int8}}, Integer}
function rrule!!(
    ::CoDual{typeof(fill!)}, a::CoDual{<:Array{<:Union{UInt8, Int8}}}, x::CoDual{<:Integer}
)
    pa = primal(a)
    old_value = copy(pa)
    fill!(pa, primal(x))
    function fill!_pullback!!(::NoRData)
        pa .= old_value
        return NoRData(), NoRData(), NoRData()
    end
    return a, fill!_pullback!!
end

@is_primitive MinimalCtx Tuple{typeof(sizehint!), Vector, Integer}
function rrule!!(f::CoDual{typeof(sizehint!)}, x::CoDual{<:Vector}, sz::CoDual{<:Integer})
    sizehint!(primal(x), primal(sz))
    sizehint!(tangent(x), primal(sz))
    return x, NoPullback(f, x, sz)
end

@is_primitive MinimalCtx Tuple{typeof(objectid), Any}
function rrule!!(f::CoDual{typeof(objectid)}, @nospecialize(x))
    return zero_fcodual(objectid(primal(x))), NoPullback(f, x)
end

@is_primitive MinimalCtx Tuple{typeof(pointer_from_objref), Any}
function rrule!!(f::CoDual{typeof(pointer_from_objref)}, x)
    y = CoDual(
        pointer_from_objref(primal(x)),
        bitcast(Ptr{tangent_type(Nothing)}, pointer_from_objref(tangent(x))),
    )
    return y, NoPullback(f, x)
end

@is_primitive MinimalCtx Tuple{typeof(CC.return_type), Vararg}
function rrule!!(f::CoDual{typeof(Core.Compiler.return_type)}, args...)
    pb!! = NoPullback(f, args...)
    return zero_fcodual(Core.Compiler.return_type(map(primal, args)...)), pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base.unsafe_pointer_to_objref), Ptr}
function rrule!!(f::CoDual{typeof(Base.unsafe_pointer_to_objref)}, x::CoDual{<:Ptr})
    y = CoDual(unsafe_pointer_to_objref(primal(x)), unsafe_pointer_to_objref(tangent(x)))
    return y, NoPullback(f, x)
end

@is_primitive MinimalCtx Tuple{typeof(Threads.threadid)}
function rrule!!(f::CoDual{typeof(Threads.threadid)})
    return zero_fcodual(Threads.threadid()), NoPullback(f)
end

@is_primitive MinimalCtx Tuple{typeof(typeintersect), Any, Any}
function rrule!!(f::CoDual{typeof(typeintersect)}, @nospecialize(a), @nospecialize(b))
    return zero_fcodual(typeintersect(primal(a), primal(b))), NoPullback(f, a, b)
end

function _increment_pointer!(x::Ptr{T}, y::Ptr{T}, N::Integer) where {T}
    increment!!(unsafe_wrap(Vector{T}, x, N), unsafe_wrap(Vector{T}, y, N))
    return x
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
    pdest = primal(dest)
    ddest = tangent(dest)
    unsafe_copyto!(pointer(dest_copy), pdest, _n)
    unsafe_copyto!(pointer(ddest_copy), ddest, _n)

    # Run primal computation.
    dsrc = tangent(src)
    unsafe_copyto!(primal(dest), primal(src), _n)
    unsafe_copyto!(tangent(dest), dsrc, _n)

    function unsafe_copyto!_pb!!(::NoRData)

        # Increment dsrc.
        _increment_pointer!(dsrc, ddest, _n)

        # Restore initial state.
        unsafe_copyto!(pdest, pointer(dest_copy), _n)
        unsafe_copyto!(ddest, pointer(ddest_copy), _n)

        return NoRData(), NoRData(), NoRData(), NoRData()
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
    pdest = primal(dest)
    ddest = tangent(dest)
    dest_copy = primal(dest)[dest_idx]
    ddest_copy = tangent(dest)[dest_idx]

    # Run primal computation.
    dsrc = tangent(src)
    unsafe_copyto!(primal(dest), _doffs, primal(src), _soffs, _n)
    unsafe_copyto!(tangent(dest), _doffs, dsrc, _soffs, _n)

    function unsafe_copyto_pb!!(::NoRData)

        # Increment dsrc.
        src_idx = _soffs:_soffs + _n - 1
        dsrc[src_idx] .= increment!!.(view(dsrc, src_idx), view(ddest, dest_idx))

        # Restore initial state.
        pdest[dest_idx] .= dest_copy
        ddest[dest_idx] .= ddest_copy

        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end

    return dest, unsafe_copyto_pb!!
end



#
# general foreigncall nodes
#

function rrule!!(
    ::CoDual{typeof(_foreigncall_)},
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
    return y, NoPullback(ntuple(_ -> NoRData(), 7))
end

# function rrule!!(
#     ::CoDual{typeof(_foreigncall_)},
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
    ::CoDual{typeof(_foreigncall_)},
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
    return y, NoPullback(ntuple(_ -> NoRData(), 9))
end

function rrule!!(
    ::CoDual{typeof(_foreigncall_)},
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
    return zero_fcodual(y), NoPullback(ntuple(_ -> NoRData(), length(args) + 8))
end

@is_primitive MinimalCtx Tuple{typeof(deepcopy), Any}
function rrule!!(::CoDual{typeof(deepcopy)}, x::CoDual)
    fdx = tangent(x)
    dx = zero_rdata(primal(x))
    y = deepcopy(x)
    fdy = tangent(y)
    function deepcopy_pb!!(dy)
        increment!!(fdx, fdy)
        return NoRData(), increment!!(dx, dy)
    end
    return y, deepcopy_pb!!
end

@is_primitive MinimalCtx Tuple{Type{UnionAll}, TypeVar, Any}
@is_primitive MinimalCtx Tuple{Type{UnionAll}, TypeVar, Type}
function rrule!!(f::CoDual{<:Type{UnionAll}}, x::CoDual{<:TypeVar}, y::CoDual{<:Type})
    return zero_fcodual(UnionAll(primal(x), primal(y))), NoPullback(f, x, y)
end

@is_primitive MinimalCtx Tuple{typeof(hash), Union{String, SubString{String}}, UInt}
function rrule!!(
    f::CoDual{typeof(hash)}, s::CoDual{P}, h::CoDual{UInt}
) where {P<:Union{String, SubString{String}}}
    return zero_fcodual(hash(primal(s), primal(h))), NoPullback(f, s, h)
end

function rrule!!(
    ::CoDual{typeof(_foreigncall_)}, ::CoDual{Val{:jl_string_ptr}}, args::Vararg{CoDual, N}
) where {N}
    x = tuple_map(primal, args)
    pb!! = NoPullback((NoRData(), NoRData(), tuple_map(_ -> NoRData(), args)...))
    return uninit_fcodual(_foreigncall_(Val(:jl_string_ptr), x...)), pb!!
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
    @eval function rrule!!(::CoDual{typeof(_foreigncall_)}, ::CoDual{Val{$name}}, args...)
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
        (true, :stability, nothing, Array{Float64, 0}, undef),
        (true, :stability, nothing, Array{Float64, 1}, undef, 5),
        (true, :stability, nothing, Array{Float64, 2}, undef, 5, 4),
        (true, :stability, nothing, Array{Float64, 3}, undef, 5, 4, 3),
        (true, :stability, nothing, Array{Float64, 4}, undef, 5, 4, 3, 2),
        (true, :stability, nothing, Array{Float64, 5}, undef, 5, 4, 3, 2, 1),
        (true, :stability, nothing, Array{Float64, 0}, undef, ()),
        (true, :stability, nothing, Array{Float64, 4}, undef, (2, 3, 4, 5)),
        (true, :stability, nothing, Array{Float64, 5}, undef, (2, 3, 4, 5, 6)),
        (false, :stability, nothing, copy, randn(5, 4)),
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
        (false, :stability, nothing, fill!, rand(Int8, 5), Int8(2)),
        (false, :stability, nothing, fill!, rand(UInt8, 5), UInt8(2)),
        (true, :stability, nothing, Base._growbeg!, randn(5), 3),
        (true, :stability, nothing, Base._growend!, randn(5), 3),
        (true, :stability, nothing, Base._growat!, randn(5), 2, 2),
        (false, :stability, nothing, objectid, 5.0),
        (true, :stability, nothing, objectid, randn(5)),
        (true, :stability, nothing, pointer_from_objref, _x),
        (
            true,
            :none, # primal is unstable
            (lb=1e-3, ub=100),
            unsafe_pointer_to_objref,
            CoDual(
                pointer_from_objref(_x),
                bitcast(Ptr{tangent_type(Nothing)}, pointer_from_objref(_dx)),
            ),
        ),
        (false, :none, nothing, Core.Compiler.return_type, sin, Tuple{Float64}),
        (
            false, :none, (lb=1e-3, ub=100.0),
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
        (
            true, :none, nothing,
            _foreigncall_,
            Val(:jl_array_ptr),
            Val(Ptr{Float64}),
            (Val(Any), ),
            Val(0), # nreq
            Val(:ccall), # calling convention
            randn(5),
        )
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
        (false, :none, nothing, reshape, randn(5, 4), (4, 5)),
        (false, :none, nothing, reshape, randn(5, 4), (2, 10)),
        (false, :none, nothing, reshape, randn(5, 4), (10, 2)),
        (false, :none, nothing, reshape, randn(5, 4), (5, 4, 1)),
        (false, :none, nothing, reshape, randn(5, 4), (2, 10, 1)),
        (false, :none, nothing, unsafe_copyto_tester, randn(5), randn(3), 2),
        (false, :none, nothing, unsafe_copyto_tester, randn(5), randn(6), 4),
        (
            false, :none, nothing,
            unsafe_copyto_tester, [randn(3) for _ in 1:5], [randn(4) for _ in 1:6], 4,
        ),
        (
            false, :none, (lb=0.1, ub=150),
            x -> unsafe_pointer_to_objref(pointer_from_objref(x)), _x,
        ),
        (false, :none, nothing, isassigned, randn(5), 4),
        (false, :none, nothing, x -> (Base._growbeg!(x, 2); x[1:2] .= 2.0), randn(5)),
    ]
    memory = Any[_x]
    return test_cases, memory
end
