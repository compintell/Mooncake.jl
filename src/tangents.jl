"""
    NoTangent

The type in question has no meaningful notion of a tangent space.
Generally, you shouldn't use this -- just let the default recursive tangent
construction work.
You might need to use this for `primitive type`s though.
"""
struct NoTangent end

"""
    PossiblyUninitTangent{T}

Represents a `T` which maybe or may not be present. Does not distinguish between 0 and
not being present.
"""
struct PossiblyUninitTangent{T}
    tangent::T
    PossiblyUninitTangent{T}(tangent::T) where {T} = new{T}(tangent)
    PossiblyUninitTangent{T}() where {T} = new{T}()
end

_copy(x::P) where {P<:PossiblyUninitTangent} = is_init(x) ? P(_copy(x.tangent)) : P()

@inline PossiblyUninitTangent(tangent::T) where {T} = PossiblyUninitTangent{T}(tangent)
@inline PossiblyUninitTangent(T::Type) = PossiblyUninitTangent{T}()

@inline is_init(t::PossiblyUninitTangent) = isdefined(t, :tangent)
is_init(t) = true

val(x::PossiblyUninitTangent) = is_init(x) ? x.tangent : error("Uninitialised")
val(x) = x

struct Tangent{Tfields<:NamedTuple}
    fields::Tfields
end

Base.:(==)(x::Tangent, y::Tangent) = x.fields == y.fields

mutable struct MutableTangent{Tfields<:NamedTuple}
    fields::Tfields
    MutableTangent{Tfields}() where {Tfields} = new{Tfields}()
    MutableTangent(fields::Tfields) where {Tfields} = MutableTangent{Tfields}(fields)
    function MutableTangent{Tfields}(
        fields::NamedTuple{names}
    ) where {names,Tfields<:NamedTuple{names}}
        return new{Tfields}(fields)
    end
end

Base.:(==)(x::MutableTangent, y::MutableTangent) = x.fields == y.fields

fields_type(::Type{MutableTangent{Tfields}}) where {Tfields<:NamedTuple} = Tfields
fields_type(::Type{Tangent{Tfields}}) where {Tfields<:NamedTuple} = Tfields
fields_type(::Type{<:Union{MutableTangent,Tangent}}) = NamedTuple

const PossiblyMutableTangent{T} = Union{MutableTangent{T},Tangent{T}}

"""
    get_tangent_field(t::Union{MutableTangent, Tangent}, i::Int)

Gets the `i`th field of data in `t`.

Has the same semantics that `getfield!` would have if the data in the `fields` field of `t`
were actually fields of `t`. This is the moral equivalent of `getfield` for
`MutableTangent`.
"""
@inline get_tangent_field(t::PossiblyMutableTangent, i::Int) = val(getfield(t.fields, i))

@inline function get_tangent_field(t::PossiblyMutableTangent{F}, s::Symbol) where {F}
    return get_tangent_field(t, _sym_to_int(F, Val(s)))
end

"""
    set_tangent_field!(t::MutableTangent{Tfields}, i::Int, x) where {Tfields}

Sets the value of the `i`th field of the data in `t` to value `x`.

Has the same semantics that `setfield!` would have if the data in the `fields` field of `t`
were actually fields of `t`. This is the moral equivalent of `setfield!` for
`MutableTangent`.
"""
@inline function set_tangent_field!(t::MutableTangent{Tfields}, i::Int, x) where {Tfields}
    fields = t.fields
    Ti = fieldtype(Tfields, i)
    new_val = Ti <: PossiblyUninitTangent ? Ti(x) : x
    new_fields = Tfields(ntuple(n -> n == i ? new_val : fields[n], fieldcount(Tfields)))
    t.fields = new_fields
    return x
end

@inline function set_tangent_field!(t::MutableTangent{T}, s::Symbol, x) where {T}
    return set_tangent_field!(t, _sym_to_int(T, Val(s)), x)
end

@generated function _sym_to_int(::Type{Tfields}, ::Val{s}) where {Tfields,s}
    return findfirst(==(s), fieldnames(Tfields))
end

function tangent_field_types_exprs(P::Type)
    tangent_type_exprs = map(fieldtypes(P), always_initialised(P)) do _P, init
        T_expr = Expr(:call, :tangent_type, _P)
        return init ? T_expr : Expr(:curly, PossiblyUninitTangent, T_expr)
    end
    return tangent_type_exprs
end

# It is essential that this gets inlined. If it does not, then we run into performance
# issues with the recursion to compute tangent types for nested types.
@generated function tangent_field_types(::Type{P}) where {P}
    return Expr(:call, :tuple, tangent_field_types_exprs(P)...)
end

@generated function build_tangent(::Type{P}, fields::Vararg{Any,N}) where {P,N}
    tangent_values_exprs = map(enumerate(tangent_field_types(P))) do (n, tt)
        tt <: PossiblyUninitTangent && return n <= N ? :($tt(fields[$n])) : :($tt())
        return :(fields[$n])
    end
    tuple_expr = Expr(:tuple, tangent_values_exprs...)
    return Expr(:call, tangent_type(P), Expr(:call, NamedTuple{fieldnames(P)}, tuple_expr))
end

"""
    @tt_effects tangent_type(...)

Effects which ought to be applied to `tangent_type`.
"""
macro tt_effects(expr)
    return esc(:(Base.@assume_effects :consistent :removable $expr))
end

"""
    tangent_type(P)

There must be a single type used to represents tangents of primals of type `P`, and it must
be given by `tangent_type(P)`.

Warning: this function assumes the effects `:removable` and `:consistent`. This is necessary
to ensure good performance, but imposes precise constraints on your implementation. If
adding new methods to `tangent_type`, you should consult the extended help of
`Base.@assume_effects` to see what this imposes upon your implementation.

# Extended help

The tangent types which Mooncake.jl uses are quite similar in spirit to ChainRules.jl.
For example, tangent "vectors" for
1. `Float64`s are `Float64`s,
1. `Vector{Float64}`s are `Vector{Float64}`s, and
1. `struct`s are other another (special) `struct` with field types specified recursively.

There are, however, some major differences.
Firstly, while it is certainly true that the above tangent types are permissible in
ChainRules.jl, they are not the uniquely permissible types. For example, `ZeroTangent` is
also a permissible type of tangent for any of them, and `Float32` is permissible for
`Float64`. This is a general theme in ChainRules.jl -- it intentionally declines to place
restrictions on what type can be used to represent the tangent of a given type.

Mooncake.jl differs from this.
**It insists that each primal type is associated to a _single_ tangent type.**
Furthermore, this type is _always_ given by the function `Mooncake.tangent_type(primal_type)`.

Consider some more worked examples.

#### Int

`Int` is not a differentiable type, so its tangent type is `NoTangent`:
```jldoctest
julia> tangent_type(Int)
NoTangent
```

#### Tuples

The tangent type of a `Tuple` is defined recursively based on its field types. For example
```jldoctest
julia> tangent_type(Tuple{Float64, Vector{Float64}, Int})
Tuple{Float64, Vector{Float64}, NoTangent}
```

There is one edge case to be aware of: if all of the field of a `Tuple` are
non-differentiable, then the tangent type is `NoTangent`. For example,
```jldoctest
julia> tangent_type(Tuple{Int, Int})
NoTangent
```

#### Structs

As with `Tuple`s, the tangent type of a struct is, by default, given recursively.
In particular, the tangent type of a `struct` type is `Tangent`.
This type contains a `NamedTuple` containing the tangent to each field in the primal `struct`.

As with `Tuple`s, if all field types are non-differentiable, the tangent type of the entire
struct is `NoTangent`.

There are a couple of additional subtleties to consider over `Tuple`s though. Firstly, not
all fields of a `struct` have to be defined. Fortunately, Julia makes it easy to determine
how many of the fields might possibly not be defined. The tangent associated to any field
which might possibly not be defined is wrapped in a `PossiblyUninitTangent`.

Furthermore, `struct`s can have fields whose static type is abstract. For example
```jldoctest foo
julia> struct Foo
           x
       end
```
If you ask for the tangent type of `Foo`, you will see that it is
```jldoctest foo
julia> tangent_type(Foo)
Tangent{@NamedTuple{x}}
```
Observe that the field type associated to `x` is `Any`. The way to understand this result is
to observe that
1. `x` could have literally any type at runtime, so we know nothing about what its tangent
    type must be until runtime, and
1. we require that the tangent type of `Foo` be unique.
The consequence of these two considerations is that the tangent type of `Foo` must be able
to contain any type of tangent in its `x` field. It follows that the fieldtype of the `x`
field of `Foo`s tangent must be `Any`.



#### Mutable Structs

The tangent type for `mutable struct`s have the same set of considerations as `struct`s.
The only difference is that they must themselves be mutable.
Consequently, we use a type called `MutableTangent` to represent their tangents.
It is a `mutable struct` with the same structure as `Tangent`.

For example, if you ask for the `tangent_type` of
```jldoctest bar
julia> mutable struct Bar
           x::Float64
       end
```
you will find that it is
```jldoctest bar
julia> tangent_type(Bar)
MutableTangent{@NamedTuple{x::Float64}}
```


#### Primitive Types

We've already seen a couple of primitive types (`Float64` and `Int`).
The basic story here is that all primitive types require an explicit specification of what their tangent type must be.

One interesting case are `Ptr` types.
The tangent type of a `Ptr{P}` is `Ptr{T}`, where `T = tangent_type(P)`.
For example
```julia
julia> tangent_type(Ptr{Float64})
Ptr{Float64}
```

"""
tangent_type(T)

tangent_type(x) = throw(error("$x is not a type. Perhaps you meant typeof(x)?"))

# The "Bottom" type.
tangent_type(::Type{Union{}}) = NoTangent

# This is essential for DataType, as the recursive definition always recurses infinitely,
# because one of the fieldtypes is itself always a DataType. In particular, we'll always
# eventually hit `Any`, whose `super` field is `Any`.
# This makes it clear that we can't recursively construct tangents for data structures which
# refer to themselves...
tangent_type(::Type{<:Type}) = NoTangent

tangent_type(::Type{<:TypeVar}) = NoTangent

@tt_effects tangent_type(::Type{Ptr{P}}) where {P} = Ptr{tangent_type(P)}

tangent_type(::Type{<:Ptr}) = NoTangent

tangent_type(::Type{Bool}) = NoTangent

tangent_type(::Type{Char}) = NoTangent

tangent_type(::Type{Symbol}) = NoTangent

tangent_type(::Type{Module}) = NoTangent

tangent_type(::Type{Nothing}) = NoTangent

tangent_type(::Type{Expr}) = NoTangent

tangent_type(::Type{Core.TypeofVararg}) = NoTangent

tangent_type(::Type{SimpleVector}) = Vector{Any}

tangent_type(::Type{P}) where {P<:Union{UInt8,UInt16,UInt32,UInt64,UInt128}} = NoTangent

tangent_type(::Type{P}) where {P<:Union{Int8,Int16,Int32,Int64,Int128}} = NoTangent

tangent_type(::Type{<:Core.Builtin}) = NoTangent

tangent_type(::Type{P}) where {P<:IEEEFloat} = P

tangent_type(::Type{<:Core.LLVMPtr}) = NoTangent

tangent_type(::Type{String}) = NoTangent

tangent_type(::Type{<:Array{P,N}}) where {P,N} = Array{tangent_type(P),N}

tangent_type(::Type{<:Array{P,N} where {P}}) where {N} = Array

tangent_type(::Type{<:MersenneTwister}) = NoTangent

tangent_type(::Type{Core.TypeName}) = NoTangent

tangent_type(::Type{Core.MethodTable}) = NoTangent

tangent_type(::Type{DimensionMismatch}) = NoTangent

tangent_type(::Type{Method}) = NoTangent

tangent_type(::Type{<:Enum}) = NoTangent

# Inferable version of `findall` for `Tuple`s.
_findall(::Any, ::Int, ::Tuple{}) = ()
function _findall(cond, ind::Int, x::Tuple)
    tail = _findall(cond, ind + 1, x[2:end])
    return cond(x[1]) ? (ind, tail...) : tail
end

function split_union_tuple_type(tangent_types)

    # Create first split.
    ta_types = map(tangent_types) do T
        return T isa Union ? T.a : T
    end
    ta = Tuple{ta_types...}

    # Create second split.
    tb_types = map(tangent_types) do T
        return T isa Union ? T.b : T
    end
    tb = Tuple{tb_types...}

    return Union{ta,tb}
end

# Generated functions cannot emit closures, so this is defined here for use below.
isconcrete_or_union(p) = p isa Union || isconcretetype(p)

@tt_effects @generated function tangent_type(::Type{P}) where {N,P<:Tuple{Vararg{Any,N}}}

    # As with other types, tangent type of Union is Union of tangent types.
    P isa Union && return :(Union{tangent_type($(P.a)),tangent_type($(P.b))})

    # Determine whether P isa a Tuple with a Vararg, e.g, Tuple, or Tuple{Float64, Vararg}.
    # Need to exclude `UnionAll`s from this, by checking `isa(P, DataType)`, in order to
    # ensure that `Base.datatype_fieldcount(P)` will run successfully.
    isa(P, DataType) && !(@isdefined(N)) && return Any

    # Tuple{} can only have `NoTangent` as its tangent type. As before, verify we don't have
    # a UnionAll before running to ensure that datatype_fieldcount will run.
    isa(P, DataType) && N == 0 && return NoTangent

    # Expression to construct `Tuple` type containing tangent type for all fields.
    tangent_type_exprs = map(n -> :(tangent_type(fieldtype(P, $n))), 1:N)
    tangent_types = Expr(:call, tuple, tangent_type_exprs...)

    # Construct a Tuple type of the same length as `P`, containing all `NoTangent`s.
    T_all_notangent = Tuple{Vararg{NoTangent,N}}

    return quote

        # Get tangent types for all fields. If they're all `NoTangent`, return `NoTangent`.
        # i.e. if `P = Tuple{Int, Int}`, do not return `Tuple{NoTangent, NoTangent}`.
        # Simplify and return `NoTangent`.
        tangent_types = $tangent_types
        T = Tuple{tangent_types...}
        T <: $T_all_notangent && return NoTangent

        # If exactly one of the field types is a Union, then split.
        union_fields = _findall(Base.Fix2(isa, Union), 1, tangent_types)
        if length(union_fields) == 1 && all(tuple_map(isconcrete_or_union, tangent_types))
            return split_union_tuple_type(tangent_types)
        end

        # If it's _possible_ for a subtype of `P` to have tangent type `NoTangent`, then we
        # must account for that by returning the union of `NoTangent` and `T`. For example,
        # if `P = Tuple{Any, Int}`, then `P2 = Tuple{Int, Int}` is a subtype. Since `P2` has
        # tangent type `NoTangent`, it must be true that `NoTangent <: tangent_type(P)`. If,
        # on the other hand, it's not possible for `NoTangent` to be the tangent type, e.g.
        # for `Tuple{Float64, Any}`, then there's no need to take the union.
        return $T_all_notangent <: T ? Union{T,NoTangent} : T
    end
end

@tt_effects function tangent_type(::Type{P}) where {N,P<:NamedTuple{N}}
    P isa Union && return Union{tangent_type(P.a),tangent_type(P.b)}
    !isconcretetype(P) && return Union{NoTangent,NamedTuple{N}}
    TT = tangent_type(Tuple{fieldtypes(P)...})
    TT == NoTangent && return NoTangent
    return isconcretetype(TT) ? NamedTuple{N,TT} : Any
end

@tt_effects @generated function tangent_type(::Type{P}) where {P}

    # This method can only handle struct types. Something has gone wrong if P is primitive.
    if isprimitivetype(P)
        return error("$P is a primitive type. Implement a method of `tangent_type` for it.")
    end

    # If the type is a Union, then take the union type of its arguments.
    P isa Union && return :(Union{tangent_type($(P.a)),tangent_type($(P.b))})

    # If the type is itself abstract, it's tangent could be anything.
    # The same goes for if the type has any undetermined type parameters.
    (isabstracttype(P) || !isconcretetype(P)) && return Any

    tangent_fields_types_expr = Expr(:curly, Tuple, tangent_field_types_exprs(P)...)
    T_all_notangent = Tuple{Vararg{NoTangent,fieldcount(P)}}
    return quote

        # Construct a `Tuple{...}` whose fields are the tangent types of the fields of `P`.
        tangent_field_types_tuple = $tangent_fields_types_expr

        # If all fields are definitely `NoTangent`s, then return `NoTangent`.
        tangent_field_types_tuple <: $T_all_notangent && return NoTangent

        # Derive tangent type.
        bt = NamedTuple{$(fieldnames(P)),tangent_field_types_tuple}
        return $(ismutabletype(P) ? MutableTangent : Tangent){bt}
    end
end

backing_type(P::Type) = NamedTuple{fieldnames(P),Tuple{tangent_field_types(P)...}}

"""
    zero_tangent(x)

Returns the unique zero element of the tangent space of `x`.
It is an error for the zero element of the tangent space of `x` to be represented by
anything other than that which this function returns.

Internally, `zero_tangent` calls `zero_tangent_internal`, which handles different types of inputs differently.
`zero_tangent_internal` has two variants:
1. For `isbitstype` types, `zero_tangent_internal` takes one argument. 
2. Otherwise, `zero_tangent_internal` takes another argument which is an `IdDict`, which
handles both circular references and aliasing correctly.
"""
zero_tangent(x)
zero_tangent(x::P) where {P} = zero_tangent_internal(x, isbitstype(P) ? nothing : IdDict())

const StackDict = Union{Nothing,IdDict}

# the `stackdict` naming following convention of Julia's `deepcopy` and `deepcopy_internal`
# https://github.com/JuliaLang/julia/blob/48d4fd48430af58502699fdf3504b90589df3852/base/deepcopy.jl#L35
zero_tangent_internal(::Union{Int8,Int16,Int32,Int64,Int128}, ::Any) = NoTangent()
zero_tangent_internal(x::IEEEFloat, ::Any) = zero(x)
@generated function zero_tangent_internal(x::Tuple, stackdict::Any)
    zt_exprs = map(n -> :(zero_tangent_internal(x[$n], stackdict)), 1:fieldcount(x))
    return quote
        tangent_type($x) == NoTangent && return NoTangent()
        return $(Expr(:call, :tuple, zt_exprs...))
    end
end
function zero_tangent_internal(x::NamedTuple, stackdict::Any)
    tangent_type(typeof(x)) == NoTangent && return NoTangent()
    return tuple_map(Base.Fix2(zero_tangent_internal, stackdict), x)
end
function zero_tangent_internal(x::Ptr, ::Any)
    return throw(ArgumentError("zero_tangent not available for pointers."))
end
function zero_tangent_internal(x::SimpleVector, stackdict::IdDict)
    return map!(
        n -> zero_tangent_internal(x[n], stackdict),
        Vector{Any}(undef, length(x)),
        eachindex(x),
    )
end
function zero_tangent_internal(x::P, stackdict) where {P}
    tangent_type(P) == NoTangent && return NoTangent()

    if tangent_type(P) <: MutableTangent
        if !(stackdict isa IdDict)
            throw(
                ArgumentError(
                    "Internal error: stackdict must be an IdDict for mutable structs, not $(typeof(stackdict)). Please report this issue.",
                ),
            )
        end
        if haskey(stackdict, x)
            return stackdict[x]::tangent_type(P)
        end
        stackdict[x] = tangent_type(P)() # create a uninitialised MutableTangent
        # if circular reference exists, then the recursive call will first look up the stackdict
        # and return the uninitialised MutableTangent
        # after the recursive call returns, the stackdict will be initialised
        stackdict[x].fields = zero_tangent_struct_field(x, stackdict)
        return stackdict[x]::tangent_type(P)
    else
        return tangent_type(P)(zero_tangent_struct_field(x, stackdict))
    end
end

function zero_tangent_struct_field(x::P, d) where {P}
    Tfs = tangent_field_types(P)
    inits = always_initialised(P)
    tangent_field_zeros = ntuple(Val(fieldcount(P))) do n
        T = Tfs[n]
        inits[n] && return zero_tangent_internal(getfield(x, n), d)
        return isdefined(x, n) ? T(zero_tangent_internal(getfield(x, n), d)) : T()
    end
    return backing_type(P)(tangent_field_zeros)
end

"""
    uninit_tangent(x)

Related to `zero_tangent`, but a bit different. Check current implementation for
details -- this docstring is intentionally non-specific in order to avoid becoming outdated.
"""
@inline uninit_tangent(x) = zero_tangent(x)
@inline uninit_tangent(x::Ptr{P}) where {P} = bitcast(Ptr{tangent_type(P)}, x)

"""
    randn_tangent(rng::AbstractRNG, x::T) where {T}

Required for testing.
Generate a randomly-chosen tangent to `x`.
The design is closely modelled after `zero_tangent`.
"""
function randn_tangent(rng::AbstractRNG, x::T) where {T}
    return randn_tangent_internal(rng, x, isbitstype(T) ? nothing : IdDict())
end

randn_tangent_internal(::AbstractRNG, ::NoTangent, ::Any) = NoTangent()
randn_tangent_internal(rng::AbstractRNG, ::T, ::Any) where {T<:IEEEFloat} = randn(rng, T)
@generated function randn_tangent_internal(rng::AbstractRNG, x::Tuple, stackdict::Any)
    rt_exprs = map(n -> :(randn_tangent_internal(rng, x[$n], stackdict)), 1:fieldcount(x))
    return quote
        tangent_type($x) == NoTangent && return NoTangent()
        return $(Expr(:call, :tuple, rt_exprs...))
    end
end
function randn_tangent_internal(rng::AbstractRNG, x::NamedTuple, stackdict::Any)
    tangent_type(typeof(x)) == NoTangent && return NoTangent()
    return tuple_map(x -> randn_tangent_internal(rng, x, stackdict), x)
end
function randn_tangent_internal(rng::AbstractRNG, x::SimpleVector, stackdict::IdDict)
    return map!(Vector{Any}(undef, length(x)), eachindex(x)) do n
        return randn_tangent_internal(rng, x[n], stackdict)
    end
end
function randn_tangent_internal(rng::AbstractRNG, x::P, stackdict) where {P}
    tangent_type(P) == NoTangent && return NoTangent()
    if isprimitivetype(P)
        return throw(ArgumentError("$P is a primitive type. Defined randn_tangent for it."))
    end
    if tangent_type(P) <: MutableTangent
        if !(stackdict isa IdDict)
            throw(
                ArgumentError(
                    "Internal error: stackdict must be an IdDict for mutable structs, not $(typeof(stackdict)). Please report this issue.",
                ),
            )
        end
        if haskey(stackdict, x)
            return stackdict[x]::tangent_type(P)
        end
        stackdict[x] = tangent_type(P)()
        stackdict[x].fields = randn_tangent_struct_field(rng, x, stackdict)
        return stackdict[x]::tangent_type(P)
    else
        return tangent_type(P)(randn_tangent_struct_field(rng, x, stackdict))
    end
end

function randn_tangent_struct_field(rng::AbstractRNG, x::P, d) where {P}
    Tfs = tangent_field_types(P)
    inits = always_initialised(P)
    tangent_field_zeros = ntuple(Val(fieldcount(P))) do n
        T = Tfs[n]
        inits[n] && return randn_tangent_internal(rng, getfield(x, n), d)
        return isdefined(x, n) ? T(randn_tangent_internal(rng, getfield(x, n), d)) : T()
    end
    return backing_type(P)(tangent_field_zeros)
end

"""
    increment!!(x::T, y::T) where {T}

Add `x` to `y`. If `ismutabletype(T)`, then `increment!!(x, y) === x` must hold.
That is, `increment!!` will mutate `x`.
This must apply recursively if `T` is a composite type whose fields are mutable.
"""
increment!!(::NoTangent, ::NoTangent) = NoTangent()
increment!!(x::T, y::T) where {T<:IEEEFloat} = x + y
increment!!(x::Ptr{T}, y::Ptr{T}) where {T} = x === y ? x : throw(error("eurgh"))
increment!!(x::T, y::T) where {T<:Tuple} = tuple_map(increment!!, x, y)::T
increment!!(x::T, y::T) where {T<:NamedTuple} = T(tuple_map(increment!!, x, y))
function increment!!(x::T, y::T) where {T<:PossiblyUninitTangent}
    is_init(x) && is_init(y) && return T(increment!!(val(x), val(y)))
    is_init(x) && !is_init(y) && error("x is initialised, but y is not")
    !is_init(x) && is_init(y) && error("x is not initialised, but y is")
    return x
end
increment!!(x::T, y::T) where {T<:Tangent} = T(increment!!(x.fields, y.fields))
function increment!!(x::T, y::T) where {T<:MutableTangent}
    x === y && return x
    x.fields = increment!!(x.fields, y.fields)
    return x
end

struct NoCache end

Base.haskey(::NoCache, x) = false
Base.setindex!(::NoCache, v, x) = nothing

const IncCache = Union{NoCache,IdDict{Any,Bool}}

"""
    set_to_zero!!(x)

Set `x` to its zero element (`x` should be a tangent, so the zero must exist).
"""
set_to_zero!!(x) = _set_to_zero!!(IdDict{Any,Bool}(), x)

_set_to_zero!!(::IncCache, ::NoTangent) = NoTangent()
_set_to_zero!!(::IncCache, x::Base.IEEEFloat) = zero(x)
function _set_to_zero!!(c::IncCache, x::Union{Tuple,NamedTuple})
    return tuple_map(Base.Fix1(_set_to_zero!!, c), x)
end
function _set_to_zero!!(c::IncCache, x::T) where {T<:PossiblyUninitTangent}
    return is_init(x) ? T(_set_to_zero!!(c, val(x))) : x
end
_set_to_zero!!(c::IncCache, x::T) where {T<:Tangent} = T(_set_to_zero!!(c, x.fields))
function _set_to_zero!!(c::IncCache, x::MutableTangent)
    haskey(c, x) && return x
    setindex!(c, false, x)
    x.fields = _set_to_zero!!(c, x.fields)
    return x
end

"""
    _scale(a::Float64, t::T) where {T}

Required for testing.
Should be defined for all standard tangent types.

Multiply tangent `t` by scalar `a`. Always possible because any given tangent type must
correspond to a vector field. Not using `*` in order to avoid piracy.
"""
_scale(::Float64, ::NoTangent) = NoTangent()
_scale(a::Float64, t::T) where {T<:IEEEFloat} = T(a * t)
_scale(a::Float64, t::Union{Tuple,NamedTuple}) = map(Base.Fix1(_scale, a), t)
function _scale(a::Float64, t::T) where {T<:PossiblyUninitTangent}
    return is_init(t) ? T(_scale(a, val(t))) : T()
end
_scale(a::Float64, t::T) where {T<:Union{Tangent,MutableTangent}} = T(_scale(a, t.fields))

struct FieldUndefined end

"""
    _dot(t::T, s::T)::Float64 where {T}

Required for testing.
Should be defined for all standard tangent types.

Inner product between tangents `t` and `s`. Must return a `Float64`.
Always available because all tangent types correspond to finite-dimensional vector spaces.
"""
_dot(::NoTangent, ::NoTangent) = 0.0
_dot(t::T, s::T) where {T<:IEEEFloat} = Float64(t * s)
_dot(t::T, s::T) where {T<:Union{Tuple,NamedTuple}} = sum(map(_dot, t, s); init=0.0)
function _dot(t::T, s::T) where {T<:PossiblyUninitTangent}
    is_init(t) && is_init(s) && return _dot(val(t), val(s))
    return 0.0
end
function _dot(t::T, s::T) where {T<:Union{Tangent,MutableTangent}}
    return sum(_map(_dot, t.fields, s.fields); init=0.0)
end

"""
    _add_to_primal(p::P, t::T, unsafe::Bool=false) where {P, T}

Adds `t` to `p`, returning a `P`. It must be the case that `tangent_type(P) == T`.

If `unsafe` is `true` and `P` is a composite type, then `_add_to_primal` will construct a
new instance of `P` by directly invoking the `:new` instruction for `P`, rather than
attempting to use the default constructor for `P`. This is fine if you are confident that
the new `P` constructed by adding `t` to `p` will always be a valid instance of `P`, but
could cause problems if you are not confident of this.

This is, for example, fine for the following type:
```julia
struct Foo{T}
    x::Vector{T}
    y::Vector{T}
    function Foo(x::Vector{T}, y::Vector{T}) where {T}
        @assert length(x) == length(y)
        return new{T}(x, y)
    end
end
```
Here, the value returned by `_add_to_primal` will satisfy the invariant asserted in the
inner constructor for `Foo`.
"""
_add_to_primal(p, t) = _add_to_primal(p, t, false)
_add_to_primal(x, ::NoTangent, ::Bool) = x
_add_to_primal(x::T, t::T, ::Bool) where {T<:IEEEFloat} = x + t
function _add_to_primal(x::SimpleVector, t::Vector{Any}, unsafe::Bool)
    return svec(map(n -> _add_to_primal(x[n], t[n], unsafe), eachindex(x))...)
end
function _add_to_primal(x::Tuple, t::Tuple, unsafe::Bool)
    return _map((x, t) -> _add_to_primal(x, t, unsafe), x, t)
end
function _add_to_primal(x::NamedTuple, t::NamedTuple, unsafe::Bool)
    return _map((x, t) -> _add_to_primal(x, t, unsafe), x, t)
end

struct AddToPrimalException <: Exception
    primal_type::Type
end

function Base.showerror(io::IO, err::AddToPrimalException)
    msg =
        "Attempted to construct an instance of $(err.primal_type) using the default " *
        "constuctor. In most cases, this error is caused by the lack of existence of the " *
        "default constructor for this type. There are two approaches to dealing with " *
        "this problem. The first is to avoid having to call `_add_to_primal` on this " *
        "type, which can be achieved by avoiding testing functions whose arguments are " *
        "of this type. If this cannot be avoided, you should consider using calling " *
        "`Mooncake._add_to_primal` with its third positional argument set to `true`. " *
        "If you are using some of Mooncake's testing functionality, this can be achieved " *
        "by setting the `unsafe_perturb` setting to `true` -- check the docstring " *
        "for `Mooncake._add_to_primal` to ensure that your use case is unlikely to " *
        "cause problems."
    return println(io, msg)
end

function _add_to_primal(p::P, t::T, unsafe::Bool) where {P,T<:Union{Tangent,MutableTangent}}
    Tt = tangent_type(P)
    if Tt != typeof(t)
        throw(ArgumentError("p of type $P has tangent_type $Tt, but t is of type $T"))
    end
    tmp = map(fieldnames(P)) do f
        tf = getfield(t.fields, f)
        isdefined(p, f) &&
            is_init(tf) &&
            return _add_to_primal(getfield(p, f), val(tf), unsafe)
        !isdefined(p, f) && !is_init(tf) && return FieldUndefined()
        throw(error("unable to handle undefined-ness"))
    end
    i = findfirst(==(FieldUndefined()), tmp)

    # If unsafe mode is enabled, then call `_new_` directly, and avoid the possibility that
    # the default inner constructor for `P` does not exist.
    if unsafe
        return i === nothing ? _new_(P, tmp...) : _new_(P, tmp[1:(i - 1)]...)
    end

    # If unsafe mode is disabled, try to use the default constructor for `P`. If this does
    # not work, then throw an informative error message.
    try
        return i === nothing ? P(tmp...) : P(tmp[1:(i - 1)]...)
    catch e
        if e isa MethodError
            throw(AddToPrimalException(P))
        else
            rethrow(e)
        end
    end
end

"""
    _diff(p::P, q::P) where {P}

Required for testing.

Computes the difference between `p` and `q`, which _must_ be of the same type, `P`.
Returns a tangent of type `tangent_type(P)`.
"""
function _diff(p::P, q::P) where {P}
    tangent_type(P) === NoTangent && return NoTangent()
    T = Tangent{NamedTuple{(),Tuple{}}}
    tangent_type(P) === T && return T((;))
    return _containerlike_diff(p, q)
end
_diff(p::P, q::P) where {P<:IEEEFloat} = p - q
function _diff(p::P, q::P) where {P<:SimpleVector}
    return Any[_diff(a, b) for (a, b) in zip(p, q)]
end
function _diff(p::P, q::P) where {P<:Union{Tuple,NamedTuple}}
    return tangent_type(P) == NoTangent ? NoTangent() : _map(_diff, p, q)
end

function _containerlike_diff(p::P, q::P) where {P}
    diffed_fields = map(fieldnames(P)) do f
        isdefined(p, f) && isdefined(q, f) && return _diff(getfield(p, f), getfield(q, f))
        !isdefined(p, f) && !isdefined(q, f) && return FieldUndefined()
        throw(error("Unhandleable undefinedness"))
    end
    i = findfirst(==(FieldUndefined()), diffed_fields)
    diffed_fields = i === nothing ? diffed_fields : diffed_fields[1:(i - 1)]
    return build_tangent(P, diffed_fields...)
end

"""
    increment_field!!(x::T, y::V, f) where {T, V}

`increment!!` the field `f` of `x` by `y`, and return the updated `x`.
"""
@inline @generated function increment_field!!(x::Tuple, y, ::Val{i}) where {i}
    exprs = map(n -> n == i ? :(increment!!(x[$n], y)) : :(x[$n]), fieldnames(x))
    return Expr(:tuple, exprs...)
end

# Optimal for homogeneously-typed Tuples with dynamic field choice.
function increment_field!!(x::Tuple, y, i::Int)
    return ntuple(n -> n == i ? increment!!(x[n], y) : x[n], length(x))
end

@inline @generated function increment_field!!(x::T, y, ::Val{f}) where {T<:NamedTuple,f}
    i = f isa Symbol ? findfirst(==(f), fieldnames(T)) : f
    new_fields = Expr(:call, increment_field!!, :(Tuple(x)), :y, :(Val($i)))
    return Expr(:call, T, new_fields)
end

# Optimal for homogeneously-typed NamedTuples with dynamic field choice.
function increment_field!!(x::T, y, i::Int) where {T<:NamedTuple}
    return T(increment_field!!(Tuple(x), y, i))
end
function increment_field!!(x::T, y, s::Symbol) where {T<:NamedTuple}
    return T(tuple_map(n -> n == s ? increment!!(x[n], y) : x[n], fieldnames(T)))
end

function increment_field!!(x::Tangent{T}, y, f::Val{F}) where {T,F}
    y isa NoTangent && return x
    new_val = fieldtype(T, F) <: PossiblyUninitTangent ? fieldtype(T, F)(y) : y
    return Tangent(increment_field!!(x.fields, new_val, f))
end
function increment_field!!(x::MutableTangent{T}, y, f::V) where {T,F,V<:Val{F}}
    y isa NoTangent && return x
    new_val = fieldtype(T, F) <: PossiblyUninitTangent ? fieldtype(T, F)(y) : y
    setfield!(x, :fields, increment_field!!(x.fields, new_val, f))
    return x
end

increment_field!!(x, y, f::Symbol) = increment_field!!(x, y, Val(f))
increment_field!!(x, y, n::Int) = increment_field!!(x, y, Val(n))

# Fallback method for when a tangent type for a struct is declared to be `NoTangent`.
for T in [Symbol, Int, Val]
    @eval increment_field!!(::NoTangent, ::NoTangent, f::Union{$T}) = NoTangent()
end

"""
    tangent_test_cases()

Constructs a `Vector` of `Tuple`s containing test cases for the tangent infrastructure.

If the returned tuple has 2 elements, the elements should be interpreted as follows:
1 - interface_only
2 - primal value

interface_only is a Bool which will be used to determine which subset of tests to run.

If the returned tuple has 5 elements, then the elements are interpreted as follows:
1 - interface_only
2 - primal value
3, 4, 5 - tangents, where <5> == increment!!(<3>, <4>).

Test cases in the first format make use of `zero_tangent` / `randn_tangent` etc to generate
tangents, but they're unable to check that `increment!!` is correct in an absolute sense.
"""
function tangent_test_cases()
    N_large = 33
    _names = Tuple(map(n -> Symbol("x$n"), 1:N_large))

    abs_test_cases = vcat(
        [
            (sin, NoTangent(), NoTangent(), NoTangent()),
            (map(Float16, (5.0, 4.0, 3.1, 7.1))...),
            (5.0f0, 4.0f0, 3.0f0, 7.0f0),
            (5.1, 4.0, 3.0, 7.0),
            (svec(5.0), Any[4.0], Any[3.0], Any[7.0]),
            ([3.0, 2.0], [1.0, 2.0], [2.0, 3.0], [3.0, 5.0]),
            (Float64[], Float64[], Float64[], Float64[]),
            (
                [1, 2],
                [NoTangent(), NoTangent()],
                [NoTangent(), NoTangent()],
                [NoTangent(), NoTangent()],
            ),
            (
                [[1.0], [1.0, 2.0]],
                [[2.0], [2.0, 3.0]],
                [[3.0], [4.0, 5.0]],
                [[5.0], [6.0, 8.0]],
            ),
            (
                setindex!(Vector{Vector{Float64}}(undef, 2), [1.0], 1),
                setindex!(Vector{Vector{Float64}}(undef, 2), [2.0], 1),
                setindex!(Vector{Vector{Float64}}(undef, 2), [3.0], 1),
                setindex!(Vector{Vector{Float64}}(undef, 2), [5.0], 1),
            ),
            (
                setindex!(Vector{Vector{Float64}}(undef, 2), [1.0], 2),
                setindex!(Vector{Vector{Float64}}(undef, 2), [2.0], 2),
                setindex!(Vector{Vector{Float64}}(undef, 2), [3.0], 2),
                setindex!(Vector{Vector{Float64}}(undef, 2), [5.0], 2),
            ),
            ((6.0, [1.0, 2.0]), (5.0, [3.0, 4.0]), (4.0, [4.0, 3.0]), (9.0, [7.0, 7.0])),
            ((), NoTangent(), NoTangent(), NoTangent()),
            ((1,), NoTangent(), NoTangent(), NoTangent()),
            ((2, 3), NoTangent(), NoTangent(), NoTangent()),
            (
                Mooncake.tuple_fill(5.0, Val(N_large)),
                Mooncake.tuple_fill(6.0, Val(N_large)),
                Mooncake.tuple_fill(7.0, Val(N_large)),
                Mooncake.tuple_fill(13.0, Val(N_large)),
            ),
            (
                (a=6.0, b=[1.0, 2.0]),
                (a=5.0, b=[3.0, 4.0]),
                (a=4.0, b=[4.0, 3.0]),
                (a=9.0, b=[7.0, 7.0]),
            ),
            ((;), NoTangent(), NoTangent(), NoTangent()),
            (
                NamedTuple{_names}(Mooncake.tuple_fill(5.0, Val(N_large))),
                NamedTuple{_names}(Mooncake.tuple_fill(6.0, Val(N_large))),
                NamedTuple{_names}(Mooncake.tuple_fill(7.0, Val(N_large))),
                NamedTuple{_names}(Mooncake.tuple_fill(13.0, Val(N_large))),
            ),
            (
                TestResources.TypeStableMutableStruct{Float64}(5.0, 3.0),
                build_tangent(TestResources.TypeStableMutableStruct{Float64}, 5.0, 4.0),
                build_tangent(TestResources.TypeStableMutableStruct{Float64}, 3.0, 3.0),
                build_tangent(TestResources.TypeStableMutableStruct{Float64}, 8.0, 7.0),
            ),
            ( # complete init
                TestResources.StructFoo(6.0, [1.0, 2.0]),
                build_tangent(TestResources.StructFoo, 5.0, [3.0, 4.0]),
                build_tangent(TestResources.StructFoo, 3.0, [2.0, 1.0]),
                build_tangent(TestResources.StructFoo, 8.0, [5.0, 5.0]),
            ),
            ( # partial init
                TestResources.StructFoo(6.0),
                build_tangent(TestResources.StructFoo, 5.0),
                build_tangent(TestResources.StructFoo, 4.0),
                build_tangent(TestResources.StructFoo, 9.0),
            ),
            ( # complete init
                TestResources.MutableFoo(6.0, [1.0, 2.0]),
                build_tangent(TestResources.MutableFoo, 5.0, [3.0, 4.0]),
                build_tangent(TestResources.MutableFoo, 3.0, [2.0, 1.0]),
                build_tangent(TestResources.MutableFoo, 8.0, [5.0, 5.0]),
            ),
            ( # partial init
                TestResources.MutableFoo(6.0),
                build_tangent(TestResources.MutableFoo, 5.0),
                build_tangent(TestResources.MutableFoo, 4.0),
                build_tangent(TestResources.MutableFoo, 9.0),
            ),
            (
                TestResources.StructNoFwds(5.0),
                build_tangent(TestResources.StructNoFwds, 5.0),
                build_tangent(TestResources.StructNoFwds, 4.0),
                build_tangent(TestResources.StructNoFwds, 9.0),
            ),
            (
                TestResources.StructNoRvs([5.0]),
                build_tangent(TestResources.StructNoRvs, [5.0]),
                build_tangent(TestResources.StructNoRvs, [4.0]),
                build_tangent(TestResources.StructNoRvs, [9.0]),
            ),
            (UnitRange{Int}(5, 7), NoTangent(), NoTangent(), NoTangent()),
        ],
        map([
            LowerTriangular{Float64,Matrix{Float64}},
            UpperTriangular{Float64,Matrix{Float64}},
            UnitLowerTriangular{Float64,Matrix{Float64}},
            UnitUpperTriangular{Float64,Matrix{Float64}},
        ]) do T
            return (
                T(randn(2, 2)),
                build_tangent(T, [1.0 2.0; 3.0 4.0]),
                build_tangent(T, [2.0 1.0; 5.0 4.0]),
                build_tangent(T, [3.0 3.0; 8.0 8.0]),
            )
        end,
        [
            (p, NoTangent(), NoTangent(), NoTangent()) for
            p in [Array, Float64, Union{Float64,Float32}, Union, UnionAll, typeof(<:)]
        ],
    )
    rel_test_cases = Any[
        (2.0, 3),
        (3, 2.0),
        (2.0, 1.0),
        (randn(10), 3),
        (3, randn(10)),
        (randn(10), randn(10)),
        (a=2.0, b=3),
        (a=3, b=2.0),
        (a=randn(10), b=3),
        (a=3, b=randn(10)),
        (a=randn(10), b=randn(10)),
        (Base.TOML.ErrorType(1), NoTangent()), # Enum
        # Regression tests to catch type inference failures, see https://github.com/compintell/Mooncake.jl/pull/422
        (((((randn(33)...,),),),),),
        (((((((((randn(33)...,),),),),), randn(5)...),),),),
        Base.OneTo{Int},
    ]
    VERSION >= v"1.11" && push!(rel_test_cases, fill!(Memory{Float64}(undef, 3), 3.0))
    return vcat(
        map(x -> (false, x...), abs_test_cases),
        map(x -> (false, x), rel_test_cases),
        map(Mooncake.TestTypes.instantiate, Mooncake.TestTypes.PRIMALS),
    )
end
