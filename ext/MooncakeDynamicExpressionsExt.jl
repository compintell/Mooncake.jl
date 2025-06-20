module MooncakeDynamicExpressionsExt

using DynamicExpressions:
    DynamicExpressions as DE,
    AbstractExpressionNode,
    Expression,
    Nullable,
    constructorof,
    branch_copy,
    leaf_copy,
    get_child,
    get_children,
    set_children!
using Mooncake
using Mooncake: NoTangent
using Random: AbstractRNG

################################################################################
# Main tangent type
################################################################################

mutable struct TangentNode{Tv,D}
    const degree::UInt8
    val::Union{Tv,NoTangent}
    children::NTuple{D,Union{@NamedTuple{null::NoTangent,x::TangentNode{Tv,D}},NoTangent}}
end

function TangentNode{Tv,D}(
    val_tan::Union{Tv,NoTangent}, children::Vararg{Union{TangentNode{Tv,D},NoTangent},deg}
) where {Tv,D,deg}
    return TangentNode{Tv,D}(
        UInt8(deg),
        val_tan,
        ntuple(i -> i <= deg ? _wrap_nullable(children[i]) : NoTangent(), Val(D)),
    )
end

function Mooncake.tangent_type(::Type{<:AbstractExpressionNode{T,D}}) where {T,D}
    Tv = Mooncake.tangent_type(T)
    return Tv === NoTangent ? NoTangent : TangentNode{Tv,D}
end
function Mooncake.tangent_type(::Type{Nullable{<:AbstractExpressionNode{T,D}}}) where {T,D}
    Tv = Mooncake.tangent_type(T)
    return Union{@NamedTuple{null::NoTangent,x::TangentNode{Tv,D}},NoTangent}
end
function Mooncake.tangent_type(::Type{TangentNode{Tv,D}}) where {Tv,D}
    return TangentNode{Tv,D}
end
function Mooncake.tangent_type(
    ::Type{TangentNode{Tv,D}}, ::Type{Mooncake.NoRData}
) where {Tv,D}
    return TangentNode{Tv,D}
end
function Mooncake.tangent(t::TangentNode, ::Mooncake.NoRData)
    return t
end
function Mooncake.rdata(::TangentNode)
    return Mooncake.NoRData()
end

_unwrap_nullable(c::NoTangent) = c
_unwrap_nullable(c::NamedTuple{(:null, :x)}) = c.x
_wrap_nullable(c::NoTangent) = c
_wrap_nullable(c::TangentNode) = (; null=NoTangent(), x=c)

function DE.get_child(t::TangentNode, i::Int)
    return _unwrap_nullable(t.children[i])
end
function _get_child(t, ::Val{i}) where {i}
    return get_child(t, i)
end
function DE.get_children(t::TangentNode, ::Val{d}) where {d}
    return ntuple(i -> _unwrap_nullable(t.children[i]), Val(d))
end
function DE.set_children!(
    t::TangentNode{Tv,D},
    children::Tuple{
        Union{TangentNode{Tv,D},NoTangent},
        Vararg{Union{TangentNode{Tv,D},NoTangent},deg_m_1},
    },
) where {Tv,D,deg_m_1}
    deg = deg_m_1 + 1
    #! format: off
    t.children = ntuple(
        i -> i <= deg ? _wrap_nullable(children[i]) : NoTangent(), Val(D)
    )
    #! format: on
end
function DE.extract_gradient(
    gradient::Mooncake.Tangent{@NamedTuple{tree::TN, metadata::Mooncake.NoTangent}},
    tree::Expression{T},
) where {Tv,TN<:TangentNode{Tv},T}
    return DE.extract_gradient(gradient.fields.tree, DE.get_tree(tree))
end
function DE.extract_gradient(
    dtree::TangentNode{Tv,D}, tree::AbstractExpressionNode{T,D}
) where {Tv,D,T}
    num_constants = count(t -> t.degree == 0 && t.constant, tree)
    ar = Vector{Tv}(undef, num_constants)
    _extract_gradient!(ar, dtree, tree)
    return ar
end
@generated function _extract_gradient!(
    ar, gradient::TangentNode{Tv,D}, tree::AbstractExpressionNode{T,D}, idx=firstindex(ar)
) where {Tv,D,T}
    quote
        if tree.degree == 0
            if tree.constant
                ar[idx] = gradient.val::Tv
                idx = nextind(ar, idx)
            end
        else
            deg = tree.degree
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> Base.Cartesian.@nexprs(
                    i,
                    j -> begin
                        idx = _extract_gradient!(
                            ar,
                            DE.get_child(gradient, j),
                            DE.get_child(tree, j),
                            idx,
                        )
                    end
                )
            )
        end
        return idx
    end
end

################################################################################
# zero_tangent / randn_tangent
################################################################################

struct InitHelper{F,ARGS<:Tuple,M<:Mooncake.MaybeCache}
    f::F
    args::ARGS
    dict::M
end
function (helper::InitHelper)(p::N) where {T,D,N<:AbstractExpressionNode{T,D}}
    Tv = Mooncake.tangent_type(T)
    Tv === NoTangent && return NoTangent()
    return get!(helper.dict, p) do
        helper_inner(helper, p)
    end::TangentNode{Tv,D}
end
@generated function helper_inner(
    helper::InitHelper, p::N
) where {T,D,N<:AbstractExpressionNode{T,D}}
    quote
        Tv = Mooncake.tangent_type(T)
        deg = p.degree
        if deg == 0
            if p.constant
                TangentNode{Tv,D}(helper_call(helper, p.val))
            else
                TangentNode{Tv,D}(NoTangent())
            end
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i ->
                    TangentNode{Tv,D}(NoTangent(), map(helper, get_children(p, Val(i)))...),
            )
        end
    end
end
function helper_call(helper::InitHelper, val)
    return helper.f(helper.args..., val, helper.dict)
end

function Mooncake.zero_tangent_internal(
    p::N, dict::Mooncake.MaybeCache
) where {T,N<:AbstractExpressionNode{T}}
    return InitHelper(Mooncake.zero_tangent_internal, (), dict)(p)
end
function Mooncake.randn_tangent_internal(
    rng::AbstractRNG, p::N, dict::Mooncake.MaybeCache
) where {T,N<:AbstractExpressionNode{T}}
    return InitHelper(Mooncake.randn_tangent_internal, (rng,), dict)(p)
end

################################################################################
# In‑place mutation helpers
################################################################################

struct IncrementHelper{F,C<:Mooncake.IncCache}
    f::F
    cache::C
end
@generated function (helper::IncrementHelper)(t::TangentNode{Tv,D}, s...) where {Tv,D}
    quote
        if haskey(helper.cache, t) || (!isempty(s) && t === first(s))
            return t
        end
        helper.cache[t] = true
        ts = (t, s...)
        deg = t.degree
        if deg == 0
            t.val = helper_call(helper, ts...)
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> set_children!(
                    t,
                    Base.Cartesian.@ntuple(
                        i, c -> helper(map(Base.Fix2(_get_child, Val(c)), ts)...),
                    )
                )
            )
        end
        return t
    end
end
function helper_call(helper::IncrementHelper, t, s...)
    return helper.f(helper.cache, t.val, map(ti -> ti.val, s)...)
end

function Mooncake.increment_internal!!(c::Mooncake.IncCache, t::TangentNode, s::TangentNode)
    return IncrementHelper(Mooncake.increment_internal!!, c)(t, s)
end
function Mooncake.set_to_zero_internal!!(c::Mooncake.IncCache, t::TangentNode)
    return IncrementHelper(Mooncake.set_to_zero_internal!!, c)(t)
end

################################################################################
# Algebraic helpers (_dot / _scale / _add_to_primal / _diff)
################################################################################

Mooncake._dot_internal(c::Mooncake.MaybeCache, t::NoTangent, s::TangentNode) = 0.0
Mooncake._dot_internal(c::Mooncake.MaybeCache, t::TangentNode, s::NoTangent) = 0.0
@generated function Mooncake._dot_internal(
    c::Mooncake.MaybeCache, t::TangentNode{Tv,D}, s::TangentNode{Tv,D}
) where {Tv,D}
    quote
        key = (t, s)
        haskey(c, key) && return c[key]::Float64
        c[key] = 0.0
        deg = t.degree
        res = if deg == 0
            if (t.val isa NoTangent || s.val isa NoTangent)
                0.0
            else
                Mooncake._dot_internal(c, t.val, s.val)
            end
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> Base.Cartesian.@ncall(
                    i,
                    +,
                    j -> Mooncake._dot_internal(c, get_child(t, j), get_child(s, j))
                )
            )
        end
        c[key] = res
        return res
    end
end

function Mooncake._scale_internal(
    c::Mooncake.MaybeCache, a::Number, t::TangentNode{Tv,D}
) where {Tv,D}
    return get!(c, t) do
        _scale_internal_helper(c, a, t)
    end::TangentNode{Tv,D}
end
@generated function _scale_internal_helper(
    c::Mooncake.MaybeCache, a::Number, t::TangentNode{Tv,D}
) where {Tv,D}
    quote
        deg = t.degree
        if deg == 0
            TangentNode{Tv,D}(Mooncake._scale_internal(c, a, t.val))
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> Base.Cartesian.@ncall(
                    i,
                    TangentNode{Tv,D},
                    NoTangent(),
                    j -> Mooncake._scale_internal(c, a, get_child(t, j))
                )
            )
        end
    end
end

function Mooncake._add_to_primal_internal(
    c::Mooncake.MaybeCache, p::N, t::TangentNode{Tv,D}, unsafe::Bool
) where {T,D,N<:AbstractExpressionNode{T,D},Tv}
    key = (p, t, unsafe)
    return get!(c, key) do
        _add_to_primal_internal_helper(c, p, t, unsafe)
    end::N
end
@generated function _add_to_primal_internal_helper(
    c::Mooncake.MaybeCache, p::N, t::TangentNode{Tv,D}, unsafe::Bool
) where {T,D,N<:AbstractExpressionNode{T,D},Tv}
    quote
        deg = p.degree
        if deg == 0
            new_leaf = leaf_copy(p)
            if p.constant
                new_leaf.val = Mooncake._add_to_primal_internal(c, p.val, t.val, unsafe)
            end
            new_leaf
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> Base.Cartesian.@ncall(
                    i,
                    branch_copy,
                    p,
                    j -> Mooncake._add_to_primal_internal(
                        c, get_child(p, j), get_child(t, j), unsafe
                    )
                )
            )
        end
    end
end

function Mooncake._diff_internal(
    c::Mooncake.MaybeCache, p::N, q::N
) where {T,D,N<:AbstractExpressionNode{T,D}}
    Tv = Mooncake.tangent_type(T)
    Tv === NoTangent && return NoTangent()
    key = (p, q)
    return get!(c, key) do
        _diff_internal_helper(c, p, q)
    end::TangentNode{Tv,D}
end

@generated function _diff_internal_helper(
    c::Mooncake.MaybeCache, p::N, q::N
) where {T,D,N<:AbstractExpressionNode{T,D}}
    quote
        Tv = Mooncake.tangent_type(T)
        deg = p.degree
        if p.degree == 0
            if p.constant
                TangentNode{Tv,D}(Mooncake._diff_internal(c, p.val, q.val))
            else
                TangentNode{Tv,D}(NoTangent())
            end
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> Base.Cartesian.@ncall(
                    i,
                    TangentNode{Tv,D},
                    NoTangent(),
                    j -> Mooncake._diff_internal(c, get_child(p, j), get_child(q, j))
                )
            )
        end
    end
end

################################################################################
# getfield / lgetfield rrules
################################################################################

@inline _field_sym(x::Symbol) = x
@inline _field_sym(i::Int) =
    if i == 1
        :degree
    elseif i == 2
        :val
    elseif i == 3
        :children
    else
        :non_differentiable
    end
@inline _field_sym(::Type{Val{F}}) where {F} = _field_sym(F)
@inline _field_sym(::Val{F}) where {F} = _field_sym(F)

struct Pullback{T,field_sym,n_args}
    pt::T
end
function (pb::Pullback{T,field_sym,n_args})(Δy_rdata) where {T,field_sym,n_args}
    if field_sym === :val && !(Δy_rdata isa Mooncake.NoRData)
        pb.pt.val = Mooncake.increment_rdata!!(pb.pt.val, Δy_rdata)
    end
    return ntuple(_ -> Mooncake.NoRData(), Val(n_args))
end

function _rrule_getfield_common(
    obj_cd::Mooncake.CoDual{N,TangentNode{Tv,D}}, ::Val{field_sym}, ::Val{n_args}
) where {T,D,N<:AbstractExpressionNode{T,D},Tv,field_sym,n_args}
    p = Mooncake.primal(obj_cd)
    pt = Mooncake.tangent(obj_cd)

    value_primal = getfield(p, field_sym)

    fdata_for_output = if field_sym === :val
        Mooncake.fdata(pt.val)
    elseif field_sym === :children
        map(value_primal, pt.children) do child_p, child_t
            if child_t isa Mooncake.NoTangent
                Mooncake.uninit_fdata(child_p)
            else
                Mooncake.FData(Mooncake.fdata(child_t))
            end
        end
    else
        Mooncake.NoFData()
    end
    y_cd = Mooncake.CoDual(value_primal, fdata_for_output)
    return y_cd, Pullback{typeof(pt),field_sym,n_args}(pt)
end

# lgetfield(AEN, Val{field})
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{
    typeof(Mooncake.lgetfield),AbstractExpressionNode,Val
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake.lgetfield)},
    obj_cd::Mooncake.CoDual{N,TangentNode{Tv,D}},
    vfield_cd::Mooncake.CoDual{Val{F},Mooncake.NoFData},
) where {T,D,N<:AbstractExpressionNode{T,D},Tv,F}
    return _rrule_getfield_common(obj_cd, Val(_field_sym(F)), Val(3))
end

# getfield by Symbol
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{
    typeof(getfield),AbstractExpressionNode,Symbol
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(getfield)},
    obj_cd::Mooncake.CoDual{N,TangentNode{Tv,D}},
    sym_cd::Mooncake.CoDual{Symbol,Mooncake.NoFData},
) where {T,D,N<:AbstractExpressionNode{T,D},Tv}
    return _rrule_getfield_common(obj_cd, Val(Mooncake.primal(sym_cd)), Val(3))
end

# getfield by Int
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{
    typeof(getfield),AbstractExpressionNode,Int
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(getfield)},
    obj_cd::Mooncake.CoDual{N,TangentNode{Tv,D}},
    idx_cd::Mooncake.CoDual{Int,Mooncake.NoFData},
) where {T,D,N<:AbstractExpressionNode{T,D},Tv}
    return _rrule_getfield_common(obj_cd, _field_sym(Mooncake.primal(idx_cd)), 3)
end

################################################################################
# Test‑utility helpers
################################################################################

@generated function Mooncake.TestUtils.populate_address_map_internal(
    m::Mooncake.TestUtils.AddressMap, p::N, t::TangentNode{Tv,D}
) where {T,D,N<:AbstractExpressionNode{T,D},Tv}
    quote
        kp = Base.pointer_from_objref(p)
        kt = Base.pointer_from_objref(t)
        !haskey(m, kp) && (m[kp] = kt)
        deg = p.degree
        if deg == 0
            if p.constant
                Mooncake.TestUtils.populate_address_map_internal(m, p.val, t.val)
            end
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> Base.Cartesian.@nexprs(
                    i,
                    j -> Mooncake.TestUtils.populate_address_map_internal(
                        m, get_child(p, j), get_child(t, j)
                    )
                )
            )
        end
        return m
    end
end

function Mooncake.TestUtils.has_equal_data_internal(
    x::N, y::N, equndef::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {T,N<:AbstractExpressionNode{T}}
    idp = (objectid(x), objectid(y))
    # Just use regular `AbstractExpressionNode` Base.:(==)
    return get!(() -> x == y, d, idp)
end

function Mooncake.TestUtils.has_equal_data_internal(
    t::TangentNode{Tv,D},
    s::TangentNode{Tv,D},
    equndef::Bool,
    d::Dict{Tuple{UInt,UInt},Bool},
) where {Tv,D}
    idp = (objectid(t), objectid(s))
    return get!(d, idp) do
        _has_equal_data_internal_helper(t, s, equndef, d)
    end
end
@generated function _has_equal_data_internal_helper(
    t::TangentNode{Tv,D},
    s::TangentNode{Tv,D},
    equndef::Bool,
    d::Dict{Tuple{UInt,UInt},Bool},
) where {Tv,D}
    quote
        deg = t.degree
        deg == s.degree && if t.degree == 0
            Mooncake.TestUtils.has_equal_data_internal(t.val, s.val, equndef, d)
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> Base.Cartesian.@nall(
                    i,
                    j -> Mooncake.TestUtils.has_equal_data_internal(
                        get_child(t, j), get_child(s, j), equndef, d
                    )
                )
            )
        end
    end
end

end
