frule!!(f::F, args::Vararg{Dual,N}) where {F,N} = frule!!(zero_dual(f), args...)

@is_primitive MinimalCtx Tuple{typeof(sin),Number}
function frule!!(::Dual{typeof(sin)}, x::Dual{<:Number})
    return Dual(sin(primal(x)), cos(primal(x)) * tangent(x))
end

@is_primitive MinimalCtx Tuple{typeof(cos),Number}
function frule!!(::Dual{typeof(cos)}, x::Dual{<:Number})
    return Dual(cos(primal(x)), -sin(primal(x)) * tangent(x))
end
