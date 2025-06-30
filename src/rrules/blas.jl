function blas_name(name::Symbol)
    return (BLAS.USE_BLAS64 ? Symbol(name, "64_") : name, Symbol(BLAS.libblastrampoline))
end

function _trans(flag, mat)
    flag === 'T' && return transpose(mat)
    flag === 'C' && return adjoint(mat)
    flag === 'N' && return mat
    throw(error("Unrecognised flag $flag"))
end

function tri!(A, u::Char, d::Char)
    return u == 'L' ? tril!(A, d == 'U' ? -1 : 0) : triu!(A, d == 'U' ? 1 : 0)
end

const BlasRealFloat = Union{Float32,Float64}
const BlasComplexFloat = Union{ComplexF32,ComplexF64}

"""
    arrayify(x::CoDual{<:AbstractArray{<:BlasFloat}})

Return the primal field of `x`, and convert its fdata into an array of the same type as the
primal. This operation is not guaranteed to be possible for all array types, but seems to be
possible for all array types of interest so far.
"""
function arrayify(x::Union{Dual{A},CoDual{A}}) where {A<:AbstractArray{<:BlasFloat}}
    return arrayify(primal(x), tangent(x))  # NOTE: for complex number, the tangent is a reinterpreted version of the primal
end
arrayify(x::Array{P}, dx::Array{P}) where {P<:BlasRealFloat} = (x, dx)
function arrayify(x::Array{P}, dx::Array{<:Tangent}) where {P<:BlasComplexFloat}
    return x, reinterpret(P, dx)
end
function arrayify(x::A, dx::FData) where {A<:SubArray{<:BlasRealFloat}}
    _, _dx = arrayify(x.parent, dx.data.parent)
    return x, A(_dx, x.indices, x.offset1, x.stride1)
end
function arrayify(x::A, dx::Tangent) where {A<:SubArray{<:BlasRealFloat}}
    _, _dx = arrayify(x.parent, dx.fields.parent)
    return x, A(_dx, x.indices, x.offset1, x.stride1)
end
function arrayify(x::A, dx::FData) where {A<:Base.ReshapedArray{<:BlasRealFloat}}
    _, _dx = arrayify(x.parent, dx.data.parent)
    return x, A(_dx, x.dims, x.mi)
end
function arrayify(x::A, dx::Tangent) where {A<:Base.ReshapedArray{<:BlasRealFloat}}
    _, _dx = arrayify(x.parent, dx.fields.parent)
    return x, A(_dx, x.dims, x.mi)
end
function arrayify(x::Base.ReinterpretArray{T}, dx::FData) where {T<:BlasFloat}
    _, _dx = arrayify(x.parent, dx.data.parent)
    return x, reinterpret(T, _dx)
end

function arrayify(x::A, dx::DA) where {A,DA}
    msg =
        "Encountered unexpected array type in `Mooncake.arrayify`. This error is likely " *
        "due to a call to a BLAS or LAPACK function with an array type that " *
        "Mooncake has not been told about. A new method of `Mooncake.arrayify` is needed." *
        " Please open an issue at " *
        "https://github.com/chalk-lab/Mooncake.jl/issues . " *
        "It should contain this error message and the associated stack trace.\n\n" *
        "Array type: $A\n\nFData type: $DA."
    return error(msg)
end

#
# Utility
#

@zero_derivative MinimalCtx Tuple{typeof(BLAS.get_num_threads)}
@zero_derivative MinimalCtx Tuple{typeof(BLAS.lbt_get_num_threads)}
@zero_derivative MinimalCtx Tuple{typeof(BLAS.set_num_threads),Union{Integer,Nothing}}
@zero_derivative MinimalCtx Tuple{typeof(BLAS.lbt_set_num_threads),Any}

#
# LEVEL 1
#

for (fname, elty) in ((:cblas_ddot, :Float64), (:cblas_sdot, :Float32))
    @eval @inline function frule!!(
        ::Dual{typeof(_foreigncall_)},
        ::Dual{Val{$(blas_name(fname))}},
        ::Dual, # return type
        ::Dual, # argument types
        ::Dual, # nreq
        ::Dual, # calling convention
        _n::Dual{BLAS.BlasInt},
        _DX::Dual{Ptr{$elty}},
        _incx::Dual{BLAS.BlasInt},
        _DY::Dual{Ptr{$elty}},
        _incy::Dual{BLAS.BlasInt},
        args::Vararg{Any,N},
    ) where {N}
        GC.@preserve args begin
            # Load in values from pointers.
            n, incx, incy = map(primal, (_n, _incx, _incy))
            xinds = 1:incx:(incx * n)
            yinds = 1:incy:(incy * n)
            DX = view(unsafe_wrap(Vector{$elty}, primal(_DX), n * incx), xinds)
            DY = view(unsafe_wrap(Vector{$elty}, primal(_DY), n * incy), yinds)

            _dDX = view(unsafe_wrap(Vector{$elty}, tangent(_DX), n * incx), xinds)
            _dDY = view(unsafe_wrap(Vector{$elty}, tangent(_DY), n * incy), yinds)

            return Dual(dot(DX, DY), dot(DX, _dDY) + dot(_dDX, DY))
        end
    end
    @eval @inline function rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{$(blas_name(fname))}},
        ::CoDual, # return type
        ::CoDual, # argument types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        _n::CoDual{BLAS.BlasInt},
        _DX::CoDual{Ptr{$elty}},
        _incx::CoDual{BLAS.BlasInt},
        _DY::CoDual{Ptr{$elty}},
        _incy::CoDual{BLAS.BlasInt},
        args::Vararg{Any,N},
    ) where {N}
        GC.@preserve args begin
            # Load in values from pointers.
            n, incx, incy = map(primal, (_n, _incx, _incy))
            xinds = 1:incx:(incx * n)
            yinds = 1:incy:(incy * n)
            DX = view(unsafe_wrap(Vector{$elty}, primal(_DX), n * incx), xinds)
            DY = view(unsafe_wrap(Vector{$elty}, primal(_DY), n * incy), yinds)

            _dDX = view(unsafe_wrap(Vector{$elty}, tangent(_DX), n * incx), xinds)
            _dDY = view(unsafe_wrap(Vector{$elty}, tangent(_DY), n * incy), yinds)

            out = dot(DX, DY)
        end

        function ddot_pb!!(dv)
            GC.@preserve args begin
                _dDX .+= DY .* dv
                _dDY .+= DX .* dv
            end
            return tuple_fill(NoRData(), Val(N + 11))
        end

        # Run primal computation.
        return zero_fcodual(out), ddot_pb!!
    end
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.nrm2),Int,X,Int
    } where {T<:BlasFloat,X<:Union{Ptr{T},AbstractArray{T}}},
)
function frule!!(
    ::Dual{typeof(BLAS.nrm2)},
    n::Dual{<:Integer},
    X_dX::Dual{<:Union{Ptr{T},AbstractArray{T}}},
    incx::Dual{<:Integer},
) where {T<:BlasFloat}
    X, dX = arrayify(X_dX)
    y = BLAS.nrm2(primal(n), X, primal(incx))
    dy = zero(y)
    @inbounds for i in 1:primal(incx):(primal(n) * primal(incx))
        dy = dy + real(X[i] * dX[i]') + real(X[i]' * dX[i])
    end
    return Dual(y, dy / 2y)
end
function rrule!!(
    ::CoDual{typeof(BLAS.nrm2)},
    n::CoDual{<:Integer},
    X_dX::CoDual{<:Union{Ptr{T},AbstractArray{T}} where {T<:BlasFloat}},
    incx::CoDual{<:Integer},
)
    X, dX = arrayify(X_dX)
    y = BLAS.nrm2(n.x, X, incx.x)
    function nrm2_pb!!(dy)
        view(dX, 1:(incx.x):(incx.x * n.x)) .+=
            view(X, 1:(incx.x):(incx.x * n.x)) .* (dy / y)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, NoFData()), nrm2_pb!!
end

@is_primitive(
    MinimalCtx,
    Tuple{typeof(BLAS.nrm2),X} where {T<:BlasFloat,X<:Union{Ptr{T},AbstractArray{T}}},
)
function frule!!(
    ::Dual{typeof(BLAS.nrm2)},
    X_dX::Dual{<:Union{Ptr{T},AbstractArray{T}} where {T<:BlasFloat}},
)
    X, dX = arrayify(X_dX)
    y = BLAS.nrm2(X)
    dy = zero(y)
    @inbounds for i in eachindex(X)
        dy = dy + real(X[i] * dX[i]') + real(X[i]' * dX[i])
    end
    return Dual(y, dy / (2y))
end
function rrule!!(
    ::CoDual{typeof(BLAS.nrm2)},
    X_dX::CoDual{<:Union{Ptr{T},AbstractArray{T}} where {T<:BlasFloat}},
)
    X, dX = arrayify(X_dX)
    y = BLAS.nrm2(X)
    function nrm2_pb!!(dy)
        dX .+= X .* (dy / y)
        return NoRData(), NoRData()
    end
    return CoDual(y, NoFData()), nrm2_pb!!
end

@is_primitive(
    MinimalCtx,
    Tuple{typeof(BLAS.scal!),Integer,P,AbstractArray{P},Integer} where {P<:BlasRealFloat}
)
function frule!!(
    ::Dual{typeof(BLAS.scal!)},
    _n::Dual{<:Integer},
    a_da::Dual{P},
    X_dX::Dual{<:AbstractArray{P}},
    _incx::Dual{<:Integer},
) where {P<:BlasRealFloat}

    # Extract params.
    n = primal(_n)
    incx = primal(_incx)
    a, da = extract(a_da)
    X, dX = arrayify(X_dX)

    # Compute Frechet derivative.
    BLAS.scal!(n, a, dX, incx)
    BLAS.axpy!(n, da, X, incx, dX, incx)

    # Perform primal computation.
    BLAS.scal!(n, a, X, incx)
    return X_dX
end
function rrule!!(
    ::CoDual{typeof(BLAS.scal!)},
    _n::CoDual{<:Integer},
    a_da::CoDual{P},
    X_dX::CoDual{<:AbstractArray{P}},
    _incx::CoDual{<:Integer},
) where {P<:BlasRealFloat}

    # Extract params.
    n = primal(_n)
    incx = primal(_incx)
    a = primal(a_da)
    X, dX = arrayify(X_dX)

    # Take a copy of previous state in order to recover it on the reverse pass.
    X_copy = copy(X)
    dX_copy = copy(dX)

    # Run primal computation.
    BLAS.scal!(n, a, X, incx)

    function scal_adjoint(::NoRData)

        # Set primal to previous state.
        X .= X_copy

        # Compute gradient w.r.t. scaling.
        ∇a = BLAS.dot(n, X, incx, dX, incx)

        # Compute gradient w.r.t. DX.
        BLAS.scal!(n, a, dX, incx)
        BLAS.axpy!(n, one(P), dX, incx, dX_copy, incx)

        return NoRData(), NoRData(), ∇a, NoRData(), NoRData()
    end
    return X_dX, scal_adjoint
end

#
# LEVEL 2
#

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.gemv!),Char,P,AbstractMatrix{P},AbstractVector{P},P,AbstractVector{P}
    } where {P<:BlasRealFloat},
)

@inline function frule!!(
    ::Dual{typeof(BLAS.gemv!)},
    tA::Dual{Char},
    alpha::Dual{P},
    A_dA::Dual{<:AbstractMatrix{P}},
    x_dx::Dual{<:AbstractVector{P}},
    beta::Dual{P},
    y_dy::Dual{<:AbstractVector{P}},
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    x, dx = arrayify(x_dx)
    y, dy = arrayify(y_dy)
    α, dα = extract(alpha)
    β, dβ = extract(beta)

    # Derivative computation.
    BLAS.gemv!(primal(tA), dα, A, x, β, dy)
    BLAS.gemv!(primal(tA), α, dA, x, one(P), dy)
    BLAS.gemv!(primal(tA), α, A, dx, one(P), dy)

    # Strong zero is essential here, in case `y` has undefined element values.
    if !iszero(dβ)
        @inbounds for n in eachindex(y)
            tmp = dβ * y[n]
            dy[n] = ifelse(isnan(y[n]), dy[n], tmp + dy[n])
        end
    end

    # Primal computation.
    BLAS.gemv!(primal(tA), α, A, x, β, y)

    return y_dy
end

@inline function rrule!!(
    ::CoDual{typeof(BLAS.gemv!)},
    _tA::CoDual{Char},
    _alpha::CoDual{P},
    _A::CoDual{<:AbstractMatrix{P}},
    _x::CoDual{<:AbstractVector{P}},
    _beta::CoDual{P},
    _y::CoDual{<:AbstractVector{P}},
) where {P<:BlasRealFloat}

    # Pull out primals and tangents (the latter only where necessary).
    trans = _tA.x
    alpha = _alpha.x
    A, dA = arrayify(_A)
    x, dx = arrayify(_x)
    beta = _beta.x
    y, dy = arrayify(_y)

    # Take copies before adding.
    y_copy = copy(y)

    # Run primal.
    BLAS.gemv!(trans, alpha, A, x, beta, y)

    function gemv!_pb!!(::NoRData)

        # Increment fdata.
        if trans == 'N'
            dalpha = dot(dy, A, x)
            dA .+= alpha .* dy .* x'
            BLAS.gemv!('T', alpha, A, dy, one(eltype(A)), dx)
        else
            dalpha = dot(dy, A', x)
            dA .+= alpha .* x .* dy'
            BLAS.gemv!('N', alpha, A, dy, one(eltype(A)), dx)
        end
        dbeta = dot(y_copy, dy)
        dy .*= beta

        # Restore primal.
        copyto!(y, y_copy)

        # Return rdata.
        return NoRData(), NoRData(), dalpha, NoRData(), NoRData(), dbeta, NoRData()
    end

    return _y, gemv!_pb!!
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.symv!),Char,T,AbstractMatrix{T},AbstractVector{T},T,AbstractVector{T}
    } where {T<:BlasRealFloat},
)

function frule!!(
    ::Dual{typeof(BLAS.symv!)},
    uplo::Dual{Char},
    alpha::Dual{T},
    A_dA::Dual{<:AbstractMatrix{T}},
    x_dx::Dual{<:AbstractVector{T}},
    beta::Dual{T},
    y_dy::Dual{<:AbstractVector{T}},
) where {T<:BlasRealFloat}
    # Extract primals.
    ul = primal(uplo)
    α = primal(alpha)
    β, dβ = extract(beta)
    A, dA = arrayify(A_dA)
    x, dx = arrayify(x_dx)
    y, dy = arrayify(y_dy)

    # Compute Frechet derivative.
    BLAS.symv!(ul, tangent(alpha), A, x, β, dy)
    BLAS.symv!(ul, α, dA, x, one(T), dy)
    BLAS.symv!(ul, α, A, dx, one(T), dy)
    if !iszero(dβ)
        @inbounds for n in eachindex(y)
            tmp = dβ * y[n]
            dy[n] = ifelse(isnan(y[n]), dy[n], tmp + dy[n])
        end
    end

    # Run primal computation.
    BLAS.symv!(ul, α, A, x, β, y)

    return y_dy
end

function rrule!!(
    ::CoDual{typeof(BLAS.symv!)},
    uplo::CoDual{Char},
    alpha::CoDual{T},
    A_dA::CoDual{<:AbstractMatrix{T}},
    x_dx::CoDual{<:AbstractVector{T}},
    beta::CoDual{T},
    y_dy::CoDual{<:AbstractVector{T}},
) where {T<:BlasRealFloat}

    # Extract primals.
    ul = primal(uplo)
    α = primal(alpha)
    β = primal(beta)
    A, dA = arrayify(A_dA)
    x, dx = arrayify(x_dx)
    y, dy = arrayify(y_dy)

    # In this rule we optimise carefully for the special case a == 1 && b == 0, which
    # corresponds to simply multiplying symm(A) and x together, and writing the result to y.
    # This is an extremely common edge case, so it's important to do well for it.
    y_copy = copy(y)
    tmp_ref = Ref{Vector{T}}()
    if (α == 1 && β == 0)
        BLAS.symv!(ul, α, A, x, β, y)
    else
        tmp = BLAS.symv(ul, one(T), A, x)
        tmp_ref[] = tmp
        BLAS.axpby!(α, tmp, β, y)
    end

    function symv!_adjoint(::NoRData)
        if (α == 1 && β == 0)
            dα = dot(dy, y)
            BLAS.copyto!(y, y_copy)
        else
            # Reset y.
            BLAS.copyto!(y, y_copy)

            # gradient w.r.t. α. Safe to write into memory for copy of y.
            BLAS.symv!(ul, one(T), A, x, zero(T), y_copy)
            dα = dot(dy, y_copy)
        end

        # gradient w.r.t. A.
        dA_tmp = dy * x'
        if ul == 'L'
            dA .+= α .* LowerTriangular(dA_tmp)
            dA .+= α .* UpperTriangular(dA_tmp)'
        else
            dA .+= α .* LowerTriangular(dA_tmp)'
            dA .+= α .* UpperTriangular(dA_tmp)
        end
        @inbounds for n in diagind(dA)
            dA[n] -= α * dA_tmp[n]
        end

        # gradient w.r.t. x.
        BLAS.symv!(ul, α, A, dy, one(T), dx)

        # gradient w.r.t. beta.
        dβ = dot(dy, y)

        # gradient w.r.t. y.
        BLAS.scal!(β, dy)

        return NoRData(), NoRData(), dα, NoRData(), NoRData(), dβ, NoRData()
    end
    return y_dy, symv!_adjoint
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.trmv!),Char,Char,Char,AbstractMatrix{T},AbstractVector{T}
    } where {T<:BlasRealFloat},
)

function frule!!(
    ::Dual{typeof(BLAS.trmv!)},
    _uplo::Dual{Char},
    _trans::Dual{Char},
    _diag::Dual{Char},
    A_dA::Dual{<:AbstractMatrix{T}},
    x_dx::Dual{<:AbstractVector{T}},
) where {T<:BlasRealFloat}
    # Extract primals.
    uplo = primal(_uplo)
    trans = primal(_trans)
    diag = primal(_diag)
    A, dA = arrayify(A_dA)
    x, dx = arrayify(x_dx)

    # Frechet derivative computation.
    BLAS.trmv!(uplo, trans, diag, A, dx)
    tmp = copy(x)
    BLAS.trmv!(uplo, trans, diag, dA, tmp)
    dx .+= tmp
    if diag === 'U'
        dx .-= x
    end

    # Primal computation.
    BLAS.trmv!(uplo, trans, diag, A, x)

    return x_dx
end

function rrule!!(
    ::CoDual{typeof(BLAS.trmv!)},
    _uplo::CoDual{Char},
    _trans::CoDual{Char},
    _diag::CoDual{Char},
    A_dA::CoDual{<:AbstractMatrix{T}},
    x_dx::CoDual{<:AbstractVector{T}},
) where {T<:BlasRealFloat}

    # Extract primals.
    uplo = primal(_uplo)
    trans = primal(_trans)
    diag = primal(_diag)
    A, dA = arrayify(A_dA)
    x, dx = arrayify(x_dx)
    x_copy = copy(x)

    # Run primal computation.
    BLAS.trmv!(uplo, trans, diag, A, x)

    # Set dx to zero.
    dx .= zero(T)

    function trmv_pb!!(::NoRData)

        # Restore the original value of x.
        x .= x_copy

        # Increment the tangents.
        trans == 'N' ? inc_tri!(dA, dx, x, uplo, diag) : inc_tri!(dA, x, dx, uplo, diag)
        BLAS.trmv!(uplo, trans == 'N' ? 'T' : 'N', diag, A, dx)

        return tuple_fill(NoRData(), Val(6))
    end
    return x_dx, trmv_pb!!
end

function inc_tri!(A, x, y, uplo, diag)
    if uplo == 'L' && diag == 'U'
        @inbounds for q in 1:size(A, 2), p in (q + 1):size(A, 1)
            A[p, q] = fma(x[p], y[q], A[p, q])
        end
    elseif uplo == 'L' && diag == 'N'
        @inbounds for q in 1:size(A, 2), p in q:size(A, 1)
            A[p, q] = fma(x[p], y[q], A[p, q])
        end
    elseif uplo == 'U' && diag == 'U'
        @inbounds for q in 1:size(A, 2), p in 1:(q - 1)
            A[p, q] = fma(x[p], y[q], A[p, q])
        end
    elseif uplo == 'U' && diag == 'N'
        @inbounds for q in 1:size(A, 2), p in 1:q
            A[p, q] = fma(x[p], y[q], A[p, q])
        end
    else
        error("Unexpected uplo $uplo or diag $diag")
    end
end

#
# LEVEL 3
#

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.gemm!),
        Char,
        Char,
        T,
        AbstractMatrix{T},
        AbstractMatrix{T},
        T,
        AbstractMatrix{T},
    } where {T<:BlasRealFloat},
)

function frule!!(
    ::Dual{typeof(BLAS.gemm!)},
    transA::Dual{Char},
    transB::Dual{Char},
    alpha::Dual{T},
    A_dA::Dual{<:AbstractMatrix{T}},
    B_dB::Dual{<:AbstractMatrix{T}},
    beta::Dual{T},
    C_dC::Dual{<:AbstractMatrix{T}},
) where {T<:BlasRealFloat}
    tA = primal(transA)
    tB = primal(transB)
    α, dα = extract(alpha)
    β, dβ = extract(beta)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)
    C, dC = arrayify(C_dC)

    # Tangent computation.
    BLAS.gemm!(tA, tB, α, dA, B, β, dC)
    BLAS.gemm!(tA, tB, α, A, dB, one(T), dC)
    if !iszero(dα)
        BLAS.gemm!(tA, tB, dα, A, B, one(T), dC)
    end
    if !iszero(dβ)
        @inbounds for n in eachindex(C)
            dC[n] = ifelse_nan(C[n], dC[n], dC[n] + dβ * C[n])
        end
    end

    # Primal computation.
    BLAS.gemm!(tA, tB, α, A, B, β, C)

    return C_dC
end

function ifelse_nan(cond, left::P, right::P) where {P<:BlasRealFloat}
    return isnan(cond) * left + !isnan(cond) * right
end

function rrule!!(
    ::CoDual{typeof(BLAS.gemm!)},
    transA::CoDual{Char},
    transB::CoDual{Char},
    alpha::CoDual{T},
    A::CoDual{<:AbstractMatrix{T}},
    B::CoDual{<:AbstractMatrix{T}},
    beta::CoDual{T},
    C::CoDual{<:AbstractMatrix{T}},
) where {T<:BlasRealFloat}
    tA = primal(transA)
    tB = primal(transB)
    a = primal(alpha)
    b = primal(beta)
    p_A, dA = arrayify(A)
    p_B, dB = arrayify(B)
    p_C, dC = arrayify(C)

    # In this rule we optimise carefully for the special case a == 1 && b == 0, which
    # corresponds to simply multiplying A and B together, and writing the result to C.
    # This is an extremely common edge case, so it's important to do well for it.
    p_C_copy = copy(p_C)
    tmp_ref = Ref{Matrix{T}}()
    if (a == 1 && b == 0)
        BLAS.gemm!(tA, tB, a, p_A, p_B, b, p_C)
    else
        tmp = BLAS.gemm(tA, tB, one(T), p_A, p_B)
        tmp_ref[] = tmp
        p_C .= a .* tmp .+ b .* p_C
    end

    function gemm!_pb!!(::NoRData)

        # Compute pullback w.r.t. alpha.
        da = (a == 1 && b == 0) ? dot(dC, p_C) : dot(dC, tmp_ref[])

        # Restore previous state.
        BLAS.copyto!(p_C, p_C_copy)

        # Compute pullback w.r.t. beta.
        db = dot(dC, p_C)

        # Increment cotangents.
        if tA == 'N'
            BLAS.gemm!('N', tB == 'N' ? 'T' : 'N', a, dC, p_B, one(T), dA)
        else
            BLAS.gemm!(tB == 'N' ? 'N' : 'T', 'T', a, p_B, dC, one(T), dA)
        end
        if tB == 'N'
            BLAS.gemm!(tA == 'N' ? 'T' : 'N', 'N', a, p_A, dC, one(T), dB)
        else
            BLAS.gemm!('T', tA == 'N' ? 'N' : 'T', a, dC, p_A, one(T), dB)
        end
        dC .*= b

        return NoRData(), NoRData(), NoRData(), da, NoRData(), NoRData(), db, NoRData()
    end
    return C, gemm!_pb!!
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.symm!),
        Char,
        Char,
        T,
        AbstractMatrix{T},
        AbstractMatrix{T},
        T,
        AbstractMatrix{T},
    } where {T<:BlasRealFloat},
)
function frule!!(
    ::Dual{typeof(BLAS.symm!)},
    side::Dual{Char},
    uplo::Dual{Char},
    alpha::Dual{T},
    A_dA::Dual{<:AbstractMatrix{T}},
    B_dB::Dual{<:AbstractMatrix{T}},
    beta::Dual{T},
    C_dC::Dual{<:AbstractMatrix{T}},
) where {T<:BlasRealFloat}

    # Extract primals.
    s = primal(side)
    ul = primal(uplo)
    α, dα = extract(alpha)
    β, dβ = extract(beta)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)
    C, dC = arrayify(C_dC)

    # Compute Frechet derivative.
    BLAS.symm!(s, ul, α, A, dB, β, dC)
    BLAS.symm!(s, ul, α, dA, B, one(T), dC)
    if !iszero(dα)
        BLAS.symm!(s, ul, dα, A, B, one(T), dC)
    end
    if !iszero(dβ)
        @inbounds for n in eachindex(C)
            dC[n] = ifelse_nan(C[n], dC[n], dC[n] + dβ * C[n])
        end
    end

    # Run primal computation.
    BLAS.symm!(s, ul, α, A, B, β, C)
    return C_dC
end
function rrule!!(
    ::CoDual{typeof(BLAS.symm!)},
    side::CoDual{Char},
    uplo::CoDual{Char},
    alpha::CoDual{T},
    A_dA::CoDual{<:AbstractMatrix{T}},
    B_dB::CoDual{<:AbstractMatrix{T}},
    beta::CoDual{T},
    C_dC::CoDual{<:AbstractMatrix{T}},
) where {T<:BlasRealFloat}

    # Extract primals.
    s = primal(side)
    ul = primal(uplo)
    α = primal(alpha)
    β = primal(beta)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)
    C, dC = arrayify(C_dC)

    # In this rule we optimise carefully for the special case a == 1 && b == 0, which
    # corresponds to simply multiplying symm(A) and B together, and writing the result to C.
    # This is an extremely common edge case, so it's important to do well for it.
    C_copy = copy(C)
    tmp_ref = Ref{Matrix{T}}()
    if (α == 1 && β == 0)
        BLAS.symm!(s, ul, α, A, B, β, C)
    else
        tmp = BLAS.symm(s, ul, one(T), A, B)
        tmp_ref[] = tmp
        C .= α .* tmp .+ β .* C
    end

    function symm!_adjoint(::NoRData)
        if (α == 1 && β == 0)
            dα = dot(dC, C)
            BLAS.copyto!(C, C_copy)
        else
            # Reset C.
            BLAS.copyto!(C, C_copy)

            # gradient w.r.t. α. Safe to write into memory for copy of C.
            BLAS.symm!(s, ul, one(T), A, B, zero(T), C_copy)
            dα = dot(dC, C_copy)
        end

        # gradient w.r.t. A.
        dA_tmp = s == 'L' ? dC * B' : B' * dC
        if ul == 'L'
            dA .+= α .* LowerTriangular(dA_tmp)
            dA .+= α .* UpperTriangular(dA_tmp)'
        else
            dA .+= α .* LowerTriangular(dA_tmp)'
            dA .+= α .* UpperTriangular(dA_tmp)
        end
        @inbounds for n in diagind(dA)
            dA[n] -= α * dA_tmp[n]
        end

        # gradient w.r.t. B.
        BLAS.symm!(s, ul, α, A, dC, one(T), dB)

        # gradient w.r.t. beta.
        dβ = dot(dC, C)

        # gradient w.r.t. C.
        dC .*= β

        return NoRData(), NoRData(), NoRData(), dα, NoRData(), NoRData(), dβ, NoRData()
    end
    return C_dC, symm!_adjoint
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.syrk!),Char,Char,P,AbstractMatrix{P},P,AbstractMatrix{P}
    } where {P<:BlasRealFloat}
)
function frule!!(
    ::Dual{typeof(BLAS.syrk!)},
    _uplo::Dual{Char},
    _t::Dual{Char},
    α_dα::Dual{P},
    A_dA::Dual{<:AbstractMatrix{P}},
    β_dβ::Dual{P},
    C_dC::Dual{<:AbstractMatrix{P}},
) where {P<:BlasRealFloat}

    # Extract values from pairs.
    uplo = primal(_uplo)
    t = primal(_t)
    α, dα = extract(α_dα)
    A, dA = arrayify(A_dA)
    β, dβ = extract(β_dβ)
    C, dC = arrayify(C_dC)

    # Compute Frechet derivative.
    BLAS.syr2k!(uplo, t, α, A, dA, β, dC)
    iszero(dα) || BLAS.syrk!(uplo, t, dα, A, one(P), dC)
    if !iszero(dβ)
        dC .+= dβ .* (uplo == 'U' ? triu(C) : tril(C))
    end

    # Run primal computation.
    BLAS.syrk!(uplo, t, α, A, β, C)

    return C_dC
end
function rrule!!(
    ::CoDual{typeof(BLAS.syrk!)},
    _uplo::CoDual{Char},
    _t::CoDual{Char},
    α_dα::CoDual{P},
    A_dA::CoDual{<:AbstractMatrix{P}},
    β_dβ::CoDual{P},
    C_dC::CoDual{<:AbstractMatrix{P}},
) where {P<:BlasRealFloat}

    # Extract values from pairs.
    uplo = primal(_uplo)
    trans = primal(_t)
    α = primal(α_dα)
    A, dA = arrayify(A_dA)
    β = primal(β_dβ)
    C, dC = arrayify(C_dC)

    # Run forwards pass, and remember previous value of `C` for the reverse-pass.
    C_copy = collect(C)
    BLAS.syrk!(uplo, trans, α, A, β, C)

    function syrk_adjoint(::NoRData)
        # Restore previous state.
        C .= C_copy

        # C_copy no longer required, so its memory can be used to store other intermediate
        # results. Renaming for clarity.
        tmp = C_copy

        # Increment gradients.
        B = uplo == 'U' ? triu(dC) : tril(dC)
        ∇β = sum(B .* C)
        ∇α = tr(B' * _trans(trans, A) * _trans(trans, A)')
        # @show _t, size(A), size(B)
        dA .+= α * (trans == 'N' ? (B + B') * A : A * (B + B'))
        dC .= (uplo == 'U' ? tril!(dC, -1) : triu!(dC, 1)) .+ β .* B

        return NoRData(), NoRData(), NoRData(), ∇α, NoRData(), ∇β, NoRData()
    end

    return C_dC, syrk_adjoint
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.trmm!),Char,Char,Char,Char,P,AbstractMatrix{P},AbstractMatrix{P}
    } where {P<:BlasRealFloat}
)
function frule!!(
    ::Dual{typeof(BLAS.trmm!)},
    _side::Dual{Char},
    _uplo::Dual{Char},
    _ta::Dual{Char},
    _diag::Dual{Char},
    α_dα::Dual{P},
    A_dA::Dual{<:AbstractMatrix{P}},
    B_dB::Dual{<:AbstractMatrix{P}},
) where {P<:BlasRealFloat}

    # Extract data.
    side = primal(_side)
    uplo = primal(_uplo)
    ta = primal(_ta)
    diag = primal(_diag)
    α, dα = extract(α_dα)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)

    # Compute Frechet derivative.
    BLAS.trmm!(side, uplo, ta, diag, α, A, dB)
    dB .+= BLAS.trmm!(side, uplo, ta, diag, α, dA, copy(B))
    if diag == 'U'
        dB .-= α .* B
    end
    if !iszero(dα)
        dB .+= BLAS.trmm!(side, uplo, ta, diag, dα, A, copy(B))
    end

    # Compute primal.
    BLAS.trmm!(side, uplo, ta, diag, α, A, B)
    return B_dB
end
function rrule!!(
    ::CoDual{typeof(BLAS.trmm!)},
    _side::CoDual{Char},
    _uplo::CoDual{Char},
    _ta::CoDual{Char},
    _diag::CoDual{Char},
    α_dα::CoDual{P},
    A_dA::CoDual{<:AbstractMatrix{P}},
    B_dB::CoDual{<:AbstractMatrix{P}},
) where {P<:BlasRealFloat}

    # Extract values.
    side = primal(_side)
    uplo = primal(_uplo)
    tA = primal(_ta)
    diag = primal(_diag)
    α = primal(α_dα)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)
    B_copy = copy(B)

    # Run primal.
    BLAS.trmm!(side, uplo, tA, diag, α, A, B)

    function trmm_adjoint(::NoRData)

        # Compute α gradient.
        ∇α = tr(dB'B) / α

        # Restore initial state.
        B .= B_copy

        # Increment gradients.
        if side == 'L'
            dA .+= α .* tri!(tA == 'N' ? dB * B' : B * dB', uplo, diag)
        else
            dA .+= α .* tri!(tA == 'N' ? B'dB : dB'B, uplo, diag)
        end

        # Compute dB tangent.
        BLAS.trmm!(side, uplo, tA == 'N' ? 'T' : 'N', diag, α, A, dB)

        return tuple_fill(NoRData(), Val(5))..., ∇α, NoRData(), NoRData()
    end

    return B_dB, trmm_adjoint
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.trsm!),Char,Char,Char,Char,P,AbstractMatrix{P},AbstractMatrix{P}
    } where {P<:BlasRealFloat},
)

function frule!!(
    ::Dual{typeof(BLAS.trsm!)},
    _side::Dual{Char},
    _uplo::Dual{Char},
    _t::Dual{Char},
    _diag::Dual{Char},
    α_dα::Dual{P},
    A_dA::Dual{<:AbstractMatrix{P}},
    B_dB::Dual{<:AbstractMatrix{P}},
) where {P<:BlasRealFloat}

    # Extract parameters.
    side = primal(_side)
    uplo = primal(_uplo)
    trans = primal(_t)
    diag = primal(_diag)
    α, dα = extract(α_dα)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)

    # Compute Frechet derivative.
    BLAS.trsm!(side, uplo, trans, diag, α, A, dB)
    tmp = copy(B)
    trsm!(side, uplo, trans, diag, one(P), A, tmp) # tmp now contains inv(A) B.
    dB .+= dα .* tmp

    tmp2 = copy(tmp)
    BLAS.trmm!(side, uplo, trans, diag, α, dA, tmp) # tmp now contains α dA inv(A) B.
    if diag == 'U'
        tmp .-= α .* tmp2
    end
    BLAS.trsm!(side, uplo, trans, diag, one(P), A, tmp) # tmp is now α inv(A) dA inv(A) B.
    dB .-= tmp

    # Run primal computation.
    BLAS.trsm!(side, uplo, trans, diag, α, A, B)
    return B_dB
end

function rrule!!(
    ::CoDual{typeof(BLAS.trsm!)},
    _side::CoDual{Char},
    _uplo::CoDual{Char},
    _t::CoDual{Char},
    _diag::CoDual{Char},
    α_dα::CoDual{P},
    A_dA::CoDual{<:AbstractMatrix{P}},
    B_dB::CoDual{<:AbstractMatrix{P}},
) where {P<:BlasRealFloat}

    # Extract parameters.
    side = primal(_side)
    uplo = primal(_uplo)
    trans = primal(_t)
    diag = primal(_diag)
    α = primal(α_dα)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)

    # Copy memory which will be overwritten by primal computation.
    B_copy = copy(B)

    # Run primal computation.
    trsm!(side, uplo, trans, diag, α, A, B)

    function trsm_adjoint(::NoRData)
        # Compute α gradient.
        ∇α = tr(dB'B) / α

        # Increment cotangents.
        if side == 'L'
            if trans == 'N'
                tmp = trsm!('L', uplo, 'T', diag, -one(P), A, dB * B')
                dA .+= tri!(tmp, uplo, diag)
            else
                tmp = trsm!('R', uplo, 'T', diag, -one(P), A, B * dB')
                dA .+= tri!(tmp, uplo, diag)
            end
        else
            if trans == 'N'
                tmp = trsm!('R', uplo, 'T', diag, -one(P), A, B'dB)
                dA .+= tri!(tmp, uplo, diag)
            else
                tmp = trsm!('L', uplo, 'T', diag, -one(P), A, dB'B)
                dA .+= tri!(tmp, uplo, diag)
            end
        end

        # Restore initial state.
        B .= B_copy

        # Compute dB tangent.
        BLAS.trsm!(side, uplo, trans == 'N' ? 'T' : 'N', diag, α, A, dB)
        return tuple_fill(NoRData(), Val(5))..., ∇α, NoRData(), NoRData()
    end

    return B_dB, trsm_adjoint
end

function blas_matrices(rng::AbstractRNG, P::Type{<:BlasFloat}, p::Int, q::Int)
    Xs = Any[
        randn(rng, P, p, q),
        view(randn(rng, P, p + 5, 2q), 3:(p + 2), 1:2:(2q)),
        view(randn(rng, P, 3p, 3, 2q), (p + 1):(2p), 2, 1:2:(2q)),
        reshape(view(randn(rng, P, p * q + 5), 1:(p * q)), p, q),
    ]
    @assert all(X -> size(X) == (p, q), Xs)
    @assert all(Base.Fix2(isa, AbstractMatrix{P}), Xs)
    return Xs
end

function invertible_blas_matrices(rng::AbstractRNG, P::Type{<:BlasFloat}, p::Int)
    return map(blas_matrices(rng, P, p, p)) do A
        U, _, V = svd(0.1 * A + I)
        λs = p > 1 ? collect(range(1.0, 2.0; length=p)) : [1.0]
        A .= collect(U * Diagonal(λs) * V')
        return A
    end
end

function blas_vectors(rng::AbstractRNG, P::Type{<:BlasFloat}, p::Int)
    xs = Any[
        randn(rng, P, p),
        view(randn(rng, P, p + 5), 3:(p + 2)),
        view(randn(rng, P, 3p, 3), 1:2:(2p), 2),
        reshape(view(randn(rng, P, 1, p + 5), 1:1, 1:p), p),
    ]
    @assert all(x -> length(x) == p, xs)
    @assert all(Base.Fix2(isa, AbstractVector{P}), xs)
    return xs
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:blas})
    t_flags = ['N', 'T', 'C']
    αs = [1.0, -0.25]
    dαs = [0.0, 0.44]
    βs = [0.0, 0.33]
    dβs = [0.0, -0.11]
    uplos = ['L', 'U']
    dAs = ['N', 'U']
    Ps = [Float64, Float32]
    rng = rng_ctor(123456)

    test_cases = vcat(

        #
        # BLAS LEVEL 1
        #

        # nrm2(x)
        map_prod([Ps..., ComplexF64, ComplexF32]) do (P,)
            return map([randn(rng, P, 105)]) do x
                (false, :stability, nothing, BLAS.nrm2, x)
            end
        end...,

        # nrm2(n, x, incx)
        map_prod([Ps..., ComplexF64, ComplexF32], [5, 3], [1, 2]) do (P, n, incx)
            return map([randn(rng, P, 105)]) do x
                (false, :stability, nothing, BLAS.nrm2, n, x, incx)
            end
        end...,
        map_prod(Ps, [1, 3, 11], [1, 2, 11]) do (P, n, incx)
            flags = (false, :stability, nothing)
            return (flags..., BLAS.scal!, n, randn(rng, P), randn(rng, P, n * incx), incx)
        end,

        #
        # BLAS LEVEL 2
        #

        # gemv!
        map_prod(t_flags, [1, 3], [1, 2], Ps, αs, βs) do (tA, M, N, P, α, β)
            As = blas_matrices(rng, P, tA == 'N' ? M : N, tA == 'N' ? N : M)
            xs = blas_vectors(rng, P, N)
            ys = blas_vectors(rng, P, M)
            flags = (false, :stability, (lb=1e-3, ub=10.0))
            return map(As, xs, ys) do A, x, y
                (flags..., BLAS.gemv!, tA, P(α), A, x, P(β), y)
            end
        end...,

        # symv!
        map_prod(['L', 'U'], αs, βs, Ps) do (uplo, α, β, P)
            As = blas_matrices(rng, P, 5, 5)
            ys = blas_vectors(rng, P, 5)
            xs = blas_vectors(rng, P, 5)
            return map(As, xs, ys) do A, x, y
                (false, :stability, nothing, BLAS.symv!, uplo, P(α), A, x, P(β), y)
            end
        end...,

        # trmv!
        map_prod(uplos, t_flags, dAs, [1, 3], Ps) do (ul, tA, dA, N, P)
            As = blas_matrices(rng, P, N, N)
            bs = blas_vectors(rng, P, N)
            return map(As, bs) do A, b
                (false, :stability, nothing, BLAS.trmv!, ul, tA, dA, A, b)
            end
        end...,

        # #
        # # BLAS LEVEL 3
        # #

        # gemm!
        map_prod(t_flags, t_flags, αs, βs, Ps, dαs, dβs) do (tA, tB, α, β, P, dα, dβ)
            As = blas_matrices(rng, P, tA == 'N' ? 3 : 4, tA == 'N' ? 4 : 3)
            Bs = blas_matrices(rng, P, tB == 'N' ? 4 : 5, tB == 'N' ? 5 : 4)
            Cs = blas_matrices(rng, P, 3, 5)

            return map(As, Bs, Cs) do A, B, C
                a_da = CoDual(P(α), P(dα))
                b_db = CoDual(P(β), P(dβ))
                (false, :stability, nothing, BLAS.gemm!, tA, tB, a_da, A, B, b_db, C)
            end
        end...,

        # symm!
        map_prod(['L', 'R'], ['L', 'U'], αs, βs, Ps) do (side, ul, α, β, P)
            nA = side == 'L' ? 5 : 7
            As = blas_matrices(rng, P, nA, nA)
            Bs = blas_matrices(rng, P, 5, 7)
            Cs = blas_matrices(rng, P, 5, 7)
            return map(As, Bs, Cs) do A, B, C
                (false, :stability, nothing, BLAS.symm!, side, ul, P(α), A, B, P(β), C)
            end
        end...,

        # syrk!
        map_prod(uplos, t_flags, Ps, dαs, dβs) do (uplo, t, P, dα, dβ)
            As = blas_matrices(rng, P, t == 'N' ? 3 : 4, t == 'N' ? 4 : 3)
            return map(As) do A
                α_dα = CoDual(randn(rng, P), P(dα))
                β_dβ = CoDual(randn(rng, P), P(dβ))
                C = randn(rng, P, 3, 3)
                (false, :stability, nothing, BLAS.syrk!, uplo, t, α_dα, A, β_dβ, C)
            end
        end...,

        # trmm!
        map_prod(
            ['L', 'R'], uplos, t_flags, dAs, [1, 3], [1, 2], Ps, dαs
        ) do (side, ul, tA, dA, M, N, P, dα)
            t = tA == 'N'
            R = side == 'L' ? M : N
            As = blas_matrices(rng, P, R, R)
            Bs = blas_matrices(rng, P, M, N)
            return map(As, Bs) do A, B
                α_dα = CoDual(randn(rng, P), P(dα))
                (false, :stability, nothing, BLAS.trmm!, side, ul, tA, dA, α_dα, A, B)
            end
        end...,

        # trsm!
        map_prod(
            ['L', 'R'], uplos, t_flags, dAs, [1, 3], [1, 2], Ps
        ) do (side, ul, tA, dA, M, N, P)
            t = tA == 'N'
            R = side == 'L' ? M : N
            a = randn(rng, P)
            As = map(blas_matrices(rng, P, R, R)) do A
                A[diagind(A)] .+= 1
                return A
            end
            Bs = blas_matrices(rng, P, M, N)
            return map(As, Bs) do A, B
                (false, :stability, nothing, BLAS.trsm!, side, ul, tA, dA, a, A, B)
            end
        end...,
    )

    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:blas})
    t_flags = ['N', 'T', 'C']
    aliased_gemm! = (tA, tB, a, b, A, C) -> BLAS.gemm!(tA, tB, a, A, A, b, C)
    Ps = [Float32, Float64]
    uplos = ['L', 'U']
    dAs = ['N', 'U']
    rng = rng_ctor(123)

    test_cases = vcat(

        # Utility
        (false, :stability, nothing, BLAS.get_num_threads),
        (false, :stability, nothing, BLAS.lbt_get_num_threads),
        (false, :stability, nothing, BLAS.set_num_threads, 1),
        (false, :stability, nothing, BLAS.lbt_set_num_threads, 1),

        #
        # BLAS LEVEL 1
        #

        map(Ps) do P
            flags = (false, :none, nothing)
            Any[
                (flags..., BLAS.dot, 3, randn(rng, P, 5), 1, randn(rng, P, 4), 1),
                (flags..., BLAS.dot, 3, randn(rng, P, 6), 2, randn(rng, P, 4), 1),
                (flags..., BLAS.dot, 3, randn(rng, P, 6), 1, randn(rng, P, 9), 3),
                (flags..., BLAS.dot, 3, randn(rng, P, 12), 3, randn(rng, P, 9), 2),
            ]
        end...,

        #
        # BLAS LEVEL 3
        #

        # aliased gemm!
        map_prod(t_flags, t_flags, Ps) do (tA, tB, P)
            As = blas_matrices(rng, P, 5, 5)
            Bs = blas_matrices(rng, P, 5, 5)
            a = randn(rng, P)
            b = randn(rng, P)
            return map_prod(As, Bs) do (A, B)
                (false, :none, nothing, aliased_gemm!, tA, tB, a, b, A, B)
            end
        end...,
    )
    memory = Any[]
    return test_cases, memory
end
