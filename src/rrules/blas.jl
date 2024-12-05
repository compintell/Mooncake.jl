function blas_name(name::Symbol)
    return (BLAS.USE_BLAS64 ? Symbol(name, "64_") : name, Symbol(BLAS.libblastrampoline))
end

function wrap_ptr_as_view(ptr::Ptr{T}, N::Int, inc::Int) where {T}
    return view(unsafe_wrap(Vector{T}, ptr, N * inc), 1:inc:(N * inc))
end

function wrap_ptr_as_view(ptr::Ptr{T}, buffer_nrows::Int, nrows::Int, ncols::Int) where {T}
    return view(unsafe_wrap(Matrix{T}, ptr, (buffer_nrows, ncols)), 1:nrows, :)
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

const MatrixOrView{T} = Union{Matrix{T},SubArray{T,2,<:Array{T}}}
const VecOrView{T} = Union{Vector{T},SubArray{T,1,<:Array{T}}}
const BlasRealFloat = Union{Float32,Float64}

#
# Utility
#

@zero_adjoint MinimalCtx Tuple{typeof(BLAS.get_num_threads)}
@zero_adjoint MinimalCtx Tuple{typeof(BLAS.lbt_get_num_threads)}
@zero_adjoint MinimalCtx Tuple{typeof(BLAS.set_num_threads),Union{Integer,Nothing}}
@zero_adjoint MinimalCtx Tuple{typeof(BLAS.lbt_set_num_threads),Any}

#
# LEVEL 1
#

for (fname, elty) in ((:cblas_ddot, :Float64), (:cblas_sdot, :Float32))
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

for (fname, elty) in ((:dscal_, :Float64), (:sscal_, :Float32))
    @eval @inline function Mooncake.rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{$(blas_name(fname))}},
        ::CoDual, # return type
        ::CoDual, # argument types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        n::CoDual{Ptr{BLAS.BlasInt}},
        DA::CoDual{Ptr{$elty}},
        DX::CoDual{Ptr{$elty}},
        incx::CoDual{Ptr{BLAS.BlasInt}},
        args::Vararg{Any,N},
    ) where {N}
        GC.@preserve args begin

            # Load in values from pointers, and turn pointers to memory buffers into Vectors.
            _n = unsafe_load(primal(n))
            _incx = unsafe_load(primal(incx))
            _DA = unsafe_load(primal(DA))
            _DX = unsafe_wrap(Vector{$elty}, primal(DX), _n * _incx)
            _DX_s = unsafe_wrap(Vector{$elty}, tangent(DX), _n * _incx)

            inds = 1:_incx:(_incx * _n)
            DX_copy = _DX[inds]
            BLAS.scal!(_n, _DA, _DX, _incx)

            dDA = tangent(DA)
            dDX = tangent(DX)
        end

        function dscal_pullback!!(::NoRData)
            GC.@preserve args begin

                # Set primal to previous state.
                _DX[inds] .= DX_copy

                # Compute cotangent w.r.t. scaling.
                unsafe_store!(dDA, BLAS.dot(_n, _DX, _incx, dDX, _incx) + unsafe_load(dDA))

                # Compute cotangent w.r.t. DX.
                BLAS.scal!(_n, _DA, _DX_s, _incx)
            end

            return tuple_fill(NoRData(), Val(10 + N))
        end
        return zero_fcodual(Cvoid()), dscal_pullback!!
    end
end

#
# LEVEL 2
#

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.gemv!),Char,P,MatrixOrView{P},VecOrView{P},P,VecOrView{P}
    } where {P<:BlasRealFloat},
)

@inline function rrule!!(
    ::CoDual{typeof(BLAS.gemv!)},
    _tA::CoDual{Char},
    _alpha::CoDual{P},
    _A::CoDual{<:MatrixOrView{P}},
    _x::CoDual{<:VecOrView{P}},
    _beta::CoDual{P},
    _y::CoDual{<:VecOrView{P}},
) where {P<:BlasRealFloat}

    # Pull out primals and tangents (the latter only where necessary).
    trans = _tA.x
    alpha = _alpha.x
    A, dA = viewify(_A)
    x, dx = viewify(_x)
    beta = _beta.x
    y, dy = viewify(_y)

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
        typeof(BLAS.symv!),Char,T,MatrixOrView{T},VecOrView{T},T,VecOrView{T}
    } where {T<:BlasRealFloat},
)

function rrule!!(
    ::CoDual{typeof(BLAS.symv!)},
    uplo::CoDual{Char},
    alpha::CoDual{T},
    A_dA::CoDual{<:MatrixOrView{T}},
    x_dx::CoDual{<:VecOrView{T}},
    beta::CoDual{T},
    y_dy::CoDual{<:VecOrView{T}},
) where {T<:BlasRealFloat}

    # Extract primals.
    ul = primal(uplo)
    α = primal(alpha)
    β = primal(beta)
    A, dA = viewify(A_dA)
    x, dx = viewify(x_dx)
    y, dy = viewify(y_dy)

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

for (trmv, elty) in ((:dtrmv_, :Float64), (:strmv_, :Float32))
    @eval @inline function rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{$(blas_name(trmv))}},
        ::CoDual,
        ::CoDual,
        ::CoDual,
        ::CoDual,
        _uplo::CoDual{Ptr{UInt8}},
        _trans::CoDual{Ptr{UInt8}},
        _diag::CoDual{Ptr{UInt8}},
        _N::CoDual{Ptr{BLAS.BlasInt}},
        _A::CoDual{Ptr{$elty}},
        _lda::CoDual{Ptr{BLAS.BlasInt}},
        _x::CoDual{Ptr{$elty}},
        _incx::CoDual{Ptr{BLAS.BlasInt}},
        args::Vararg{Any,Nargs},
    ) where {Nargs}
        GC.@preserve args begin
            # Load in data.
            uplo, trans, diag = map(Char ∘ unsafe_load ∘ primal, (_uplo, _trans, _diag))
            N, lda, incx = map(unsafe_load ∘ primal, (_N, _lda, _incx))
            A = wrap_ptr_as_view(primal(_A), lda, N, N)
            x = wrap_ptr_as_view(primal(_x), N, incx)
            x_copy = copy(x)

            # Run primal computation.
            BLAS.trmv!(uplo, trans, diag, A, x)

            _dA = tangent(_A)
            _dx = tangent(_x)
        end

        function trmv_pb!!(::NoRData)
            GC.@preserve args begin

                # Load up the tangents.
                dA = wrap_ptr_as_view(_dA, lda, N, N)
                dx = wrap_ptr_as_view(_dx, N, incx)

                # Restore the original value of x.
                x .= x_copy

                # Increment the tangents.
                dA .+= tri!(trans == 'N' ? dx * x' : x * dx', uplo, diag)
                BLAS.trmv!(uplo, trans == 'N' ? 'T' : 'N', diag, A, dx)
            end

            return tuple_fill(NoRData(), Val(14 + Nargs))
        end
        return zero_fcodual(Cvoid()), trmv_pb!!
    end
end

#
# LEVEL 3
#

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.gemm!),Char,Char,T,MatrixOrView{T},MatrixOrView{T},T,MatrixOrView{T}
    } where {T<:BlasRealFloat},
)

function rrule!!(
    ::CoDual{typeof(BLAS.gemm!)},
    transA::CoDual{Char},
    transB::CoDual{Char},
    alpha::CoDual{T},
    A::CoDual{<:MatrixOrView{T}},
    B::CoDual{<:MatrixOrView{T}},
    beta::CoDual{T},
    C::CoDual{<:MatrixOrView{T}},
) where {T<:BlasRealFloat}
    tA = primal(transA)
    tB = primal(transB)
    a = primal(alpha)
    b = primal(beta)
    p_A, dA = viewify(A)
    p_B, dB = viewify(B)
    p_C, dC = viewify(C)

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

viewify(A::CoDual{<:Vector}) = primal(A), tangent(A)
viewify(A::CoDual{<:Matrix}) = view(primal(A), :, :), view(tangent(A), :, :)
function viewify(A::CoDual{P}) where {P<:SubArray}
    p_A = primal(A)
    return p_A, P(tangent(A).data.parent, p_A.indices, p_A.offset1, p_A.stride1)
end

for (gemm, elty) in ((:dgemm_, :Float64), (:sgemm_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{$(blas_name(gemm))}},
        ::CoDual{Val{Cvoid}},
        ::CoDual, # arg types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        tA::CoDual{Ptr{UInt8}},
        tB::CoDual{Ptr{UInt8}},
        m::CoDual{Ptr{BLAS.BlasInt}},
        n::CoDual{Ptr{BLAS.BlasInt}},
        ka::CoDual{Ptr{BLAS.BlasInt}},
        alpha::CoDual{Ptr{$elty}},
        A::CoDual{Ptr{$elty}},
        LDA::CoDual{Ptr{BLAS.BlasInt}},
        B::CoDual{Ptr{$elty}},
        LDB::CoDual{Ptr{BLAS.BlasInt}},
        beta::CoDual{Ptr{$elty}},
        C::CoDual{Ptr{$elty}},
        LDC::CoDual{Ptr{BLAS.BlasInt}},
        args::Vararg{Any,Nargs},
    ) where {Nargs}
        GC.@preserve args begin
            _tA = Char(unsafe_load(primal(tA)))
            _tB = Char(unsafe_load(primal(tB)))
            _m = unsafe_load(primal(m))
            _n = unsafe_load(primal(n))
            _ka = unsafe_load(primal(ka))
            _alpha = unsafe_load(primal(alpha))
            _A = primal(A)
            _LDA = unsafe_load(primal(LDA))
            _B = primal(B)
            _LDB = unsafe_load(primal(LDB))
            _beta = unsafe_load(primal(beta))
            _C = primal(C)
            _LDC = unsafe_load(primal(LDC))

            A_mat = wrap_ptr_as_view(
                primal(A), _LDA, (_tA == 'N' ? (_m, _ka) : (_ka, _m))...
            )
            B_mat = wrap_ptr_as_view(
                primal(B), _LDB, (_tB == 'N' ? (_ka, _n) : (_n, _ka))...
            )
            C_mat = wrap_ptr_as_view(primal(C), _LDC, _m, _n)
            C_copy = collect(C_mat)

            BLAS.gemm!(_tA, _tB, _alpha, A_mat, B_mat, _beta, C_mat)

            dalpha = tangent(alpha)
            dA = tangent(A)
            dB = tangent(B)
            dbeta = tangent(beta)
            dC = tangent(C)
        end

        function gemm!_pullback!!(::NoRData)
            GC.@preserve args begin
                # Restore previous state.
                C_mat .= C_copy

                # Convert pointers to views.
                dA_mat = wrap_ptr_as_view(dA, _LDA, (_tA == 'N' ? (_m, _ka) : (_ka, _m))...)
                dB_mat = wrap_ptr_as_view(dB, _LDB, (_tB == 'N' ? (_ka, _n) : (_n, _ka))...)
                dC_mat = wrap_ptr_as_view(dC, _LDC, _m, _n)

                # Increment cotangents.
                unsafe_store!(dbeta, unsafe_load(dbeta) + tr(dC_mat' * C_mat))
                dalpha_inc = tr(dC_mat' * _trans(_tA, A_mat) * _trans(_tB, B_mat))
                unsafe_store!(dalpha, unsafe_load(dalpha) + dalpha_inc)
                dA_mat .+=
                    _alpha * transpose(_trans(_tA, _trans(_tB, B_mat) * transpose(dC_mat)))
                dB_mat .+=
                    _alpha * transpose(_trans(_tB, transpose(dC_mat) * _trans(_tA, A_mat)))
                dC_mat .*= _beta
            end

            return tuple_fill(NoRData(), Val(19 + Nargs))
        end
        return zero_fcodual(Cvoid()), gemm!_pullback!!
    end
end

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(BLAS.symm!),Char,Char,T,MatrixOrView{T},MatrixOrView{T},T,MatrixOrView{T}
    } where {T<:BlasRealFloat},
)

function rrule!!(
    ::CoDual{typeof(BLAS.symm!)},
    side::CoDual{Char},
    uplo::CoDual{Char},
    alpha::CoDual{T},
    A_dA::CoDual{<:MatrixOrView{T}},
    B_dB::CoDual{<:MatrixOrView{T}},
    beta::CoDual{T},
    C_dC::CoDual{<:MatrixOrView{T}},
) where {T<:BlasRealFloat}

    # Extract primals.
    s = primal(side)
    ul = primal(uplo)
    α = primal(alpha)
    β = primal(beta)
    A, dA = viewify(A_dA)
    B, dB = viewify(B_dB)
    C, dC = viewify(C_dC)

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

for (syrk, elty) in ((:dsyrk_, :Float64), (:ssyrk_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{$(blas_name(syrk))}},
        ::CoDual{Val{Cvoid}},
        ::CoDual, # arg types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        uplo::CoDual{Ptr{UInt8}},
        trans::CoDual{Ptr{UInt8}},
        n::CoDual{Ptr{BLAS.BlasInt}},
        k::CoDual{Ptr{BLAS.BlasInt}},
        alpha::CoDual{Ptr{$elty}},
        A::CoDual{Ptr{$elty}},
        LDA::CoDual{Ptr{BLAS.BlasInt}},
        beta::CoDual{Ptr{$elty}},
        C::CoDual{Ptr{$elty}},
        LDC::CoDual{Ptr{BLAS.BlasInt}},
        args::Vararg{Any,Nargs},
    ) where {Nargs}
        GC.@preserve args begin
            _uplo = Char(unsafe_load(primal(uplo)))
            _t = Char(unsafe_load(primal(trans)))
            _n = unsafe_load(primal(n))
            _k = unsafe_load(primal(k))
            _alpha = unsafe_load(primal(alpha))
            _A = primal(A)
            _LDA = unsafe_load(primal(LDA))
            _beta = unsafe_load(primal(beta))
            _C = primal(C)
            _LDC = unsafe_load(primal(LDC))

            A_mat = wrap_ptr_as_view(primal(A), _LDA, (_t == 'N' ? (_n, _k) : (_k, _n))...)
            C_mat = wrap_ptr_as_view(primal(C), _LDC, _n, _n)
            C_copy = collect(C_mat)

            BLAS.syrk!(_uplo, _t, _alpha, A_mat, _beta, C_mat)

            dalpha = tangent(alpha)
            dA = tangent(A)
            dbeta = tangent(beta)
            dC = tangent(C)
        end

        function syrk!_pullback!!(::NoRData)
            GC.@preserve args begin
                # Restore previous state.
                C_mat .= C_copy

                # Convert pointers to views.
                dA_mat = wrap_ptr_as_view(dA, _LDA, (_t == 'N' ? (_n, _k) : (_k, _n))...)
                dC_mat = wrap_ptr_as_view(dC, _LDC, _n, _n)

                # Increment cotangents.
                B = _uplo == 'U' ? triu(dC_mat) : tril(dC_mat)
                unsafe_store!(dbeta, unsafe_load(dbeta) + sum(B .* C_mat))
                dalpha_inc = tr(B' * _trans(_t, A_mat) * _trans(_t, A_mat)')
                unsafe_store!(dalpha, unsafe_load(dalpha) + dalpha_inc)
                dA_mat .+= _alpha * (_t == 'N' ? (B + B') * A_mat : A_mat * (B + B'))
                dC_mat .=
                    (_uplo == 'U' ? tril!(dC_mat, -1) : triu!(dC_mat, 1)) .+ _beta .* B
            end

            return tuple_fill(NoRData(), Val(16 + Nargs))
        end
        return zero_fcodual(Cvoid()), syrk!_pullback!!
    end
end

for (trmm, elty) in ((:dtrmm_, :Float64), (:strmm_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{$(blas_name(trmm))}},
        ::CoDual,
        ::CoDual, # arg types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        _side::CoDual{Ptr{UInt8}},
        _uplo::CoDual{Ptr{UInt8}},
        _trans::CoDual{Ptr{UInt8}},
        _diag::CoDual{Ptr{UInt8}},
        _M::CoDual{Ptr{BLAS.BlasInt}},
        _N::CoDual{Ptr{BLAS.BlasInt}},
        _alpha::CoDual{Ptr{$elty}},
        _A::CoDual{Ptr{$elty}},
        _lda::CoDual{Ptr{BLAS.BlasInt}},
        _B::CoDual{Ptr{$elty}},
        _ldb::CoDual{Ptr{BLAS.BlasInt}},
        args::Vararg{Any,Nargs},
    ) where {Nargs}
        GC.@preserve args begin

            # Load in data and store B for the reverse-pass.
            side, ul, tA, diag = map(
                Char ∘ unsafe_load ∘ primal, (_side, _uplo, _trans, _diag)
            )
            M, N, lda, ldb = map(unsafe_load ∘ primal, (_M, _N, _lda, _ldb))
            alpha = unsafe_load(primal(_alpha))
            R = side == 'L' ? M : N
            A = wrap_ptr_as_view(primal(_A), lda, R, R)
            B = wrap_ptr_as_view(primal(_B), ldb, M, N)
            B_copy = copy(B)

            # Run primal.
            BLAS.trmm!(side, ul, tA, diag, alpha, A, B)

            dalpha = tangent(_alpha)
            _dA = tangent(_A)
            _dB = tangent(_B)
        end

        function trmm!_pullback!!(::NoRData)
            GC.@preserve args begin
                # Convert pointers to views.
                dA = wrap_ptr_as_view(_dA, lda, R, R)
                dB = wrap_ptr_as_view(_dB, ldb, M, N)

                # Increment alpha tangent.
                alpha != 0 && unsafe_store!(dalpha, unsafe_load(dalpha) + tr(dB'B) / alpha)

                # Restore initial state.
                B .= B_copy

                # Increment cotangents.
                if side == 'L'
                    dA .+= alpha .* tri!(tA == 'N' ? dB * B' : B * dB', ul, diag)
                else
                    dA .+= alpha .* tri!(tA == 'N' ? B'dB : dB'B, ul, diag)
                end

                # Compute dB tangent.
                BLAS.trmm!(side, ul, tA == 'N' ? 'T' : 'N', diag, alpha, A, dB)
            end

            return tuple_fill(NoRData(), Val(17 + Nargs))
        end

        return zero_fcodual(Cvoid()), trmm!_pullback!!
    end
end

for (trsm, elty) in ((:dtrsm_, :Float64), (:strsm_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{$(blas_name(trsm))}},
        ::CoDual,
        ::CoDual, # arg types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        _side::CoDual{Ptr{UInt8}},
        _uplo::CoDual{Ptr{UInt8}},
        _trans::CoDual{Ptr{UInt8}},
        _diag::CoDual{Ptr{UInt8}},
        _M::CoDual{Ptr{BLAS.BlasInt}},
        _N::CoDual{Ptr{BLAS.BlasInt}},
        _alpha::CoDual{Ptr{$elty}},
        _A::CoDual{Ptr{$elty}},
        _lda::CoDual{Ptr{BLAS.BlasInt}},
        _B::CoDual{Ptr{$elty}},
        _ldb::CoDual{Ptr{BLAS.BlasInt}},
        args::Vararg{Any,Nargs},
    ) where {Nargs}
        GC.@preserve args begin
            side = Char(unsafe_load(primal(_side)))
            uplo = Char(unsafe_load(primal(_uplo)))
            trans = Char(unsafe_load(primal(_trans)))
            diag = Char(unsafe_load(primal(_diag)))
            M = unsafe_load(primal(_M))
            N = unsafe_load(primal(_N))
            R = side == 'L' ? M : N
            alpha = unsafe_load(primal(_alpha))
            lda = unsafe_load(primal(_lda))
            ldb = unsafe_load(primal(_ldb))
            A = wrap_ptr_as_view(primal(_A), lda, R, R)
            B = wrap_ptr_as_view(primal(_B), ldb, M, N)
            B_copy = copy(B)

            trsm!(side, uplo, trans, diag, alpha, A, B)

            dalpha = tangent(_alpha)
            _dA = tangent(_A)
            _dB = tangent(_B)
        end

        function trsm_pb!!(::NoRData)
            GC.@preserve args begin
                # Convert pointers to views.
                dA = wrap_ptr_as_view(_dA, lda, R, R)
                dB = wrap_ptr_as_view(_dB, ldb, M, N)

                # Increment alpha tangent.
                alpha != 0 && unsafe_store!(dalpha, unsafe_load(dalpha) + tr(dB'B) / alpha)

                # Increment cotangents.
                if side == 'L'
                    if trans == 'N'
                        tmp = trsm!('L', uplo, 'T', diag, -one($elty), A, dB * B')
                        dA .+= tri!(tmp, uplo, diag)
                    else
                        tmp = trsm!('R', uplo, 'T', diag, -one($elty), A, B * dB')
                        dA .+= tri!(tmp, uplo, diag)
                    end
                else
                    if trans == 'N'
                        tmp = trsm!('R', uplo, 'T', diag, -one($elty), A, B'dB)
                        dA .+= tri!(tmp, uplo, diag)
                    else
                        tmp = trsm!('L', uplo, 'T', diag, -one($elty), A, dB'B)
                        dA .+= tri!(tmp, uplo, diag)
                    end
                end

                # Restore initial state.
                B .= B_copy

                # Compute dB tangent.
                BLAS.trsm!(side, uplo, trans == 'N' ? 'T' : 'N', diag, alpha, A, dB)
            end

            return tuple_fill(NoRData(), Val(17 + Nargs))
        end
        return zero_fcodual(Cvoid()), trsm_pb!!
    end
end

function blas_matrices(rng::AbstractRNG, P::Type{<:BlasRealFloat}, p::Int, q::Int)
    Xs = Any[
        randn(rng, P, p, q),
        view(randn(rng, P, p + 5, 2q), 3:(p + 2), 1:2:(2q)),
        view(randn(rng, P, 3p, 3, 2q), (p + 1):(2p), 2, 1:2:(2q)),
    ]
    @assert all(X -> size(X) == (p, q), Xs)
    @assert all(Base.Fix2(isa, AbstractMatrix{P}), Xs)
    return Xs
end

function blas_vectors(rng::AbstractRNG, P::Type{<:BlasRealFloat}, p::Int)
    xs = Any[
        randn(rng, P, p),
        view(randn(rng, P, p + 5), 3:(p + 2)),
        view(randn(rng, P, 3p), 1:2:(2p)),
        view(randn(rng, P, 3p, 3), 1:2:(2p), 2),
    ]
    @assert all(x -> length(x) == p, xs)
    @assert all(Base.Fix2(isa, AbstractVector{P}), xs)
    return xs
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:blas})
    t_flags = ['N', 'T', 'C']
    alphas = [1.0, -0.25]
    betas = [0.0, 0.33]
    rng = rng_ctor(123456)

    test_cases = vcat(

        # gemv!
        vec(
            reduce(
                vcat,
                map(product(t_flags, [1, 3], [1, 2])) do (tA, M, N)
                    t = tA == 'N'
                    As = blas_matrices(rng, Float64, t ? M : N, t ? N : M)
                    xs = blas_vectors(rng, Float64, N)
                    ys = blas_vectors(rng, Float64, M)
                    flags = (false, :stability, (lb=1e-3, ub=5.0))
                    return map(product(As, xs, ys)) do (A, x, y)
                        return (flags..., BLAS.gemv!, tA, randn(), A, x, randn(), y)
                    end
                end,
            ),
        ),

        # symv!
        vec(
            reduce(
                vcat,
                map(product(['L', 'U'], alphas, betas)) do (uplo, α, β)
                    As = blas_matrices(rng, Float64, 5, 5)
                    ys = blas_vectors(rng, Float64, 5)
                    xs = blas_vectors(rng, Float64, 5)
                    return map(product(As, xs, ys)) do (A, x, y)
                        (false, :stability, nothing, BLAS.symv!, uplo, α, A, x, β, y)
                    end
                end,
            ),
        ),

        # gemm!
        vec(
            reduce(
                vcat,
                map(product(t_flags, t_flags, alphas, betas)) do (tA, tB, a, b)
                    As = blas_matrices(rng, Float64, tA == 'N' ? 3 : 4, tA == 'N' ? 4 : 3)
                    Bs = blas_matrices(rng, Float64, tB == 'N' ? 4 : 5, tB == 'N' ? 5 : 4)
                    Cs = blas_matrices(rng, Float64, 3, 5)
                    return map(product(As, Bs, Cs)) do (A, B, C)
                        (false, :none, nothing, BLAS.gemm!, tA, tB, a, A, B, b, C)
                    end
                end,
            ),
        ),

        # symm!
        vec(
            reduce(
                vcat,
                map(product(['L', 'R'], ['L', 'U'], alphas, betas)) do (side, uplo, α, β)
                    nA = side == 'L' ? 5 : 7
                    As = blas_matrices(rng, Float64, nA, nA)
                    Bs = blas_matrices(rng, Float64, 5, 7)
                    Cs = blas_matrices(rng, Float64, 5, 7)
                    return map(product(As, Bs, Cs)) do (A, B, C)
                        (false, :stability, nothing, BLAS.symm!, side, uplo, α, A, B, β, C)
                    end
                end,
            ),
        ),
    )

    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:blas})
    t_flags = ['N', 'T', 'C']
    aliased_gemm! = (tA, tB, a, b, A, C) -> BLAS.gemm!(tA, tB, a, A, A, b, C)

    test_cases = vcat(

        # Utility
        (false, :stability, nothing, BLAS.get_num_threads),
        (false, :stability, nothing, BLAS.lbt_get_num_threads),
        (false, :stability, nothing, BLAS.set_num_threads, 1),
        (false, :stability, nothing, BLAS.lbt_set_num_threads, 1),

        #
        # BLAS LEVEL 1
        #

        Any[
            (false, :none, nothing, BLAS.dot, 3, randn(5), 1, randn(4), 1),
            (false, :none, nothing, BLAS.dot, 3, randn(6), 2, randn(4), 1),
            (false, :none, nothing, BLAS.dot, 3, randn(6), 1, randn(9), 3),
            (false, :none, nothing, BLAS.dot, 3, randn(12), 3, randn(9), 2),
            (false, :none, nothing, BLAS.scal!, 10, 2.4, randn(30), 2),
        ],

        #
        # BLAS LEVEL 2
        #

        # trmv!
        vec(
            reduce(
                vcat,
                map(product(['L', 'U'], t_flags, ['N', 'U'], [1, 3])) do (ul, tA, dA, N)
                    As = [randn(N, N), view(randn(15, 15), 3:(N + 2), 4:(N + 3))]
                    bs = [randn(N), view(randn(14), 4:(N + 3))]
                    return map(product(As, bs)) do (A, b)
                        (false, :none, nothing, BLAS.trmv!, ul, tA, dA, A, b)
                    end
                end,
            ),
        ),

        #
        # BLAS LEVEL 3
        #

        # aliased gemm!
        vec(
            map(product(t_flags, t_flags)) do (tA, tB)
                A = randn(5, 5)
                B = randn(5, 5)
                (false, :none, nothing, aliased_gemm!, tA, tB, randn(), randn(), A, B)
            end,
        ),

        # syrk!
        vec(
            map(product(['U', 'L'], t_flags)) do (uplo, t)
                A = t == 'N' ? randn(3, 4) : randn(4, 3)
                C = randn(3, 3)
                return (false, :none, nothing, BLAS.syrk!, uplo, t, randn(), A, randn(), C)
            end,
        ),

        # trmm!
        vec(
            reduce(
                vcat,
                map(
                    product(['L', 'R'], ['U', 'L'], t_flags, ['N', 'U'], [1, 3], [1, 2])
                ) do (side, ul, tA, dA, M, N)
                    t = tA == 'N'
                    R = side == 'L' ? M : N
                    As = [randn(R, R), view(randn(15, 15), 3:(R + 2), 4:(R + 3))]
                    Bs = [randn(M, N), view(randn(15, 15), 2:(M + 1), 5:(N + 4))]
                    flags = (false, :none, nothing)
                    return map(product(As, Bs)) do (A, B)
                        (flags..., BLAS.trmm!, side, ul, tA, dA, randn(), A, B)
                    end
                end,
            ),
        ),

        # trsm!
        vec(
            reduce(
                vcat,
                map(
                    product(['L', 'R'], ['U', 'L'], t_flags, ['N', 'U'], [1, 3], [1, 2])
                ) do (side, ul, tA, dA, M, N)
                    t = tA == 'N'
                    R = side == 'L' ? M : N
                    As = [randn(R, R) + 5I, view(randn(15, 15), 3:(R + 2), 4:(R + 3)) + 5I]
                    Bs = [randn(M, N), view(randn(15, 15), 2:(M + 1), 5:(N + 4))]
                    flags = (false, :none, nothing)
                    return map(product(As, Bs)) do (A, B)
                        (flags..., BLAS.trsm!, side, ul, tA, dA, randn(), A, B)
                    end
                end,
            ),
        ),
    )
    memory = Any[]
    return test_cases, memory
end
