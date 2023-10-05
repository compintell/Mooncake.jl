blas_name(name::Symbol) = (Symbol(name, "64_"), Symbol(BLAS.libblastrampoline))

function wrap_ptr_as_view(ptr::Ptr{T}, N::Int, inc::Int) where {T}
    return view(unsafe_wrap(Vector{T}, ptr, N * inc), 1:inc:N*inc)
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

function t_char(x::UInt8)
    x == 0x4e && return 'N'
    x == 0x54 && return 'T'
    x == 0x43 && return 'C'
    throw(error("unrecognised char-code $x"))
end


#
# LEVEL 1
#

for (fname, elty) in ((:cblas_ddot,:Float64), (:cblas_sdot,:Float32))
    @eval function rrule!!(
        ::CoDual{typeof(__foreigncall__)},
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
        args...,
    )
        # Load in values from pointers.
        n, incx, incy = map(primal, (_n, _incx, _incy))
        xinds = 1:incx:incx * n
        yinds = 1:incy:incy * n
        DX = view(unsafe_wrap(Vector{$elty}, primal(_DX), n * incx), xinds)
        DY = view(unsafe_wrap(Vector{$elty}, primal(_DY), n * incy), yinds)

        function ddot_pb!!(dv, d1, d2, d3, d4, d5, d6, dn, dDX, dincx, dDY, dincy, dargs...)
            _dDX = view(unsafe_wrap(Vector{$elty}, dDX, n * incx), xinds)
            _dDY = view(unsafe_wrap(Vector{$elty}, dDY, n * incy), yinds)
            _dDX .+= DY .* dv
            _dDY .+= DX .* dv
            return d1, d2, d3, d4, d5, d6, dn, dDX, dincx, dDY, dincy, dargs...
        end

        # Run primal computation.
        return CoDual(dot(DX, DY), zero($elty)), ddot_pb!!
    end
end

for (fname, elty) in ((:dscal_, :Float64), (:sscal_, :Float32))
    @eval function Taped.rrule!!(
        ::CoDual{typeof(__foreigncall__)},
        ::CoDual{Val{$(blas_name(fname))}},
        ::CoDual, # return type
        ::CoDual, # argument types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        n::CoDual{Ptr{BLAS.BlasInt}},
        DA::CoDual{Ptr{$elty}},
        DX::CoDual{Ptr{$elty}},
        incx::CoDual{Ptr{BLAS.BlasInt}},
        args...,
    )
        # Load in values from pointers, and turn pointers to memory buffers into Vectors.
        _n = unsafe_load(primal(n))
        _incx = unsafe_load(primal(incx))
        _DA = unsafe_load(primal(DA))
        _DX = unsafe_wrap(Vector{$elty}, primal(DX), _n * _incx)
        _DX_s = unsafe_wrap(Vector{$elty}, shadow(DX), _n * _incx)

        inds = 1:_incx:(_incx * _n)
        DX_copy = _DX[inds]
        BLAS.scal!(_n, _DA, _DX, _incx)

        function dscal_pullback!!(_, a, b, c, d, e, f, dn, dDA, dDX, dincx, dargs...)

            # Set primal to previous state.
            _DX[inds] .= DX_copy

            # Compute cotangent w.r.t. scaling.
            unsafe_store!(dDA, BLAS.dot(_n, _DX, _incx, dDX, _incx) + unsafe_load(dDA))

            # Compute cotangent w.r.t. DX.
            BLAS.scal!(_n, _DA, _DX_s, _incx)

            return a, b, c, d, e, f, dn, dDA, dDX, dincx, dargs...
        end
        return CoDual(Cvoid(), zero_tangent(Cvoid())), dscal_pullback!!
    end
end



#
# LEVEL 2
#

for (gemv, elty) in ((:dgemv_, :Float64), (:sgemm_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(Umlaut.__foreigncall__)},
        ::CoDual{Val{$(blas_name(gemv))}},
        ::CoDual,
        ::CoDual,
        ::CoDual,
        ::CoDual,
        _tA::CoDual{Ptr{UInt8}},
        _M::CoDual{Ptr{BLAS.BlasInt}},
        _N::CoDual{Ptr{BLAS.BlasInt}},
        _alpha::CoDual{Ptr{$elty}},
        _A::CoDual{Ptr{$elty}},
        _lda::CoDual{Ptr{BLAS.BlasInt}},
        _x::CoDual{Ptr{$elty}},
        _incx::CoDual{Ptr{BLAS.BlasInt}},
        _beta::CoDual{Ptr{$elty}},
        _y::CoDual{Ptr{$elty}},
        _incy::CoDual{Ptr{BLAS.BlasInt}},
        args...
    )
        # Load in data.
        tA = t_char(unsafe_load(primal(_tA)))
        M, N, lda, incx, incy = map(unsafe_load ∘ primal, (_M, _N, _lda, _incx, _incy))
        alpha = unsafe_load(primal(_alpha))
        beta = unsafe_load(primal(_beta))

        # Run primal.
        A = wrap_ptr_as_view(primal(_A), lda, M, N)
        Nx = tA == 'N' ? N : M
        Ny = tA == 'N' ? M : N
        x = view(unsafe_wrap(Vector{$elty}, primal(_x), incx * Nx), 1:incx:incx * Nx)
        y = view(unsafe_wrap(Vector{$elty}, primal(_y), incy * Ny), 1:incy:incy * Ny)
        y_copy = copy(y)

        BLAS.gemv!(tA, alpha, A, x, beta, y)

        function gemv_pb!!(
            _, d1, d2, d3, d4, d5, d6,
            dt, dM, dN, dalpha, _dA, dlda, _dx, dincx, dbeta, _dy, dincy, dargs...
        )
            # Load up the tangents.
            dA = wrap_ptr_as_view(_dA, lda, M, N)
            dx = view(unsafe_wrap(Vector{$elty}, _dx, incx * Nx), 1:incx:incx * Nx)
            dy = view(unsafe_wrap(Vector{$elty}, _dy, incy * Ny), 1:incy:incy * Ny)

            # Increment the tangents.
            unsafe_store!(dalpha, unsafe_load(dalpha) + dot(dy, _trans(tA, A), x))
            dA .+= _trans(tA, alpha * dy * x')
            dx .+= alpha * _trans(tA, A)'dy
            unsafe_store!(dbeta, unsafe_load(dbeta) + dot(y_copy, dy))
            dy .*= beta

            # Restore the original value of `y`.
            y .= y_copy

            return d1, d2, d3, d4, d5, d6, dt, dM, dN, dalpha, _dA, dlda, _dx, dincx, dbeta,
                _dy, dincy, dargs...
        end
        return uninit_codual(Cvoid()), gemv_pb!!
    end
end

tri!(A, u, d) = u == 'L' ? tril!(A, d == 'U' ? -1 : 0) : triu!(A, d == 'U' ? 1 : 0)

for (trmv, elty) in ((:dtrmv_, :Float64), (:strmv_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(Umlaut.__foreigncall__)},
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
        args...
    )
        # Load in data.
        uplo, trans, diag = map(Char ∘ unsafe_load ∘ primal, (_uplo, _trans, _diag))
        N, lda, incx = map(unsafe_load ∘ primal, (_N, _lda, _incx))
        A = wrap_ptr_as_view(primal(_A), lda, N, N)
        x = wrap_ptr_as_view(primal(_x), N, incx)
        x_copy = copy(x)

        # Run primal computation.
        BLAS.trmv!(uplo, trans, diag, A, x)

        function trmv_pb!!(
            _, d1, d2, d3, d4, d5, d6, du, dt, ddiag, dN, _dA, dlda, _dx, dincx, dargs...
        )
            # Load up the tangents.
            dA = wrap_ptr_as_view(_dA, lda, N, N)
            dx = wrap_ptr_as_view(_dx, N, incx)

            # Restore the original value of x.
            x .= x_copy

            # Increment the tangents.
            dA .+= tri!(trans == 'N' ? dx * x' : x * dx', uplo, diag)
            BLAS.trmv!(uplo, trans == 'N' ? 'T' : 'N', diag, A, dx)

            return d1, d2, d3, d4, d5, d6, du, dt, ddiag, dN, _dA, dlda, _dx, dincx,
                dargs...
        end
        return uninit_codual(Cvoid()), trmv_pb!!
    end
end



#
# LEVEL 3
#

for (gemm, elty) in ((:dgemm_, :Float64), (:sgemm_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(__foreigncall__)},
        ::CoDual{Val{$(blas_name(gemm))}},
        RT::CoDual{Val{Cvoid}},
        AT::CoDual, # arg types
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
        args...,
    )
        _tA = t_char(unsafe_load(primal(tA)))
        _tB = t_char(unsafe_load(primal(tB)))
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

        A_mat = wrap_ptr_as_view(primal(A), _LDA, (_tA == 'N' ? (_m, _ka) : (_ka, _m))...)
        B_mat = wrap_ptr_as_view(primal(B), _LDB, (_tB == 'N' ? (_ka, _n) : (_n, _ka))...)
        C_mat = wrap_ptr_as_view(primal(C), _LDC, _m, _n)
        C_copy = collect(C_mat)

        BLAS.gemm!(_tA, _tB, _alpha, A_mat, B_mat, _beta, C_mat)

        function gemm!_pullback!!(
            _, df, dname, dRT, dAT, dnreq, dconvention,
            dtA, dtB, dm, dn, dka, dalpha, dA, dLDA, dB, dLDB, dbeta, dC, dLDC, dargs...,
        )
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
            dA_mat .+= _alpha * transpose(_trans(_tA, _trans(_tB, B_mat) * transpose(dC_mat)))
            dB_mat .+= _alpha * transpose(_trans(_tB, transpose(dC_mat) * _trans(_tA, A_mat)))
            dC_mat .*= _beta

            return df, dname, dRT, dAT, dnreq, dconvention,
                dtA, dtB, dm, dn, dka, dalpha, dA, dLDA, dB, dLDB, dbeta, dC, dLDC, dargs...
        end
        return CoDual(Cvoid(), zero_tangent(Cvoid())), gemm!_pullback!!
    end
end

for (trmm, elty) in ((:dtrmm_, :Float64), (:strmm_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(__foreigncall__)},
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
        args...,
    )
        # Load in data and store B for the reverse-pass.
        side, ul, tA, diag = map(Char ∘ unsafe_load ∘ primal, (_side, _uplo, _trans, _diag))
        M, N, lda, ldb = map(unsafe_load ∘ primal, (_M, _N, _lda, _ldb))
        alpha = unsafe_load(primal(_alpha))
        R = side == 'L' ? M : N
        A = wrap_ptr_as_view(primal(_A), lda, R, R)
        B = wrap_ptr_as_view(primal(_B), ldb, M, N)
        B_copy = copy(B)

        # Run primal.
        BLAS.trmm!(side, ul, tA, diag, alpha, A, B)

        function trmm!_pullback!!(
            _, d1, d2, d3, d4, d5, d6,
            dside, duplo, dtrans, ddiag, dM, dN, dalpha, _dA, dlda, _dB, dlbd, dargs...,
        )
            # Convert pointers to views.
            dA = wrap_ptr_as_view(_dA, lda, R, R)
            dB = wrap_ptr_as_view(_dB, ldb, M, N)

            # Increment alpha tangent.
            unsafe_store!(dalpha, unsafe_load(dalpha) + dot(dB, B) / alpha)

            # Restore initial state.
            B .= B_copy

            # Increment cotangents.
            if side == 'L'
                dA .+= tri!(tA == 'N' ? dB * B' : B * dB', ul, diag)
            else
                dA .+= tri!(tA == 'N' ? B'dB : dB'B, ul, diag)
            end

            # Compute dB tangent.
            BLAS.trmm!(side, ul, tA == 'N' ? 'N' : 'T', diag, alpha, A, dB)

            return d1, d2, d3, d4, d5, d6,
                dside, duplo, dtrans, ddiag, dM, dN, dalpha, _dA, dlda, _dB, dlbd, dargs...
        end
        return CoDual(Cvoid(), zero_tangent(Cvoid())), trmm!_pullback!!
    end
end
