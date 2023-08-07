blas_name(name::Symbol) = (Symbol(name, "64_"), Symbol(BLAS.libblastrampoline))

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

function wrap_ptr_as_view(ptr::Ptr{T}, buffer_nrows::Int, nrows::Int, ncols::Int) where {T}
    return view(unsafe_wrap(Matrix{T}, ptr, (buffer_nrows, ncols)), 1:nrows, :)
end

for (gemm, elty) in (
    (:dgemm_, :Float64),
    (:sgemm_, :Float32),
)
    @eval function rrule!!(
        ::CoDual{typeof(__foreigncall__)},
        ::CoDual{Val{$(blas_name(gemm))}},
        RT::CoDual{Val{Cvoid}},
        AT::CoDual, # arg types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        tA::CoDual{Ptr{UInt8}},
        tB::CoDual{Ptr{UInt8}},
        m::CoDual{Ptr{Int}},
        n::CoDual{Ptr{Int}},
        ka::CoDual{Ptr{Int}},
        alpha::CoDual{Ptr{$elty}},
        A::CoDual{Ptr{$elty}},
        LDA::CoDual{Ptr{Int}},
        B::CoDual{Ptr{$elty}},
        LDB::CoDual{Ptr{Int}},
        beta::CoDual{Ptr{$elty}},
        C::CoDual{Ptr{$elty}},
        LDC::CoDual{Ptr{Int}},
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
