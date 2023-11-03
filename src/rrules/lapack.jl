for (fname, elty) in ((:dgetrf_, :Float64), (:sgetrf_, :Float32))
    TInt = :(Ptr{BLAS.BlasInt})
    @eval function rrule!!(
        ::CoDual{typeof(__foreigncall__)},
        ::CoDual{Val{$(blas_name(fname))}},
        ::CoDual, # return type
        ::CoDual, # argument types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        _M::CoDual{$TInt}, # Number of rows in matrix A. M >= 0
        _N::CoDual{$TInt}, # Number of cols in matrix A. N >= 0
        _A::CoDual{Ptr{$elty}}, # matrix of size (LDA, N)
        _LDA::CoDual{$TInt}, # leading dimension of A
        _IPIV::CoDual{$TInt}, # pivot indices
        _INFO::CoDual{$TInt}, # some info of some kind
        args...,
    )
        # Extract names.
        M, N, LDA, IPIV, INFO = map(primal, (_M, _N, _LDA, _IPIV, _INFO))
        M_val = unsafe_load(M)
        N_val = unsafe_load(N)
        LDA_val = unsafe_load(LDA)
        data_len = LDA_val * N_val
        A, dA = primal(_A), tangent(_A)

        @assert M_val === N_val

        # Store the initial state.
        A_mat = wrap_ptr_as_view(A, LDA_val, M_val, N_val)
        A_store = copy(A_mat)

        # Run the primal.
        ccall(
            $(blas_name(fname)), Cvoid, ($TInt, $TInt, Ptr{$elty}, $TInt, $TInt, $TInt),
            M, N, A, LDA, IPIV, INFO,    
        )

        # Zero out the tangent.
        foreach(n -> unsafe_store!(dA, zero($elty), n), 1:data_len)

        function getrf_pb!!(
            _, d1, d2, d3, d4, d5, d6, dM, dN, dA, dLDA, dIPIV, dINFO, dargs...
        )
            # Run reverse-pass.
            L, U = UnitLowerTriangular(A_mat), UpperTriangular(A_mat)
            dA_mat = wrap_ptr_as_view(dA, LDA_val, M_val, N_val)
            dL, dU = tril(dA_mat, -1), UpperTriangular(dA_mat)

            # Figure out the pivot matrix used.
            p = LinearAlgebra.ipiv2perm(unsafe_wrap(Array, IPIV, N_val), N_val)

            # Compute pullback using Seth's method.
            __dF = tril(L'dL, -1) + UpperTriangular(dU * U')
            dA_mat .= (inv(L') * __dF * inv(U'))[invperm(p), :]

            # Restore initial state.
            A_mat .= A_store

            return d1, d2, d3, d4, d5, d6, dM, dN, dA, dLDA, dIPIV, dINFO, dargs...
        end
        return CoDual(Cvoid(), zero_tangent(Cvoid())), getrf_pb!!
    end
end

for (fname, elty) in ((:dtrtrs_, :Float64), (:strtrs_, :Float32))

    TInt = :(Ptr{BLAS.BlasInt})
    @eval function rrule!!(
        ::CoDual{typeof(__foreigncall__)},
        ::CoDual{Val{$(blas_name(fname))}},
        ::CoDual, # return type
        ::CoDual, # argument types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        _ul::CoDual{Ptr{UInt8}},
        _tA::CoDual{Ptr{UInt8}},
        _diag::CoDual{Ptr{UInt8}},
        _N::CoDual{Ptr{BLAS.BlasInt}},
        _Nrhs::CoDual{Ptr{BLAS.BlasInt}},
        _A::CoDual{Ptr{$elty}},
        _lda::CoDual{Ptr{BLAS.BlasInt}},
        _B::CoDual{Ptr{$elty}},
        _ldb::CoDual{Ptr{BLAS.BlasInt}},
        _info::CoDual{Ptr{BLAS.BlasInt}},
        args...,
    )
        # Load in data.
        ul_p, tA_p, diag_p = map(primal, (_ul, _tA, _diag))
        N_p, Nrhs_p, lda_p, ldb_p, info_p = map(primal, (_N, _Nrhs, _lda, _ldb, _info))
        ul, tA, diag, N, Nrhs, lda, ldb, info = map(
            unsafe_load, (ul_p, tA_p, diag_p, N_p, Nrhs_p, lda_p, ldb_p, info_p),
        )

        A = wrap_ptr_as_view(primal(_A), lda, N, N)
        B = wrap_ptr_as_view(primal(_B), ldb, N, Nrhs)
        B_copy = copy(B)

        # Run the primal.
        ccall(
            $(blas_name(fname)),
            Cvoid,
            (
                Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt},
                Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt},
                Clong, Clong, Clong,
            ),
            ul_p, tA_p, diag_p, N_p, Nrhs_p, primal(_A), lda_p, primal(_B),ldb_p, info_p,
            1, 1, 1,
        )

        function trtrs_pb!!(
            _, d1, d2, d3, d4, d5, d6,
            dul, dtA, ddiag, dN, dNrhs, _dA, dlda, _dB, dldb, dINFO, dargs...
        )
            # Compute cotangent of B.
            dB = wrap_ptr_as_view(_dB, ldb, N, Nrhs)
            LAPACK.trtrs!(Char(ul), Char(tA) == 'N' ? 'T' : 'N', Char(diag), A, dB)

            # Compute cotangent of A.
            dA = wrap_ptr_as_view(_dA, lda, N, N)
            if Char(tA) == 'N'
                dA .-= tri!(dB * B', Char(ul), Char(diag))
            else
                dA .-= tri!(B * dB', Char(ul), Char(diag))
            end

            # Restore initial state.
            B .= B_copy

            return d1, d2, d3, d4, d5, d6,
                dul, dtA, ddiag, dN, dNrhs, _dA, dlda, _dB, dldb, dINFO, dargs...
        end
        return CoDual(Cvoid(), zero_tangent(Cvoid())), trtrs_pb!!
    end
end

for (fname, elty) in ((:dgetrs_, :Float64), (:sgetrs_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(__foreigncall__)},
        ::CoDual{Val{$(blas_name(fname))}},
        ::CoDual, # return type
        ::CoDual, # argument types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        _tA::CoDual{Ptr{UInt8}},
        _N::CoDual{Ptr{BlasInt}},
        _Nrhs::CoDual{Ptr{BlasInt}},
        _A::CoDual{Ptr{$elty}},
        _lda::CoDual{Ptr{BlasInt}},
        _ipiv::CoDual{Ptr{BlasInt}},
        _B::CoDual{Ptr{$elty}},
        _ldb::CoDual{Ptr{BlasInt}},
        _info::CoDual{Ptr{BlasInt}},
        args...,
    )
        # Load in values.
        tA = Char(unsafe_load(primal(_tA)))
        N, Nrhs, lda, ldb, info = map(unsafe_load ∘ primal, (_N, _Nrhs, _lda, _ldb, _info))
        ipiv = unsafe_wrap(Vector{BlasInt}, primal(_ipiv), N)
        A = wrap_ptr_as_view(primal(_A), lda, N, N)
        B = wrap_ptr_as_view(primal(_B), ldb, N, Nrhs)
        B0 = copy(B)

        # Pivot B.
        p = LinearAlgebra.ipiv2perm(ipiv, N)

        if tA == 'N'
            # Apply permutation matrix.
            B .= B[p, :]

            # Run inv(L) * B and write result to B.
            LAPACK.trtrs!('L', 'N', 'U', A, B)
            B1 = copy(B) # record intermediate state for use in pullback.

            # Run inv(U) * B and write result to B.
            LAPACK.trtrs!('U', 'N', 'N', A, B)
            B2 = B
        else
            # Run inv(U)^T * B and write result to B.
            LAPACK.trtrs!('U', 'T', 'N', A, B)
            B1 = copy(B) # record intermediate state for use in pullback.

            # Run inv(L)^T * B and write result to B.
            LAPACK.trtrs!('L', 'T', 'U', A, B)
            B2 = B

            # Apply permutation matrix.
            B2 .= B2[invperm(p), :]
        end

        # We need to write to `info`.
        unsafe_store!(primal(_info), 0)

        function getrs_pb!!(
            _, d1, d2, d3, d4, d5, d6,
            dtA, dN, dNrhs, _dA, dlda, _ipiv, _dB, dldb, dINFO, dargs...
        )
            dA = wrap_ptr_as_view(_dA, lda, N, N)
            dB = wrap_ptr_as_view(_dB, ldb, N, Nrhs)

            if tA == 'N'

                # Run pullback for inv(U) * B.
                LAPACK.trtrs!('U', 'T', 'N', A, dB)
                dA .-= tri!(dB * B2', 'U', 'N')

                # Run pullback for inv(L) * B.
                LAPACK.trtrs!('L', 'T', 'U', A, dB)
                dA .-= tri!(dB * B1', 'L', 'U')

                # Undo permutation.
                dB .= dB[invperm(p), :]
            else

                # Undo permutation.
                dB .= dB[p, :]
                B2 .= B2[p, :]

                # Run pullback for inv(L^T) * B.
                LAPACK.trtrs!('L', 'N', 'U', A, dB)
                dA .-= tri!(B2 * dB', 'L', 'U')

                # Run pullback for inv(U^T) * B.
                LAPACK.trtrs!('U', 'N', 'N', A, dB)
                dA .-= tri!(B1 * dB', 'U', 'N')
            end

            # Restore initial state.
            B .= B0

            return d1, d2, d3, d4, d5, d6,
                dtA, dN, dNrhs, _dA, dlda, _ipiv, _dB, dldb, dINFO, dargs...
        end
        return CoDual(Cvoid(), zero_tangent(Cvoid())), getrs_pb!!
    end
end

for (fname, elty) in ((:dgetri_, :Float64), (:sgetri_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(__foreigncall__)},
        ::CoDual{Val{$(blas_name(fname))}},
        ::CoDual, # return type
        ::CoDual, # argument types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        _N::CoDual{Ptr{BlasInt}},
        _A::CoDual{Ptr{$elty}},
        _lda::CoDual{Ptr{BlasInt}},
        _ipiv::CoDual{Ptr{BlasInt}},
        _work::CoDual{Ptr{$elty}},
        _lwork::CoDual{Ptr{BlasInt}},
        _info::CoDual{Ptr{BlasInt}},
        args...,
    )
        # Pull out data.
        N_p, lda_p, lwork_p, info_p = map(primal, (_N, _lda, _lwork, _info))
        N, lda, lwork, info = map(unsafe_load, (N_p, lda_p, lwork_p, info_p))
        A_p = primal(_A)
        A = wrap_ptr_as_view(A_p, lda, N, N)
        A_copy = copy(A)

        # Run forwards-pass.
        ccall(
            $(blas_name(fname)), Cvoid,
            (
                Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                Ptr{BlasInt}, Ptr{BlasInt},
            ),
            N_p, A_p, lda_p, primal(_ipiv), primal(_work), lwork_p, info_p,
        )

        p = LinearAlgebra.ipiv2perm(unsafe_wrap(Array, primal(_ipiv), N), N)

        function getri_pb!!(
            _, d1, d2, d3, d4, d5, d6, dN, _dA, dlda, dipiv, dwork, dlwork, dinfo, dargs...
        )
            if lwork != -1
                dA = wrap_ptr_as_view(_dA, lda, N, N)
                A .= A[:, p]
                dA .= dA[:, p]

                # Cotangent w.r.t. L.
                dL = -(A' * dA) / UnitLowerTriangular(A_copy)'
                dU = -(UpperTriangular(A_copy)' \ (dA * A'))
                dA .= tri!(dL, 'L', 'U') .+ tri!(dU, 'U', 'N')

                # Restore initial state.
                A .= A_copy
            end

            return d1, d2, d3, d4, d5, d6,
                dN, _dA, dlda, dipiv, dwork, dlwork, dinfo, dargs...
        end
        return CoDual(Cvoid(), zero_tangent(Cvoid())), getri_pb!!
    end
end

# ccall((@blasfunc($potrf), libblastrampoline), Cvoid,
# (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
# uplo, size(A,1), A, lda, info, 1)

__sym(X) = 0.5 * (X + X')

for (fname, elty) in ((:dpotrf_, :Float64), (:spotrf_, :Float32))
    @eval function rrule!!(
        ::CoDual{typeof(__foreigncall__)},
        ::CoDual{Val{$(blas_name(fname))}},
        ::CoDual, # return type
        ::CoDual, # argument types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        _uplo::CoDual{Ptr{UInt8}},
        _N::CoDual{Ptr{BlasInt}},
        _A::CoDual{Ptr{$elty}},
        _lda::CoDual{Ptr{BlasInt}},
        _info::CoDual{Ptr{BlasInt}},
        args...,
    )
        # Pull out the data.
        uplo_p, N_p, A_p, lda_p, info_p = map(primal, (_uplo, _N, _A, _lda, _info))
        uplo, lda, N = map(unsafe_load, (uplo_p, lda_p, N_p))
        @assert Char(uplo) == 'L'

        # Make a copy of the initial state for later restoration.
        A = wrap_ptr_as_view(A_p, lda, N, N)
        A_copy = copy(A)

        # Run forwards-pass.
        ccall(
            $(blas_name(fname)), Cvoid,
            (Ptr{UInt8}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
            uplo_p, N_p, A_p, lda_p, info_p,
        )

        function potrf_pb!!(
            _, d1, d2, d3, d4, d5, d6, duplo, dN, _dA, dlda, dinfo, dargs...
        )
            dA = wrap_ptr_as_view(_dA, lda, N, N)
            dA2 = copy(dA)

            # Compute cotangents.
            E = LowerTriangular(2 * ones(N, N)) - Diagonal(ones(N))
            L = LowerTriangular(collect(A))
            B = L' \ (E' .* (dA2'L)) / L
            dA .= 0.5 * __sym(B) .* E + triu(dA2, 1)

            # Restore initial state.
            A .= A_copy

            return d1, d2, d3, d4, d5, d6, duplo, dN, _dA, dlda, dinfo, dargs...
        end
        return CoDual(Cvoid(), zero_tangent(Cvoid())), potrf_pb!!
    end
end
