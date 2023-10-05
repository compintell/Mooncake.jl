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
        A, dA = primal(_A), shadow(_A)

        @assert M_val === N_val

        # Store the initial state.
        A_mat = wrap_ptr_as_view(A, LDA_val, M_val, N_val)
        A_store = copy(A_mat)

        # Run the primal.
        ccall(
            $(blas_name(fname)), Cvoid, ($TInt, $TInt, Ptr{$elty}, $TInt, $TInt, $TInt),
            M, N, A, LDA, IPIV, INFO,    
        )

        # Zero out the shadow.
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
