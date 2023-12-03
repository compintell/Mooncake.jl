@testset "foreigncall" begin
    _x = Ref(5.0) # data used in tests which aren't protected by GC.
    _dx = randn_tangent(Xoshiro(123456), _x)

    _a, _da = randn(5), randn(5)
    _b, _db = randn(4), randn(4)
    ptr_a, ptr_da = pointer(_a), pointer(_da)
    ptr_b, ptr_db = pointer(_b), pointer(_db)

    function unsafe_copyto_tester(x::Vector{T}, y::Vector{T}, n::Int) where {T}
        GC.@preserve x y unsafe_copyto!(pointer(x), pointer(y), n)
        return x
    end

    TestUtils.run_hand_written_rrule!!_test_cases(StableRNG, Val(:foreigncall))

    @testset "$f, $(typeof(x))" for (interface_only, f, x...) in [
        (false, reshape, randn(5, 4), (4, 5)),
        (false, reshape, randn(5, 4), (2, 10)),
        (false, reshape, randn(5, 4), (10, 2)),
        (false, reshape, randn(5, 4), (5, 4, 1)),
        (false, reshape, randn(5, 4), (2, 10, 1)),
        (false, unsafe_copyto_tester, randn(5), randn(3), 2),
        (false, unsafe_copyto_tester, randn(5), randn(6), 4),
        (false, unsafe_copyto_tester, [randn(3) for _ in 1:5], [randn(4) for _ in 1:6], 4),
        (false, x -> unsafe_pointer_to_objref(pointer_from_objref(x)), _x),
        (false, isassigned, randn(5), 4),
        (false, x -> (Base._growbeg!(x, 2); x[1:2] .= 2.0), randn(5)),
    ]
        test_taped_rrule!!(sr(123456), f, deepcopy(x)...; interface_only, perf_flag=:none)
    end
    @testset "foreigncalls that should never be hit: $name" for name in [
        :jl_alloc_array_1d, :jl_alloc_array_2d, :jl_alloc_array_3d, :jl_new_array,
        :jl_array_copy, :jl_type_intersection, :memset, :jl_get_tls_world_age, :memmove,
        :jl_object_id, :jl_array_sizehint, :jl_array_grow_beg, :jl_array_grow_end,
        :jl_array_grow_at, :jl_array_del_beg, :jl_array_del_end, :jl_array_del_at,
    ]
        @test_throws(
            ErrorException,
            Taped.rrule!!(
                CoDual(Umlaut.__foreigncall__, NoTangent()),
                CoDual(Val(name), NoTangent()),
            )
        )
    end
end
