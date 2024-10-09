@testset "memory" begin
    # Set up memory with an undefined element.
    mem_with_undef = Memory{Memory{Int}}(undef, 2)
    mem_with_undef[1] = fill!(Memory{Int}(undef, 4), 2)

    # Iterate through all test cases.
    @testset "$(typeof(p))" for p in [

        # Memory
        fill!(Memory{Float64}(undef, 10), 0.0),
        fill!(Memory{Int}(undef, 5), 1),
        Memory{Vector{Float64}}([randn(1), randn(3)]),
        Memory{Vector{Float64}}(undef, 3),
        mem_with_undef,

        # MemoryRef
        memoryref(fill!(Memory{Float64}(undef, 10), 0.0)),
    ]
        TestUtils.test_data(sr(123), p)
    end
end
