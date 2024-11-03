@testset "stack" begin
    @testset "stack functionality $Inc" for Inc in [1, 8]
        s = Stack{Float64, Inc}()
        push!(s, 5.0)
        @test s.position == 1
        @test s.memory[1] == 5.0
        @test length(s.memory) == Inc

        @test pop!(s) == 5.0
        @test s.position == 0
        @test length(s.memory) == Inc
    end
end
