@testset "stack" begin
    @testset "stack functionality" begin
        s = Stack{Float64}()
        push!(s, 5.0)
        @test s.position == 1
        @test s.memory[1] == 5.0
        @test length(s.memory) == 1

        @test pop!(s) == 5.0
        @test s.position == 0
        @test length(s.memory) == 1
    end
end
