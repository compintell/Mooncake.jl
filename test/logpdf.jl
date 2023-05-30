function model()
    x = assume(Normal(0.0, 1.0), :x)
    y = assume(Normal(x, 1.0), :y)
end

@testset "logpdf" begin
    vals = (x=0.0, y=0.0)
    _, tape = Taped.trace(model; ctx=Taped.LPC())
    lpdf_tape = Taped.to_logpdf(tape)
    Umlaut.inputs!(lpdf_tape, vals, model)
    @test isapprox(
        play!(lpdf_tape, vals, model)[],
        logpdf(Normal(0.0, 1.0), vals.x) + logpdf(Normal(vals.x, 1.0), vals.y),
    )
end
