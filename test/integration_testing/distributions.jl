using Distributions

_sym(A) = A'A
_pdmat(A) = PDMat(_sym(A) + 5I)

@testset "distributions" begin
    interp = Tapir.TapirInterpreter()
    @testset "$(typeof(d))" for (interface_only, perf_flag, d, x) in Any[

        #
        # Univariate
        #

        (false, :allocs, Arcsine(), 0.5),
        (false, :allocs, Arcsine(-0.3, 0.9), 0.5),
        (false, :allocs, Arcsine(0.5, 1.1), 1.0),
        (false, :allocs, Beta(1.1, 1.1), 0.5),
        (false, :allocs, Beta(1.1, 1.5), 0.9),
        (false, :allocs, Beta(1.6, 1.5), 0.5),
        (false, :allocs, BetaPrime(1.1, 1.1), 0.5),
        (false, :allocs, BetaPrime(1.1, 1.6), 0.5),
        (false, :allocs, BetaPrime(1.6, 1.3), 0.9),
        (false, :allocs, Biweight(1.0, 2.0), 0.5),
        (false, :allocs, Biweight(-0.5, 2.5), -0.45),
        (false, :allocs, Biweight(0.0, 1.0), 0.3),
        (false, :allocs, Cauchy(), -0.5),
        (false, :allocs, Cauchy(1.0), 0.99),
        (false, :allocs, Cauchy(1.0, 0.1), 1.01),
        (false, :allocs, Chi(2.5), 0.5),
        (false, :allocs, Chi(5.5), 1.1),
        (false, :allocs, Chi(0.1), 0.7),
        (false, :allocs, Chisq(2.5), 0.5),
        (false, :allocs, Chisq(5.5), 1.1),
        (false, :allocs, Chisq(0.1), 0.7),
        (false, :allocs, Cosine(0.0, 1.0), 0.5),
        (false, :allocs, Cosine(-0.5, 2.0), -0.1),
        (false, :allocs, Cosine(0.4, 0.5), 0.0),
        (false, :allocs, Epanechnikov(0.0, 1.0), 0.5),
        (false, :allocs, Epanechnikov(-0.5, 1.2), -0.9),
        (false, :allocs, Epanechnikov(-0.4, 1.6), 0.1),
        (false, :allocs, Erlang(), 0.5),
        (false, :allocs, Erlang(), 0.1),
        (false, :allocs, Erlang(), 0.9),
        (false, :allocs, Exponential(), 0.1),
        (false, :allocs, Exponential(0.5), 0.9),
        (false, :allocs, Exponential(1.4), 0.05),
        (false, :allocs, FDist(2.1, 3.5), 0.7),
        (false, :allocs, FDist(1.4, 5.4), 3.5),
        (false, :allocs, FDist(5.5, 3.3), 7.2),
        (false, :allocs, Frechet(), 0.1),
        (false, :allocs, Frechet(), 1.1),
        (false, :allocs, Frechet(1.5, 2.4), 0.1),
        (false, :allocs, Gamma(0.9, 1.2), 4.5),
        (false, :allocs, Gamma(0.5, 1.9), 1.5),
        (false, :allocs, Gamma(1.8, 3.2), 1.0),
        (false, :allocs, GeneralizedExtremeValue(0.3, 1.3, 0.1), 2.4),
        (false, :allocs, GeneralizedExtremeValue(-0.7, 2.2, 0.4), 1.1),
        (false, :allocs, GeneralizedExtremeValue(0.5, 0.9, -0.5), -7.0),
        (false, :allocs, GeneralizedPareto(0.3, 1.1, 1.1), 5.0),
        (false, :allocs, GeneralizedPareto(-0.25, 0.9, 0.1), 0.8),
        (false, :allocs, GeneralizedPareto(0.3, 1.1, -5.1), 0.31),
        (false, :allocs, Gumbel(0.1, 0.5), 0.1),
        (false, :allocs, Gumbel(-0.5, 1.1), -0.1),
        (false, :allocs, Gumbel(0.3, 0.1), 0.3),
        (false, :allocs, InverseGaussian(0.1, 0.5), 1.1),
        (false, :allocs, InverseGaussian(0.2, 1.1), 3.2),
        (false, :allocs, InverseGaussian(0.1, 1.2), 0.5),
        (false, :allocs, JohnsonSU(0.1, 0.95, 0.1, 1.1), 0.1),
        (false, :allocs, JohnsonSU(0.15, 0.9, 0.12, 0.94), 0.5),
        (false, :allocs, JohnsonSU(0.1, 0.95, 0.1, 1.1), -0.3),
        (false, :allocs, Kolmogorov(), 1.1),
        (false, :allocs, Kolmogorov(), 0.9),
        (false, :allocs, Kolmogorov(), 1.5),
        (false, :allocs, Kumaraswamy(2.0, 5.0), 0.71),
        (false, :allocs, Kumaraswamy(0.1, 5.0), 0.2),
        (false, :allocs, Kumaraswamy(0.5, 4.5), 0.1),
        (false, :allocs, Laplace(0.1, 1.0), 0.2),
        (false, :allocs, Laplace(-0.5, 2.1), 0.5),
        (false, :allocs, Laplace(-0.35, 0.4), -0.3),
        (false, :allocs, Levy(0.1, 0.9), 4.1),
        (false, :allocs, Levy(0.5, 0.9), 0.6),
        (false, :allocs, Levy(1.1, 0.5), 2.2),
        (false, :allocs, Lindley(0.5), 2.1),
        (false, :allocs, Lindley(1.1), 3.1),
        (false, :allocs, Lindley(1.9), 3.5),
        (false, :allocs, Logistic(0.1, 1.2), 1.1),
        (false, :allocs, Logistic(0.5, 0.7), 0.6),
        (false, :allocs, Logistic(-0.5, 0.1), -0.4),
        (false, :allocs, LogitNormal(0.1, 1.1), 0.5),
        (false, :allocs, LogitNormal(0.5, 0.7), 0.6),
        (false, :allocs, LogitNormal(-0.12, 1.1), 0.1),
        (false, :allocs, LogNormal(0.0, 1.0), 0.5),
        (false, :allocs, LogNormal(0.5, 1.0), 0.5),
        (false, :allocs, LogNormal(-0.1, 1.3), 0.75),
        (false, :allocs, LogUniform(0.1, 0.9), 0.75),
        (false, :allocs, LogUniform(0.15, 7.8), 7.1),
        (false, :allocs, LogUniform(2.0, 3.0), 2.1),
        # (false, :none, NoncentralBeta(1.1, 1.1, 1.2), 0.8), # foreigncall (Rmath.dnbeta). Not implemented anywhere.
        # (false, :none, NoncentralChisq(2, 3.0), 10.0), # foreigncall (Rmath.dnchisq). Not implemented anywhere.
        # (false, :none, NoncentralF(2, 3, 1.1), 4.1), # foreigncall (Rmath.dnf). Not implemented anywhere.
        # (false, :none, NoncentralT(1.3, 1.1), 0.1), # foreigncall (Rmath.dnt). Not implemented anywhere.
        (false, :allocs, Normal(), 0.1),
        (false, :allocs, Normal(0.0, 1.0), 1.0),
        (false, :allocs, Normal(0.5, 1.0), 0.05),
        (false, :allocs, Normal(0.0, 1.5), -0.1),
        (false, :allocs, Normal(-0.1, 0.9), -0.3),
        # (false, :none NormalInverseGaussian(0.0, 1.0, 0.2, 0.1), 0.1), # foreigncall -- https://github.com/JuliaMath/SpecialFunctions.jl/blob/be1fa06fee58ec019a28fb0cd2b847ca83a5af9a/src/bessel.jl#L265
        (false, :allocs, Pareto(1.0, 1.0), 3.5),
        (false, :allocs, Pareto(1.1, 0.9), 3.1),
        (false, :allocs, Pareto(1.0, 1.0), 1.4),
        (false, :allocs, PGeneralizedGaussian(0.2), 5.0),
        (false, :allocs, PGeneralizedGaussian(0.5, 1.0, 0.3), 5.0),
        (false, :allocs, PGeneralizedGaussian(-0.1, 11.1, 6.5), -0.3),
        (false, :allocs, Rayleigh(0.5), 0.6),
        (false, :allocs, Rayleigh(0.9), 1.1),
        (false, :allocs, Rayleigh(0.55), 0.63),
        # (false, :none, Rician(0.5, 1.0), 2.1), # foreigncall (Rmath.dnchisq). Not implemented anywhere.
        (false, :allocs, Semicircle(1.0), 0.9),
        (false, :allocs, Semicircle(5.1), 5.05),
        (false, :allocs, Semicircle(0.5), -0.1),
        (false, :allocs, SkewedExponentialPower(0.1, 1.0, 0.97, 0.7), -2.0),
        (false, :allocs, SkewedExponentialPower(0.15, 1.0, 0.97, 0.7), -2.0),
        (false, :allocs, SkewedExponentialPower(0.1, 1.1, 0.99, 0.7), 0.5),
        (false, :allocs, SkewNormal(0.0, 1.0, -1.0), 0.1),
        (false, :allocs, SkewNormal(0.5, 2.0, 1.1), 0.1),
        (false, :allocs, SkewNormal(-0.5, 1.0, 0.0), 0.1),
        (false, :allocs, SymTriangularDist(0.0, 1.0), 0.5),
        (false, :allocs, SymTriangularDist(-0.5, 2.1), -2.0),
        (false, :allocs, SymTriangularDist(1.7, 0.3), 1.75),
        (false, :allocs, TDist(1.1), 99.1),
        (false, :allocs, TDist(10.1), 25.0),
        (false, :allocs, TDist(2.1), -89.5),
        (false, :allocs, TriangularDist(0.0, 1.5, 0.5), 0.45),
        (false, :allocs, TriangularDist(0.1, 1.4, 0.45), 0.12),
        (false, :allocs, TriangularDist(0.0, 1.5, 0.5), 0.2),
        (false, :allocs, Triweight(1.0, 1.0), 1.0),
        (false, :allocs, Triweight(1.1, 2.1), 1.0),
        (false, :allocs, Triweight(1.9, 10.0), -0.1),
        (false, :allocs, Uniform(0.0, 1.0), 0.2),
        (false, :allocs, Uniform(-0.1, 1.1), 1.0),
        (false, :allocs, Uniform(99.5, 100.5), 100.0),
        (false, :allocs, VonMises(0.5), 0.1),
        (false, :allocs, VonMises(0.3), -0.1),
        (false, :allocs, VonMises(0.2), -0.5),
        (false, :allocs, Weibull(0.5, 1.0), 0.45),
        (false, :allocs, Weibull(0.3, 1.1), 0.66),
        (false, :allocs, Weibull(0.75, 1.3), 0.99),

        #
        # Multivariate
        #

        (false, :allocs, MvNormal(1, 1.5), [-0.3]),
        (false, :allocs, MvNormal(2, 0.5), [0.2, -0.3]),
        (false, :allocs, MvNormal([1.0]), [-0.1]),
        (false, :allocs, MvNormal([1.0, 0.9]), [-0.1, -0.7]),
        (false, :none, MvNormal([0.0], 0.9), [0.1]),
        (false, :none, MvNormal([0.0, 0.1], 0.9), [0.1, -0.05]),
        (false, :allocs, MvNormal(Diagonal([0.1])), [0.1]),
        (false, :allocs, MvNormal(Diagonal([0.1, 0.2])), [0.1, 0.15]),
        (false, :none, MvNormal([0.1, -0.3], Diagonal(Fill(0.9, 2))), [0.1, -0.1]),
        (false, :none, MvNormal([0.1, -0.1], 0.4I), [-0.1, 0.15]),
        (false, :none, MvNormal([0.2, 0.3], Hermitian(Diagonal([0.5, 0.4]))), [-0.1, 0.05]),
        (false, :none, MvNormal([0.2, 0.3], Symmetric(Diagonal([0.5, 0.4]))), [-0.1, 0.05]),
        (false, :none, MvNormal([0.2, 0.3], Diagonal([0.5, 0.4])), [-0.1, 0.05]),
        (false, :none, MvNormal([-0.15], _pdmat([1.1]')), [-0.05]),
        (false, :none, MvNormal([0.2, -0.15], _pdmat([1.0 0.9; 0.7 1.1])), [0.05, -0.05]),
        (false, :none, MvNormal([0.2, -0.3], [0.5, 0.6]), [0.4, -0.3]),
        (false, :none, MvNormalCanon([0.1, -0.1], _pdmat([0.5 0.4; 0.45 1.0])), [0.2, -0.25]),
        (false, :none, MvLogNormal(MvNormal([0.2, -0.1], _pdmat([1.0 0.9; 0.7 1.1]))), [0.5, 0.1]),
        (false, :none, product_distribution([Normal()]), [0.3]),
        (false, :none, product_distribution([Normal(), Uniform()]), [-0.4, 0.3]),

        #
        # Matrix-variate
        #

        (
            false,
            :none,
            MatrixNormal(
                randn(sr(0), 2, 3), _pdmat(randn(sr(1), 2, 2)), _pdmat(randn(sr(2), 3, 3))
            ),
            randn(sr(4), 2, 3),
        ),
        (
            false,
            :none,
            Wishart(5, _pdmat(randn(sr(5), 3, 3))),
            Symmetric(collect(_pdmat(randn(sr(6), 3, 3)))),
        ),
        (
            false,
            :none,
            InverseWishart(5, _pdmat(randn(sr(7), 3, 3))),
            Symmetric(collect(_pdmat(randn(sr(8), 3, 3)))),
        ),
        (
            false,
            :none,
            MatrixTDist(
                3.1,
                randn(sr(9), 2, 3),
                _pdmat(randn(sr(0), 2, 2)),
                _pdmat(randn(sr(1), 3, 3)),
            ),
            randn(sr(2), 2, 3),
        ),
        (
            false,
            :none,
            MatrixBeta(5, 6.0, 7.0),
            rand(sr(123456), MatrixBeta(5, 6.0, 6.0)),
        ),
        (
            false,
            :none,
            MatrixFDist(6.0, 7.0, _pdmat(randn(sr(1234), 5, 5))),
            rand(sr(13), MatrixFDist(6.0, 7.0, _pdmat(randn(StableRNG(11), 5, 5)))),
        ),
        (false, :none, LKJ(5, 1.1), rand(sr(123456), LKJ(5, 1.1))),
    ]
        @info "$(map(typeof, (d, x)))"
        TestUtils.test_rule(
            sr(123456), logpdf, d, x;
            interp, perf_flag, interface_only, is_primitive=false,
        )
    end
    @testset "$name" for (interface_only, perf_flag, name, f, x) in Any[
        (false, :allocs, "InverseGamma", (a, b, x) -> logpdf(InverseGamma(a, b), x), (1.5, 1.4, 0.4)),
        (false, :allocs, "NormalCanon", (m, s, x) -> logpdf(NormalCanon(m, s), x), (0.1, 1.0, -0.5)),
        (false, :none, "Categorical", x -> logpdf(Categorical(x, 1 - x), 1), 0.3),
        (
            false,
            :none,
            "MvLogitNormal",
            (m, S, x) -> logpdf(MvLogitNormal(m, S), vcat(x, 1 - sum(x))),
            ([0.4, 0.6], Symmetric(_pdmat([0.9 0.4; 0.5 1.1])), [0.27, 0.24]),
        ),
        (
            false,
            :allocs,
            "truncated Beta",
            (a, b, α, β, x) -> logpdf(truncated(Beta(α, β), a, b), x),
            (0.1, 0.9, 1.1, 1.3, 0.4),
        ),
        (
            false,
            :none,
            "allocs Normal",
            (a, b, x) -> logpdf(truncated(Normal(), a, b), x),
            (-0.3, 0.3, 0.1),
        ),
        (
            false,
            :none,
            "allocs Uniform",
            (a, b, α, β, x) -> logpdf(truncated(Uniform(α, β), a, b), x),
            (0.1, 0.9, -0.1, 1.1, 0.4),
        ),
        (false, :none, "Dirichlet", (a, x) -> logpdf(Dirichlet(a), [x, 1-x]), ([1.5, 1.1], 0.6)),
        (
            false,
            :none,
            "reshape",
            x -> logpdf(reshape(product_distribution([Normal(), Uniform()]), 1, 2), x),
            ([2.1 0.7],),
        ),
        (
            false,
            :none,
            "vec",
            x -> logpdf(vec(LKJ(2, 1.1)), x),
            ([1.0, 0.489, 0.489, 1.0],),
        ),
    ]
        @info "$name"
        TestUtils.test_rule(
            sr(123456), f, x...;
            interp, perf_flag=perf_flag, interface_only, is_primitive=false,
        )
    end
end
