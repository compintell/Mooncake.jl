# Functionality to make testing straightforward.
# for T in [
#     Arcsine, Beta, BetaPrime, Biweight, Cauchy, Chi, Chisq, Chernoff, Cosine, Epanechnikov,
#     Erlang, Exponential, FDist, Frechet, Gamma, GeneralizedExtremeValue, GeneralizedPareto,
#     Gumbel, InverseGaussian, JohnsonSU, Kumaraswamy, Laplace, Levy, Lindley, Logistic,
#     LogitNormal, LogNormal, LogUniform, NoncentralBeta, NoncentralChisq, NoncentralF,
#     NoncentralT, Normal, NormalInverseGaussian, Pareto, PGeneralizedGaussian, Rayleigh,
#     Rician, Semicircle, SkewedExponentialPower, SkewNormal, SymTriangularDist, TDist,
#     TriangularDist, Triweight, Uniform, VonMises, Weibull, MvNormal, Distributions.Zeros,
#     Distributions.ScalMat, Distributions.PDiagMat, PDMat, Cholesky, MvNormalCanon,
#     MvLogitNormal, MvLogNormal, Product, MatrixBeta, Wishart, MatrixFDist, LKJ,
#     Symmetric, MatrixNormal, InverseWishart, MatrixTDist,
# ]
#     @eval Taped._add_to_primal(p::$T, t) = Taped._containerlike_add_to_primal(p, t)
#     @eval Taped._diff(p::$T, q::$T) = Taped._containerlike_diff(p, q)
# end

_sym(A) = A'A
_pdmat(A) = PDMat(_sym(A) + 5I)

@testset "distributions" begin
    @testset "$(typeof(d))" for (interface_only, d, x) in [

        #
        # Univariate
        #

        (false, Arcsine(), 0.5),
        (false, Arcsine(-0.3, 0.9), 0.5),
        (false, Arcsine(0.5, 1.1), 1.0),
        (false, Beta(1.1, 1.1), 0.5),
        (false, Beta(1.1, 1.5), 0.9),
        (false, Beta(1.6, 1.5), 0.5),
        (false, BetaPrime(1.1, 1.1), 0.5),
        (false, BetaPrime(1.1, 1.6), 0.5),
        (false, BetaPrime(1.6, 1.3), 0.9),
        (false, Biweight(1.0, 2.0), 0.5),
        (false, Biweight(-0.5, 2.5), -0.45),
        (false, Biweight(0.0, 1.0), 0.3),
        (false, Cauchy(), -0.5),
        (false, Cauchy(1.0), 0.99),
        (false, Cauchy(1.0, 0.1), 1.01),
        # (false, Chernoff(), 0.5), takes forever
        (false, Chi(2.5), 0.5),
        (false, Chi(5.5), 1.1),
        (false, Chi(0.1), 0.7),
        (false, Chisq(2.5), 0.5),
        (false, Chisq(5.5), 1.1),
        (false, Chisq(0.1), 0.7),
        (false, Cosine(0.0, 1.0), 0.5),
        (false, Cosine(-0.5, 2.0), -0.1),
        (false, Cosine(0.4, 0.5), 0.0),
        (false, Epanechnikov(0.0, 1.0), 0.5),
        (false, Epanechnikov(-0.5, 1.2), -0.9),
        (false, Epanechnikov(-0.4, 1.6), 0.1),
        (false, Erlang(), 0.5),
        (false, Erlang(), 0.1),
        (false, Erlang(), 0.9),
        (false, Exponential(), 0.1),
        (false, Exponential(0.5), 0.9),
        (false, Exponential(1.4), 0.05),
        (false, FDist(2.1, 3.5), 0.7),
        (false, FDist(1.4, 5.4), 3.5),
        (false, FDist(5.5, 3.3), 7.2),
        (false, Frechet(), 0.1),
        (false, Frechet(), 1.1),
        (false, Frechet(1.5, 2.4), 0.1),
        (false, Gamma(0.9, 1.2), 4.5),
        (false, Gamma(0.5, 1.9), 1.5),
        (false, Gamma(1.8, 3.2), 1.0),
        (false, GeneralizedExtremeValue(0.3, 1.3, 0.1), 2.4),
        (false, GeneralizedExtremeValue(-0.7, 2.2, 0.4), 1.1),
        (false, GeneralizedExtremeValue(0.5, 0.9, -0.5), -7.0),
        (false, GeneralizedPareto(0.3, 1.1, 1.1), 5.0),
        (false, GeneralizedPareto(-0.25, 0.9, 0.1), 0.8),
        (false, GeneralizedPareto(0.3, 1.1, -5.1), 0.31),
        (false, Gumbel(0.1, 0.5), 0.1),
        (false, Gumbel(-0.5, 1.1), -0.1),
        (false, Gumbel(0.3, 0.1), 0.3),
        (false, InverseGaussian(0.1, 0.5), 1.1),
        (false, InverseGaussian(0.2, 1.1), 3.2),
        (false, InverseGaussian(0.1, 1.2), 0.5),
        (false, JohnsonSU(0.1, 0.95, 0.1, 1.1), 0.1),
        (false, JohnsonSU(0.15, 0.9, 0.12, 0.94), 0.5),
        (false, JohnsonSU(0.1, 0.95, 0.1, 1.1), -0.3),
        (false, Kolmogorov(), 1.1),
        (false, Kolmogorov(), 0.9),
        (false, Kolmogorov(), 1.5),
        (false, Kumaraswamy(2.0, 5.0), 0.71),
        (false, Kumaraswamy(0.1, 5.0), 0.2),
        (false, Kumaraswamy(0.5, 4.5), 0.1),
        (false, Laplace(0.1, 1.0), 0.2),
        (false, Laplace(-0.5, 2.1), 0.5),
        (false, Laplace(-0.35, 0.4), -0.3),
        (false, Levy(0.1, 0.9), 4.1),
        (false, Levy(0.5, 0.9), 0.6),
        (false, Levy(1.1, 0.5), 2.2),
        (false, Lindley(0.5), 2.1),
        (false, Lindley(1.1), 3.1),
        (false, Lindley(1.9), 3.5),
        (false, Logistic(0.1, 1.2), 1.1),
        (false, Logistic(0.5, 0.7), 0.6),
        (false, Logistic(-0.5, 0.1), -0.4),
        (false, LogitNormal(0.1, 1.1), 0.5),
        (false, LogitNormal(0.5, 0.7), 0.6),
        (false, LogitNormal(-0.12, 1.1), 0.1),
        (false, LogNormal(0.0, 1.0), 0.5),
        (false, LogNormal(0.5, 1.0), 0.5),
        (false, LogNormal(-0.1, 1.3), 0.75),
        (false, LogUniform(0.1, 0.9), 0.75),
        (false, LogUniform(0.15, 7.8), 7.1),
        (false, LogUniform(2.0, 3.0), 2.1),
        # (false, NoncentralBeta(1.1, 1.1, 1.2), 0.8), # foreigncall (Rmath.dnbeta). Not implemented anywhere.
        # (false, NoncentralChisq(2, 3.0), 10.0), # foreigncall (Rmath.dnchisq). Not implemented anywhere.
        # (false, NoncentralF(2, 3, 1.1), 4.1), # foreigncall (Rmath.dnf). Not implemented anywhere.
        # (false, NoncentralT(1.3, 1.1), 0.1), # foreigncall (Rmath.dnt). Not implemented anywhere.
        (false, Normal(), 0.1),
        (false, Normal(0.0, 1.0), 1.0),
        (false, Normal(0.5, 1.0), 0.05),
        (false, Normal(0.0, 1.5), -0.1),
        (false, Normal(-0.1, 0.9), -0.3),
        # (false, NormalInverseGaussian(0.0, 1.0, 0.2, 0.1), 0.1), # foreigncall -- https://github.com/JuliaMath/SpecialFunctions.jl/blob/be1fa06fee58ec019a28fb0cd2b847ca83a5af9a/src/bessel.jl#L265
        (false, Pareto(1.0, 1.0), 3.5),
        (false, Pareto(1.1, 0.9), 3.1),
        (false, Pareto(1.0, 1.0), 1.4),
        (false, PGeneralizedGaussian(0.2), 5.0),
        (false, PGeneralizedGaussian(0.5, 1.0, 0.3), 5.0),
        (false, PGeneralizedGaussian(-0.1, 11.1, 6.5), -0.3),
        (false, Rayleigh(0.5), 0.6),
        (false, Rayleigh(0.9), 1.1),
        (false, Rayleigh(0.55), 0.63),
        # (false, Rician(0.5, 1.0), 2.1), # foreigncall (Rmath.dnchisq). Not implemented anywhere.
        (false, Semicircle(1.0), 0.9),
        (false, Semicircle(5.1), 5.05),
        (false, Semicircle(0.5), -0.1),
        (false, SkewedExponentialPower(0.1, 1.0, 0.97, 0.7), -2.0),
        (false, SkewedExponentialPower(0.15, 1.0, 0.97, 0.7), -2.0),
        (false, SkewedExponentialPower(0.1, 1.1, 0.99, 0.7), 0.5),
        (false, SkewNormal(0.0, 1.0, -1.0), 0.1),
        (false, SkewNormal(0.5, 2.0, 1.1), 0.1),
        (false, SkewNormal(-0.5, 1.0, 0.0), 0.1),
        (false, SymTriangularDist(0.0, 1.0), 0.5),
        (false, SymTriangularDist(-0.5, 2.1), -2.0),
        (false, SymTriangularDist(1.7, 0.3), 1.75),
        (false, TDist(1.1), 99.1),
        (false, TDist(10.1), 25.0),
        (false, TDist(2.1), -89.5),
        (false, TriangularDist(0.0, 1.5, 0.5), 0.45),
        (false, TriangularDist(0.1, 1.4, 0.45), 0.12),
        (false, TriangularDist(0.0, 1.5, 0.5), 0.2),
        (false, Triweight(1.0, 1.0), 1.0),
        (false, Triweight(1.1, 2.1), 1.0),
        (false, Triweight(1.9, 10.0), -0.1),
        (false, Uniform(0.0, 1.0), 0.2),
        (false, Uniform(-0.1, 1.1), 1.0),
        (false, Uniform(99.5, 100.5), 100.0),
        (false, VonMises(0.5), 0.1),
        (false, VonMises(0.3), -0.1),
        (false, VonMises(0.2), -0.5),
        (false, Weibull(0.5, 1.0), 0.45),
        (false, Weibull(0.3, 1.1), 0.66),
        (false, Weibull(0.75, 1.3), 0.99),

        #
        # Multivariate
        #

        (false, MvNormal(1, 1.5), [-0.3]),
        (false, MvNormal(2, 0.5), [0.2, -0.3]),
        (false, MvNormal([1.0]), [-0.1]),
        (false, MvNormal([1.0, 0.9]), [-0.1, -0.7]),
        (false, MvNormal([0.0], 0.9), [0.1]),
        (false, MvNormal([0.0, 0.1], 0.9), [0.1, -0.05]),
        (false, MvNormal(Diagonal([0.1])), [0.1]),
        (false, MvNormal(Diagonal([0.1, 0.2])), [0.1, 0.15]),
        (false, MvNormal([0.1, -0.3], Diagonal(Fill(0.9, 2))), [0.1, -0.1]),
        (false, MvNormal([0.1, -0.1], 0.4I), [-0.1, 0.15]),
        (false, MvNormal([0.2, 0.3], Hermitian(Diagonal([0.5, 0.4]))), [-0.1, 0.05]),
        (false, MvNormal([0.2, 0.3], Symmetric(Diagonal([0.5, 0.4]))), [-0.1, 0.05]),
        (false, MvNormal([0.2, 0.3], Diagonal([0.5, 0.4])), [-0.1, 0.05]),
        (false, MvNormal([-0.15], _pdmat([1.1]')), [-0.05]),
        (false, MvNormal([0.2, -0.15], _pdmat([1.0 0.9; 0.7 1.1])), [0.05, -0.05]),
        (false, MvNormal([0.2, -0.3], [0.5, 0.6]), [0.4, -0.3]),
        (false, MvNormalCanon([0.1, -0.1], _pdmat([0.5 0.4; 0.45 1.0])), [0.2, -0.25]),
        (false, MvLogNormal(MvNormal([0.2, -0.1], _pdmat([1.0 0.9; 0.7 1.1]))), [0.5, 0.1]),
        (false, Product([Normal()]), [0.3]),
        (false, Product([Normal(), Uniform()]), [-0.4, 0.3]),

        #
        # Matrix-variate
        #

        (
            false,
            MatrixNormal(
                randn(sr(0), 2, 3), _pdmat(randn(sr(1), 2, 2)), _pdmat(randn(sr(2), 3, 3))
            ),
            randn(sr(4), 2, 3),
        ),
        (
            false,
            Wishart(5, _pdmat(randn(sr(5), 3, 3))),
            Symmetric(collect(_pdmat(randn(sr(6), 3, 3)))),
        ),
        (
            false,
            InverseWishart(5, _pdmat(randn(sr(7), 3, 3))),
            Symmetric(collect(_pdmat(randn(sr(8), 3, 3)))),
        ),
        (
            false,
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
            MatrixBeta(5, 6.0, 7.0),
            rand(sr(123456), MatrixBeta(5, 6.0, 6.0)),
        ),
        (
            false,
            MatrixFDist(6.0, 7.0, _pdmat(randn(sr(1234), 5, 5))),
            rand(sr(13), MatrixFDist(6.0, 7.0, _pdmat(randn(StableRNG(11), 5, 5)))),
        ),
        (false, LKJ(5, 1.1), rand(sr(123456), LKJ(5, 1.1))),
    ]
        @info "$(map(typeof, (d, x)))"
        rng = StableRNG(123456)
        test_taped_rrule!!(rng, logpdf, d, deepcopy(x); interface_only, perf_flag=:none)
    end
    @testset "$name" for (name, f, x) in [
        ("InverseGamma", (a, b, x) -> logpdf(InverseGamma(a, b), x), (1.5, 1.4, 0.4)),
        ("NormalCanon", (m, s, x) -> logpdf(NormalCanon(m, s), x), (0.1, 1.0, -0.5)),
        (
            "MvLogitNormal",
            (m, S, x) -> logpdf(MvLogitNormal(m, S), vcat(x, 1 - sum(x))),
            ([0.4, 0.6], Symmetric(_pdmat([0.9 0.4; 0.5 1.1])), [0.27, 0.24]),
        ),
        (
            "truncated Beta",
            (a, b, α, β, x) -> logpdf(truncated(Beta(α, β), a, b), x),
            (0.1, 0.9, 1.1, 1.3, 0.4),
        ),
        (
            "truncated Normal",
            (a, b, x) -> logpdf(truncated(Normal(), a, b), x),
            (-0.3, 0.3, 0.1),
        ),
        (
            "truncated Uniform",
            (a, b, α, β, x) -> logpdf(truncated(Uniform(α, β), a, b), x),
            (0.1, 0.9, -0.1, 1.1, 0.4),
        ),
        ("Dirichlet", (a, x) -> logpdf(Dirichlet(a), [x, 1-x]), ([1.5, 1.1], 0.6)),
        (
            "reshape",
            x -> logpdf(reshape(product_distribution([Normal(), Uniform()]), 1, 2), x),
            ([2.1 0.7],),
        ),
        (
            "vec",
            x -> logpdf(vec(LKJ(2, 1.1)), x),
            ([1.0, 0.489, 0.489, 1.0],),
        ),
    ]
        @info "$name"
        rng = StableRNG(123456)
        test_taped_rrule!!(rng, f, deepcopy(x)...; interface_only=false, perf_flag=:none)
    end
end
