for (M, f, arity) in DiffRules.diffrules(; filter_modules=nothing)
    if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f))
        # @warn "$M.$f is not available and hence rule for it can not be defined"
        continue  # Skip rules for methods not defined in the current scope
    end
    (f == :rem2pi || f == :ldexp) && continue # not designed for Float64s
    if arity == 1
        dx = DiffRules.diffrule(M, f, :x)
        @eval begin
            Umlaut.isprimitive(::RMC, ::typeof($M.$f), ::Float64) = true
            function rrule!!(::CoDual{typeof($M.$f)}, x::CoDual{Float64})
                x = primal(x)
                pb!!(ȳ, f̄, x̄) = f̄, x̄ + ȳ * $dx
                return CoDual(($M.$f)(x), zero(Float64)), pb!!
            end
        end
    elseif arity == 2
        da, db = DiffRules.diffrule(M, f, :a, :b)
        @eval begin
            Umlaut.isprimitive(::RMC, ::typeof($M.$f), ::Float64, ::Float64) = true
            function rrule!!(::CoDual{typeof($M.$f)}, a::CoDual{Float64}, b::CoDual{Float64})
                a = primal(a)
                b = primal(b)
                pb!!(ȳ, f̄, ā, b̄) = f̄, ā + ȳ * $da, b̄ + ȳ * $db
                return CoDual(($M.$f)(a, b), zero(Float64)), pb!!
            end
        end
    end
end
