struct ReverseDiffTracerContext <: TapedContext end

const RDTC = ReverseDiffTracerContext

for (M, f, arity) in DiffRules.diffrules(; filter_modules=[:Base])
    pb_name = Symbol(string(gensym(f)) * "_pullback")
    if arity == 1
        @eval isprimitive(::RDTC, ::typeof($M.$(f)), ::Float64) = true
    elseif arity == 2
        @eval isprimitive(::RDTC, ::typeof($M.$(f)), ::Float64, ::Float64) = true
    else
        throw(error("Expected arity = 1 or 2, got $arity"))
    end
end

