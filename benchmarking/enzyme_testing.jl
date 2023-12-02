using Enzyme

struct Positive
    x::Float64
    function Positive(x)
        @assert x > 0
        return new(x)
    end
end

function positive_test(x)
    y = Positive(x)
    return y.x * 2
end

negate_positive_field(x::Positive) = -x.x

struct MixedFields
    x::Float64
    y::Vector{Float64}
end

mixed_fields_sum(x) = x.x + sum(x.y)

sum_via_mixed_fields(x, y) = mixed_fields_sum(MixedFields(x, y))

struct MixedFieldsReal
    x::Real
    y::Vector{Float64}
end

# Example taken from the Julia manual
mutable struct SelfReferential2
    obj::SelfReferential2
    y::Float64
    function SelfReferential2(y::Float64)
        x = new()
        x.obj = x
        x.y=y
        return x
    end
end

use_self_referential(x) = sin(SelfReferential2(x).y)


struct Incomplete
    a::Float64
    b::Vector{Float64}
    Incomplete(a::Float64) = new(a)
end

# This works fine.
function use_incomplete(x)
    t = Incomplete(x)
    return sin(t.a)
end

function main()
    # This is fine
    autodiff(Reverse, positive_test, Active, Active(1.0))

    # This is clearly a hack, albeit, one that will always work fine.
    autodiff(Reverse, negate_positive_field, Active, Active(Positive(1.0)))

    # How do I handle a "MixedFields" type?
    # autodiff(Reverse, mixed_fields_sum, Active, Active(MixedFields(5.0, randn(4))))
    d = Duplicated(MixedFields(5.0, randn(4)), MixedFields(0.0, zeros(4)))
    display(d)
    println()
    autodiff(Reverse, mixed_fields_sum, Active, d)
    display(d)
    println()
    # Unclear how to get the above to work correctly.

    # # Does Enzyme do the correct thing when we go via an interesting struct?
    # y = Duplicated(randn(4), zeros(4))
    # display(autodiff(Reverse, sum_via_mixed_fields, Active, Active(1.0), y))
    # println()
    # display(y)
    # println()

    d = Duplicated(MixedFieldsReal(5.0, randn(4)), MixedFieldsReal(0.0, zeros(4)))
    display(d)
    println()
    autodiff(Reverse, mixed_fields_sum, Active, d)
    display(d)
    println()

    println("self referential")
    display(autodiff(Reverse, use_self_referential, Active, Active(5.0)))
    println()

    autodiff(Reverse, use_incomplete, Active, Active(5.0))
end
