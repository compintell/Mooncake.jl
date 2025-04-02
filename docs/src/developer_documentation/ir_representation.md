# IR Representation

Mooncake.jl operates by transforming Julia's SSA-form (static single assignment) Intermediate Representation (IR).
A good understanding of this IR is needed to understand Mooncake, so we go through the essentials here, prior to discussing how Mooncake represents this IR.

## Julia's SSA-form IR

### Straight-Line Code

You can find the IR associated to a given call signature using `Base.code_ircode_by_type`:
```jldoctest
julia> function foo(x)
           y = sin(x)
           z = cos(y)
           return z
       end
foo (generic function with 1 method)

julia> signature = Tuple{typeof(foo), Float64}
Tuple{typeof(foo), Float64}

julia> Base.code_ircode_by_type(signature)[1][1]
2 1 ─ %1 = invoke sin(_2::Float64)::Float64
3 │   %2 = invoke cos(%1::Float64)::Float64
4 └──      return %2
```

What you can see here is that the calls to `sin` and `cos` in the original function are associated to a number, denoted `%1` and `%2`. We refers to these as the "ssa"s associated to each statement.
Each statement is associated to a single ssa, and this association is simply determined by where it appears in the list of statements.
You will also notice that the argument `x` has been replaced with a `_2` in the first statement -- in general, all uses of the `n`th argument are indicated by `_n` (the first argument is the function itself).
The final statement requires no explanation.

Note that this IR is obtained after both type inference and various Julia-level optimisation passes.
This means that the type information is available for each statement.
For example, the `::Float64` at the end of the first and second statements indicates that the type of `%1` and `%2` is going to be `Float64`.
The types are also displayed at uses -- the call to `sin` involves `_2::Float64`, not just `_2`.

Additionally notice that the statements are `invoke` statements, rather than just call statments.
In Julia's IR, an `invoke` statement represents static dispatch to a particular `MethodInstance` -- i.e. running type inference + optimisation passes has determined enough about the argument types to make it possible to know exactly which `MethodInstance` of `sin` and `cos` to call.
This is very common occurence in type-stable code.

### Control Flow

The above is straight-line code -- it does not involve any control flow.
Julia has several statments which are involved in handling control flow.
For example
```jldoctest
julia> function bar(x)
           if x > 0
               return x
           else
               return 5x
           end
       end
bar (generic function with 1 method)

julia> Base.code_ircode_by_type(Tuple{typeof(bar),Float64})[1][1]
2 1 ─ %1 = Base.lt_float(0.0, _2)::Bool
  │   %2 = Base.or_int(%1, false)::Bool
  └──      goto #3 if not %2
3 2 ─      return _2
5 3 ─ %5 = Base.mul_float(5.0, _2)::Float64
  └──      return %5
```
In this example we see the statement `goto #3 if not %2`.
This should be read as "jump to basic block 3 if %2 is equal to `false`".
The second half of that statement should be clear, but to understand the first half requires knowing what a basic block is:
```julia
  1 ─
  │  
  └──
  2 ─
  3 ─
  └──
```
Here I've removed everything from the above example except for information about the basic block structure.
Each basic block is a sequence of statements which _must_ always execute one after the other, before doing something else.
In this example, we have three basic blocks -- you can see this from the numbers `1`, `2`, and `3`.
The first basic block comprises three statements, the second only one statement, and the third two statements.
Another way to investigate this structure is to look at the control-flow graph associated to the IR:
```jldoctest
julia> Base.code_ircode_by_type(Tuple{typeof(bar),Float64})[1][1].cfg
CFG with 3 blocks:
  bb 1 (stmts 1:3) → bb 3, 2
  bb 2 (stmt 4)
  bb 3 (stmts 5:6)
```
