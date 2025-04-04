# IR Representation

Mooncake.jl operates by transforming Julia's SSA-form (static single assignment) Intermediate Representation (IR), albeit we use a different data structure to represent this IR than is usually found in the compiler.
A good understanding of Julia's IR is therefore needed to understand Mooncake, so we go through the essentials here, prior to discussing how Mooncake represents this IR (and constrasting this with how the compiler represents this IR).

Please note that Julia's SSA-form IR typically changes representation slightly between minor versions of Julia, as it's not part of the public interface of the language.
The information below is accurate on version 1.11.4, but you might well find that things are slightly different on different versions.

## Julia's SSA-form IR

### Straight-Line Code

You can find the IR associated to a given signature using `Base.code_ircode_by_type`:
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
```jldoctest bar
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
Here, everything is removed from the above example except for information about the basic block structure.
To first approximation, each basic block is a sequence of statements which _must_ always execute one after the other, before doing something else.
In this example, we have three basic blocks -- you can see this from the numbers `1`, `2`, and `3`.
The first basic block comprises three statements, the second only one statement, and the third two statements.
Another way to investigate this structure is to look at the control-flow graph associated to the IR:
```jldoctest bar
julia> Base.code_ircode_by_type(Tuple{typeof(bar),Float64})[1][1].cfg
CFG with 3 blocks:
  bb 1 (stmts 1:3) → bb 3, 2
  bb 2 (stmt 4)
  bb 3 (stmts 5:6)
```
For example, the above states that "bb" (basic block) 1 comprises statement 1 to 3 of the IR, and has successor blocks 2 and 3.
Blocks 2 and 3 have no successors, because they both end in a `return` statement.

Additionally, note that `Base.lt_float` (used to check if one floating point number is less than another) and `Base.or_int` do not appear as `invoke` statements -- this is because they are not generic Julia functions.
Rather, they are Julia intrinsics:
```jldoctest
julia> Base.lt_float
lt_float (intrinsic function #31)
```
These intrinsics have special handling in the compiler.
Either way, the overall point is to be aware that these kinds of low-level intrinsics exist, and appear regularly in Julia IR.

### Simple Loops and Phi-Nodes

Finally, we shall consider a simple loop:
```jldoctest my_factorial
julia> function my_factorial(x::Int)
           n = 0
           s = 1
           while n < x
               n += 1
               s *= n
           end
           return s
       end
my_factorial (generic function with 1 method)

julia> Base.code_ircode_by_type(Tuple{typeof(my_factorial), Int})[1][1]
  1 ─      nothing::Nothing
4 2 ┄ %2 = φ (#1 => 1, #3 => %7)::Int64
  │   %3 = φ (#1 => 0, #3 => %6)::Int64
  │   %4 = Base.slt_int(%3, _2)::Bool
  └──      goto #4 if not %4
5 3 ─ %6 = Base.add_int(%3, 1)::Int64
6 │   %7 = Base.mul_int(%2, %6)::Int64
7 └──      goto #2
8 4 ─      return %2
```
There are a few new intrinsics that we have not seen previously (`Base.slt_int` (used to check whether one int is strictly less than another), `Base.add_int`, and `Base.mul_int`).
Additionally, there is the node `goto #2`, which simply states that control flow should jump to basic block 2 whenever it is hit.

The most interesting additional nodes, however, are the two `φ` nodes.
These are a defining feature of SSA-form IR. Consider the first `φ` node:
```julia
%2 = φ (#1 => 1, #3 => %7)
```
means ssa `%2` takes value `1` if the previous basic block was `#1`, and whatever value is currently associated to ssa `%7` if the previous basic block was `#3`.
It is helpful to step through this code in your head: upon calling `my_factorial` we enter basic block `#1`, and proceed directly to basic block `#2`.
Therefore, on the first iteration, `%2` takes value `1`. We never return to basic block `#1`, so all subsequent iterations must take value `%7`.
You should convince yourself that `%2` corresponds to the value of `s` at each iteration, and `%3` corresponds to the value of `n` at each iteration.


### Summary

Julia's SSA-form IR comprises a sequence of statements, which can be broken down into a collection of basic blocks.
Each basic block begins with a (potentially empty) collection of phi nodes, followed by a seqeuence of statements, and potentially finished by a _terminator_ (goto, goto-if-not, return).
Control flow is dictated by the terminators at the end of basic blocks -- if there is no terminator then we "fall through" to the next statement.

## Julia Compiler IR Representation

The Julia compiler represents the IR associated to a signature via an object called `Core.Compiler.IRCode`.
The statements are given by the `stmts` field, which is a `Core.Compiler.InstructionStream`.
An `InstructionStream` is a collection of 5 `Vector`s, each of which have the same length.
The properties of the `n`th statement in the `IR` are given by the `n`th element of each of these vectors.
For example, the `stmt` field contains the statement itself, the `type` field contains the inferred type associated to the statement.
We'll skip the rest for now.
For example, the statements associated to the `my_factorial` function above can be retrieved as follows:
```jldoctest my_factorial
julia> Base.code_ircode_by_type(Tuple{typeof(my_factorial), Int})[1][1].stmts.stmt
9-element Vector{Any}:
 nothing
 :(φ (%1 => 1, %3 => %7))
 :(φ (%1 => 0, %3 => %6))
 :(Base.slt_int(%3, _2))
 :(goto %4 if not %4)
 :(Base.add_int(%3, 1))
 :(Base.mul_int(%2, %6))
 :(goto %2)
 :(return %2)
```
The types can be accessed in a similar way:
```jldoctest my_factorial
julia> Base.code_ircode_by_type(Tuple{typeof(my_factorial), Int})[1][1].stmts.type
9-element Vector{Any}:
 Nothing
 Int64
 Int64
 Bool
 Any
 Int64
 Int64
 Any
 Any
```

As seen in [Control Flow](@ref), the control flow graph (CFG) is represented as a separate data structure, stored in the `cfg` field of the `IRCode`.
The argument types associated to the signature are stored in the `argtypes` field of the `IRCode`.




