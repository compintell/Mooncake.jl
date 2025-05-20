This represents an attempt at providing a demo on how to define customized tangent types for a recursive type to work with Mooncake.jl.

The source code of Mooncake.jl is located at the parent parent directory of this file.

Documentation file `known_limitations.md` has a minimal description of the problem.

`tangents.jl` contains the interface functions.

You should probably first read some other documentation, in `{workspace}/docs/src` directory to understand the design of the package.

file `recur_demo.jl` contains the source code for the customized tangent types and interface functions. A minimal test is at the end of the file.

You should be able to run the source code by
* make sure you are in the `recursive_type_demo` directory
  * run `julia --project=./ -e 'include("recur_demo.jl")'`

For the rest of the tests, this is a standard Julia package, so you can run it in the usual way.
