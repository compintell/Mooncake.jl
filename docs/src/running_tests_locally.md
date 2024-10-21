# Running Tests Locally

Mooncake.jl's test suite is fairly extensive. While you can, of course, run the tests in the same way that you would any Julia package, this is not usually optimal. When editing some code, you typically only want to run the tests associated with it, not the entire test suite.

Mooncake's tests are organised as follows:
1. Things that are required for most / all test suites are loaded up in `test/front_matter.jl`.
1. The tests for something in `src` are located in an identically-named file in `test`. e.g. the unit tests for `src/rrules/new.jl` are located in `test/rrules/new.jl`.

Thus, a workflow that I (Will) find works very well is the following:
1. Ensure that you have Revise.jl and TestEnv.jl installed in your default environment.
1. `dev` Mooncake.jl, start the REPL, and navigate to the top level of the Mooncake.jl directory.
1. `using TestEnv, Revise`. Better still, load both of these in your `.julia/config/startup.jl` file so that you don't ever forget to load them.
1. Run the following: `using Pkg; Pkg.activate("."); TestEnv.activate(); include("test/front_matter.jl");` to set up your environment.
1. `include` whichever test file you want to run the tests from.
1. Modify code, and re-`include` tests to check it has done was you need. Loop this until done.
1. Make a PR. This runs the entire test suite -- I find that I almost _never_ run the entire test suite locally.

The purpose of this approach is to:
1. Avoid restarting the REPL each time you make a change, and
2. Run the smallest bit of the test suite possible when making changes, in order to make development a fast and enjoyable process.

If you find that this strategy leaves you running more of the test suite than you would like, consider copy + pasting specific tests into the REPL, or commenting out a chunk of tests in the file that you are editing during development (try not to commit this).
I find this is rather crude strategy effective in practice.
