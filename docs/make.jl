using Documenter, Tapir

makedocs(
    sitename = "Tapir.jl",
    format = Documenter.HTML(),
    modules = [Tapir],
    checkdocs = :none,
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/withbayes/Tapir.jl.git", push_preview=true)
