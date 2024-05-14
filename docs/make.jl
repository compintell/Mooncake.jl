using Documenter, Tapir

makedocs(
    sitename="Tapir.jl",
    format=Documenter.HTML(),
    modules=[Tapir],
    checkdocs=:none,
)

deploydocs(repo="github.com/withbayes/Tapir.jl.git", push_preview=true)
