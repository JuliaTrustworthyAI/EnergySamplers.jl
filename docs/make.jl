#!/usr/bin/env julia
#
#

if "--help" ∈ ARGS
    println(
        """
docs/make.jl

Render the documentation using Quarto with optional arguments

Arguments
* `--help`              - print this help and exit without rendering the documentation
* `--prettyurls`        – toggle the prettyurls part to true (which is otherwise only true on CI)
* `--quarto`            – run the Quarto notebooks from the `tutorials/` folder before generating the documentation
  this has to be run locally at least once for the `tutorials/*.md` files to exist that are included in
  the documentation (see `--exclude-tutorials`) for the alternative.
  If they are generated once they are cached accordingly.
  Then you can spare time in the rendering by not passing this argument.
  If quarto is not run, some tutorials are generated as empty files, since they
  are referenced from within the documentation.
""",
    )
    exit(0)
end

# (a) Did someone say render?
if "--quarto" ∈ ARGS
    @info "Rendering Quarto"
    run(`quarto render $(@__DIR__)`)
end

using EnergySamplers
using Documenter

DocMeta.setdocmeta!(EnergySamplers, :DocTestSetup, :(using EnergySamplers); recursive=true)

makedocs(;
    modules=[EnergySamplers],
    authors="Patrick Altmeyer and contributors",
    sitename="EnergySamplers.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaTrustworthyAI.github.io/EnergySamplers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaTrustworthyAI/EnergySamplers.jl",
    devbranch="main",
)
