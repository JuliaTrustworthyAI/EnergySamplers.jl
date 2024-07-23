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
* `--project`           - specify the project name (default: `$(@__DIR__)`)
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

# (a) Specify project
using Pkg
if any(contains.(ARGS, "--project"))
    @assert sum(contains.(ARGS, "--project")) == 1 "Only one environment can be specified using the `--project` argument."
    _path =
        ARGS[findall(contains.(ARGS, "--project"))][1] |>
        x -> replace(x, "--project=" => "")
    Pkg.activate(_path)
else
    Pkg.activate(@__DIR__)
end

# (b) Did someone say render?
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
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
