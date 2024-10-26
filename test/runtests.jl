using EnergySamplers
using Test

@testset "EnergySamplers.jl" begin
    include("aqua.jl")
    include("samplers.jl")
    include("other.jl")
end
