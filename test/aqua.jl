using Aqua

@testset "Aqua.jl" begin
    # Ambiguities needs to be tested seperately until the bug in Aqua package (https://github.com/JuliaTesting/Aqua.jl/issues/77) is fixed
    Aqua.test_ambiguities([EnergySamplers]; recursive=false, broken=false)

    Aqua.test_all(EnergySamplers; ambiguities=false)
end
