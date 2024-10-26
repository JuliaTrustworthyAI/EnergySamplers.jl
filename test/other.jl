using EnergySamplers: energy_differential, energy_penalty

@testset "Other things" begin
    @testset "Energy differential" begin
        f(X::Matrix) = [prod(x) for x in eachcol(X)]
        x1 = randn(10, 1)
        x2 = randn(10, 1)
        @test isreal(energy_differential(f, x1, x2, 1))
        @test isreal(energy_penalty(f, x1, x2, 1))
    end
end
