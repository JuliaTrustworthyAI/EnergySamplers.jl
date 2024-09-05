using Distributions
using Flux
using MLJBase
using Optimisers
using EnergySamplers:
    ImproperSGLD,
    SGLD,
    JointSampler,
    ConditionalSampler,
    UnconditionalSampler,
    AbstractSampler,
    PMC

@testset "Samplers" begin
    f(x) = @.(2x + 1)  # dummy model
    nn = Chain(Dense(2, 1, σ))
    rule = ImproperSGLD()

    # Data:
    nobs = 2000
    X, y = make_circles(nobs; noise=0.1, factor=0.5)
    Xmat = Float32.(permutedims(matrix(X)))
    X = table(permutedims(Xmat))
    batch_size = Int(round(nobs / 10))

    # Distributions:
    𝒟x = Normal()
    𝒟y = Categorical(ones(2) ./ 2)

    all_samplers = Dict(
        "Conditional" => ConditionalSampler,
        "Unconditional" => UnconditionalSampler,
        "Joint" => JointSampler,
    )

    for (name, Sampler) in all_samplers
        @testset "$name" begin
            smpler = Sampler(
                𝒟x, 𝒟y; input_size=size(Xmat)[1:(end - 1)], batch_size=batch_size
            )
            @test smpler isa AbstractSampler

            X̂ = smpler(f, rule; n_samples=10)
            @test size(X̂, 2) == 10

            X̂ = smpler(nn, rule; n_samples=10)
            @test size(X̂, 2) == 10
        end
    end

    @testset "Persistent Markov Chains (PMC)" begin

        # Train a simple neural network on the data (classification)
        Xtrain = permutedims(MLJBase.matrix(X))
        ytrain = Flux.onehotbatch(y, levels(y))
        train_set = zip(eachcol(Xtrain), eachcol(ytrain))
        inputdim = size(first(train_set)[1], 1)
        outputdim = size(first(train_set)[2], 1)
        nn = Chain(Dense(inputdim, 32, relu), Dense(32, 32, relu), Dense(32, outputdim))
        loss(yhat, y) = Flux.logitcrossentropy(yhat, y)
        opt_state = Flux.setup(Flux.Adam(), nn)
        epochs = 5
        for epoch in 1:epochs
            Flux.train!(nn, train_set, opt_state) do m, x, y
                loss(m(x), y)
            end
            @info "Epoch $epoch"
            println("Accuracy: ", mean(Flux.onecold(nn(Xtrain)) .== Flux.onecold(ytrain)))
        end

        # PMC
        bs = 10
        ntrans = 100
        niter = 20
        # Conditionally sample from first class:
        smpler = ConditionalSampler(
            𝒟x, 𝒟y; input_size=size(Xmat)[1:(end - 1)], batch_size=bs
        )
        x1 = PMC(smpler, nn, ImproperSGLD(); ntransitions=ntrans, niter=niter, y=1)
        # Conditionally sample from second class:
        smpler = ConditionalSampler(
            𝒟x, 𝒟y; input_size=size(Xmat)[1:(end - 1)], batch_size=bs
        )
        x2 = PMC(smpler, nn, ImproperSGLD(); ntransitions=ntrans, niter=niter, y=2)

        # using Plots
        # plt = scatter(Xtrain[1, :], Xtrain[2, :], color=Int.(y.refs), group=Int.(y.refs), label=["X|y=0" "X|y=1"], alpha=0.1)
        # scatter!(x1[1, :], x1[2, :], color=1, label="X̂|y=0")
        # scatter!(x2[1, :], x2[2, :], color=2, label="X̂|y=1")
        # plot(plt, xlims=(-2, 2), ylims=(-2, 2))

    end

    @testset "Optimizers (Bayesian Inference Example)" begin
        # Example from taken from https://sebastiancallh.github.io/post/langevin/
        using Flux
        using Flux: gpu
        using MLDataUtils: shuffleobs, stratifiedobs, rescale!
        using RDatasets, Statistics
        using Plots, Random
        Random.seed!(0)

        data = dataset("ISLR", "Default")
        todigit(x) = x == "Yes" ? 1.0 : 0.0
        data[!, :Default] = map(todigit, data[:, :Default])
        data[!, :Student] = map(todigit, data[:, :Student])

        target = :Default
        numerics = [:Balance, :Income]
        features = [:Student, :Balance, :Income]
        train, test = (d -> stratifiedobs(first, d; p=0.7))(shuffleobs(data))

        for feature in numerics
            μ, σ = rescale!(train[!, feature]; obsdim=1)
            rescale!(test[!, feature], μ, σ; obsdim=1)
        end

        prep_X(x) = gpu(Matrix(x)')
        prep_y(y) = gpu(reshape(y, 1, :))
        train_X, test_X = prep_X.((train[:, features], test[:, features]))
        train_y, test_y = prep_y.((train[:, target], test[:, target]))
        train_set = Flux.DataLoader((train_X, train_y); batchsize=100, shuffle=false)

        function train_logreg(; steps::Int=1000, opt=Flux.Descent(2))
            Random.seed!(1)

            paramvec(θ) = reduce(hcat, cpu(θ))
            model = gpu(Dense(length(features), 1, sigmoid))
            θ = Flux.params(model)
            θ₀ = paramvec(θ)

            predict(x; thres=0.5) = model(x) .> thres
            accuracy(x, y) = mean(cpu(predict(x)) .== cpu(y))

            loss(yhat, y) = Flux.binarycrossentropy(yhat, y)
            avg_loss(yhat, y) = mean(loss(yhat, y))
            trainloss() = avg_loss(model(train_X), train_y)
            testloss() = avg_loss(model(test_X), test_y)

            trainlosses = [cpu(trainloss()); zeros(steps)]
            testlosses = [cpu(testloss()); zeros(steps)]
            weights = [cpu(θ₀); zeros(steps, length(θ₀))]

            opt_state = Flux.setup(opt, model)

            for t in 1:steps
                for data in train_set
                    input, label = data

                    # Calculate the gradient of the objective
                    # with respect to the parameters within the model:
                    grads = Flux.gradient(model) do m
                        result = m(input)
                        loss(result, label)
                    end

                    Flux.update!(opt_state, model, grads[1])
                end

                # Bookkeeping
                weights[t + 1, :] = cpu(paramvec(θ))
                trainlosses[t + 1] = cpu(trainloss())
                testlosses[t + 1] = cpu(testloss())
            end

            println("Final parameters are $(paramvec(θ))")
            println("Test accuracy is $(accuracy(test_X, test_y))")

            return model, weights, trainlosses, testlosses
        end

        results = train_logreg(; steps=1000, opt=SGLD(10.0, 1000.0, 0.9))
        model, weights, trainlosses, testlosses = results
        plot(weights; label=["Student" "Balance" "Income" "Intercept"])

        results = train_logreg(; steps=100, opt=ImproperSGLD(2.0, 0.01))
        model, weights, trainlosses, testlosses = results
        plot(weights; label=["Student" "Balance" "Income" "Intercept"])

        @test true
    end
end
