using CairoMakie


overfit_train(x) = log(x) - 0.02*x
overfit_test(x) = 3/log(x) + 10 - 0.02*x

ideal_train(x) = log(0.3x) - 0.02x - 11.8
ideal_test(x) = 3/log(x) - 10 - 0.01*x

underfit_train(x) = -(log(x) + 12/0.3x * 10) + 10 #0.01*x
underfit_test(x) = 9/log(x) + 10 - 0.1*x


function main()

    domain = 1:50

    funcs = [
        (underfit_train, underfit_test),
        (ideal_train, ideal_test),
        (overfit_train, overfit_test)
    ]

    for (func_train, func_test) in funcs

        fig = Figure()
        axs = Axis(fig[1, 1])

        # train loss
        train_loss = lines!(axs, func_train.(domain),
                            linewidth = 5,
                            color = :blue,
                            label = "Erro no Treinamento")

        # test loss
        test_loss  = lines!(axs, func_test.(domain),
                            linewidth = 5,
                            color = :red,
                            label = "Erro no Teste")


        func_train === underfit_train ? position = :rb : position = :rt

        axislegend(axs, position = position)

        hideydecorations!(axs, ticks = false, grid = false)

        axs.xlabel = "Número de Iterações no Treinamento"
        axs.ylabel = "Erro"
        axs.xticks = 0:5:50

        prefix = split(string(func_test), '_')[begin]

        save("./img/$(prefix)_learning_curves.pdf", fig)
    end
end

main()
