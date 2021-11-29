using CairoMakie

sigmoid(z) = 1 / (1 + exp(-z))
relu(z) = max(0, z) 
step(z) = z > 0 ? 1 : 0
leaky(z, a) = z > 0 ? z : a*z
identity(z) = z

function main()

    for func in [sigmoid, relu, tanh, step, leaky, identity]

        if func === leaky
            func = (x) -> leaky(x, 0.3)
        end

        fig = Figure()

        axs = Axis(fig[1, 1])

        domain = -5:0.1:5

        lines!(axs, domain, func.(domain),
               linewidth = 7,
               ylabelsize = 12)

        axs.ylabel = "Ativação g(z)"
        axs.ylabelsize = 20

        axs.xlabel = "z"
        axs.xlabelsize = 20

        save(joinpath("img", "activation_$func.pdf"), fig)
    end
end

main()
