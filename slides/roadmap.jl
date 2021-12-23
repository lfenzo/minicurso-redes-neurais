using CairoMakie

teoria(x) = 4 - x
pratica(x) = x

function main()

    domain = range(1, 3, step = 0.01)

    fig = Figure()

    axs = Axis(fig[1, 1])

    lines!(axs, domain, pratica.(domain),
           label = "Prática",
           color = :blue,
           linewidth = 3,
           linestyle = :dash)

    lines!(axs, domain, teoria.(domain),
           label = "Teoria",
           color = :orange,
           linewidth = 3)

    axs.xlabel = "Dia do Minicurso"
    axs.ylabel = "Quantidade Relativa"

    axislegend(axs, position = :rc)

    save("./img/teoria-pratica.pdf", fig)
end


main()
