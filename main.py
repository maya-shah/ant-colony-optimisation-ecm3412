from antcolony import AntColony

if __name__ == "__main__":
    # Parameters
    dataset = 'datasets/brazil58.xml'
    alpha = 1.0
    beta = 5.0
    n_ants = 100
    evaporation_rate = 0.7
    Q = 1
    iterations = 100

    ant_colony = AntColony(dataset=dataset, alpha=alpha, beta=beta, n_ants=n_ants, evaporation_rate=evaporation_rate,
                           Q=Q, iterations=iterations)

    plot_y, best_path = ant_colony.main()

    # print(*plot_y, sep='\n')

    print(f"The best path cost found is: {best_path}")
