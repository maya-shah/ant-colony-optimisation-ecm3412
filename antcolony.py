import random

import numpy as np
from bs4 import BeautifulSoup


class AntColony:
    """
    Ant Colony Class
    """
    class Ant:
        """
        Ant Class
        """
        def __init__(self, distances, start_location, pheromone_graph, alpha, beta, first_move):
            """
            Initialize Ant Class.

            :param distances: [n x n] matrix representing the distances between each city
            :param start_location: the starting location for the ant
            :param pheromone_graph: [n x n] matrix representing the pheromone graph initialized with random values
            :param alpha: the pheromone importance factor
            :param beta: probability for optimal paths
            :param first_move: true or false value if this is the first time the ant is passing through the graph
            """
            self.distances = distances
            self.start_location = start_location
            self.path = []
            self.distance_travelled = 0.0
            self.current_location = start_location
            self.pheromone_graph = pheromone_graph
            self.possible_next_locations = [x for x in range(0, len(self.distances))]
            self.visited_cities = []
            self.alpha = alpha
            self.beta = beta
            self.first_move = first_move

            self.tour = False

        def main(self) -> None:
            while self.possible_next_locations:
                next_location = self._pick_next_location()
                self._move(self.current_location, next_location)

                if not self.possible_next_locations:
                    next_location = self.path[0]
                    self._move(self.current_location, next_location)
                    break

            self.tour = True

        def get_path(self) -> list:
            if self.tour:
                return self.path

        def _pick_next_location(self):
            probability = {}
            sum = 0.0

            if self.first_move:
                possible_location = self.possible_next_locations[random.randint(0, len(self.distances) - 1)]
                return possible_location

            for possible_location in self.possible_next_locations:
                pheromone_amount = float(self.pheromone_graph[self.current_location][possible_location])
                distance = float(self.distances[self.current_location][possible_location])

                probability[possible_location] = pow(pheromone_amount, self.alpha) * pow(1 / distance, self.beta)
                sum += probability[possible_location]

            cumulative_probability = 0.0

            rand_num = random.random()

            for possible_location in probability:
                if rand_num <= cumulative_probability + (probability[possible_location] / sum):
                    self.visited_cities.append(possible_location)
                    return possible_location
                cumulative_probability += probability[possible_location] / sum

        def _move(self, start, end):
            # update path
            self.path.append(end)
            if self.possible_next_locations:
                # remove from possible locations
                self.possible_next_locations.remove(end)
            # update distance travelled
            self.distance_travelled += self.distances[start][end]
            # update current location
            self.current_location = end

            if self.first_move:
                self.first_move = False

    def __init__(self, alpha: float = 1.0, beta: float = 2.0, dataset: str = 'datasets/burma14.xml', n_ants: int = 50,
                 evaporation_rate: float = 0.5, Q: float = 1.0, iterations: int = 100, first_move: bool = True):
        """
        Initialize Ant Colony Class.

        :param alpha: the pheromone importance factor
        :param beta: probability for optimal paths
        :param first_move: true or false value if this is the first time the ant is passing through the graph
        :param dataset: a string value for which dataset to use
        :param n_ants: the number of ants in the colony
        :param evaporation_rate: the pheromone evaporation rate
        :param Q: heuristic information
        :param iterations: the number of iterations to run
        """
        self.alpha = alpha
        self.beta = beta
        self.dataset = dataset
        self.distances = self._construct_distance_graph()
        self.n_ants = n_ants
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.n_cities = len(self.distances)
        self.pheromone_graph = self._init_pheromone_graph()
        self.Q = Q
        self.ant_pheromone_graph = self._init_ant_pheromone_graph()
        self.first_move = first_move
        self.ants = self._init_ants()

    def _construct_distance_graph(self) -> list[list[float]]:
        with open(self.dataset) as f:
            data = f.read()

        bs_data = BeautifulSoup(data, "xml")

        distances, count = [], 0

        for vertex in bs_data.find_all("vertex"):
            distances.append([float(edge['cost']) for edge in vertex.find_all("edge")])
            distances[count].insert(count, 0)
            count += 1

        return distances

    def _init_pheromone_graph(self):
        pheromone_graph = [[np.round(random.random(), 4) for _ in range(self.n_cities - 1)] for _ in
                           range(self.n_cities)]

        for x in range(len(pheromone_graph)):
            pheromone_graph[x].insert(x, 0.0)

        return pheromone_graph

    def _init_ant_pheromone_graph(self):
        return [[0.0 for _ in range(self.n_cities)] for _ in range(self.n_cities)]

    def _init_ants(self):
        if self.first_move:
            ants = []
            for _ in range(self.n_ants):
                random_city = random.randint(0, len(self.distances) - 1)
                ants.append(
                    self.Ant(self.distances, random_city, self.pheromone_graph, self.alpha, self.beta, self.first_move))
            return ants

        for ant in self.ants:
            random_city = random.randint(0, len(self.distances) - 1)
            ant.__init__(self.distances, random_city, self.pheromone_graph, self.alpha, self.beta, self.first_move)
        return self.ants

    def _update_pheromone_graph(self):
        for i in range(len(self.pheromone_graph)):
            for j in range(len(self.pheromone_graph)):
                self.pheromone_graph[i][j] = (1 - self.evaporation_rate) * self.pheromone_graph[i][j]
                self.pheromone_graph[i][j] += self.ant_pheromone_graph[i][j]

    def _deposit_pheromones(self, ant):
        path = ant.get_path()
        for i in range(len(path) - 1):
            # Q / L is the amount of pheromone to deposit
            self.ant_pheromone_graph[path[i]][path[i + 1]] += float(self.Q / ant.distance_travelled)

    def main(self):
        shortest_path = [np.inf]
        evals_count = 0

        plot_y = []

        for i in range(self.iterations):
            for ant in self.ants:
                if evals_count == 10000:
                    break

                ant.main()

                self._deposit_pheromones(ant)

                if ant.distance_travelled < shortest_path[0]:
                    shortest_path.pop(0)
                    shortest_path.append(ant.distance_travelled)
                    plot_y.append(shortest_path[0])

                evals_count += 1

            self._update_pheromone_graph()
            self._init_ants()
            self._init_ant_pheromone_graph()

        return plot_y, shortest_path[0]
