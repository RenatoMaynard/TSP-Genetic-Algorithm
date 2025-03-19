import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, Markdown
import numpy as np
import tsplib95
import random
import time
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
plt.rcParams['animation.embed_limit'] = 2**128

# Global variables
INF = 9999999
dist_matrix = []
use_numpy = 0  # 0 = do not use numpy arrays for individuals, 1 = use numpy arrays

class AnimationTSP:
    """
    This class creates an animated visualization of a TSP solution improving over time.
    """

    def __init__(self, history, x_coords, y_coords, costs):
        """
        :param history: A list of solutions (each solution is a list of city indices).
        :param x_coords: List of x-coordinates of each city.
        :param y_coords: List of y-coordinates of each city.
        :param costs: A list of costs corresponding to each solution in history.
        """

        # Ensure 'history' is a list of lists
        if isinstance(history[0], list):
            self.history = history
        else:
            self.history = [h.tolist() for h in history]

        self.costs = costs
        self.points = np.column_stack((x_coords, y_coords))

        self.fig, self.ax = plt.subplots()
        self.line, = plt.plot([], [], lw=2)
        self.title = self.ax.text(
            0.8, 1.035, "", 
            bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
            transform=self.ax.transAxes,
            ha="center"
        )

    def init_animation(self):
        """
        Initializes the animation with empty lines and plots the city nodes.
        """
        # Plot the city nodes from the first solution in the history
        x_plot = [self.points[i][0] for i in self.history[0]]
        y_plot = [self.points[i][1] for i in self.history[0]]
        plt.plot(x_plot, y_plot, 'co')

        # Adjust axes with a small margin
        extra_x = (max(x_plot) - min(x_plot)) * 0.05
        extra_y = (max(y_plot) - min(y_plot)) * 0.05
        self.ax.set_xlim(min(x_plot) - extra_x, max(x_plot) + extra_x)
        self.ax.set_ylim(min(y_plot) - extra_y, max(y_plot) + extra_y)

        # Initialize the route line as empty
        self.line.set_data([], [])
        return self.line,

    def update_animation(self, frame):
        """
        For each frame, update the plot with the route of that generation/iteration.
        """
        route = self.history[frame]
        x_plot = [self.points[i, 0] for i in route + [route[0]]]
        y_plot = [self.points[i, 1] for i in route + [route[0]]]

        self.title.set_text(f"Iteration {frame}, Cost {self.costs[frame]}")
        self.line.set_data(x_plot, y_plot)
        return self.line

    def animate_routes(self):
        """
        Creates and displays the animation in a Jupyter environment.
        """
        # Setting how many frames to skip to create a shorter animation
        div = len(self.history) // 3 if len(self.history) > 3 else 1
        step = len(self.history) // div if div != 0 else 1

        ani = FuncAnimation(
            self.fig, 
            self.update_animation, 
            frames=range(0, len(self.history), step),
            init_func=self.init_animation, 
            interval=3,
            repeat=False
        )

        plt.title("TSP Route Animation")

        # Convert animation to HTML for display
        ani.interactive = True 
        html_anim = ani.to_jshtml()
        display(HTML(html_anim))


class TSPInstance:
    """
    Loads a TSP instance via tsplib95 and prepares data (coordinates, distance matrix).
    """

    def __init__(self, plot_route, instance_file):
        """
        :param plot_route: Boolean (0 or 1) to indicate whether to plot/animate the route.
        :param instance_file: File path to the TSP instance (TSPLIB format).
        """
        self.plot_route = bool(plot_route)
        self.plot_enabled = self.plot_route

        self.coord_x = []
        self.coord_y = []
        self.problem = tsplib95.load(instance_file)
        self.info = self.problem.as_keyword_dict()
        self.n = len(self.problem.get_graph())

        # If the instance can be plotted (EUC_2D, GEO, ATT), save city coordinates
        if self.plot_route and self._can_plot():
            for i in range(1, self.n + 1):
                x, y = self.info['NODE_COORD_SECTION'][i]
                self.coord_x.append(x)
                self.coord_y.append(y)
        else:
            self.plot_route = False

    def _can_plot(self):
        """
        Checks if the TSP instance has a coordinate-based distance (e.g., EUC_2D, GEO, ATT).
        """
        dist_type = self.info['EDGE_WEIGHT_TYPE']
        if dist_type in ['EUC_2D', 'GEO', 'ATT']:
            return True
        else:
            print("Plotting is not supported for this EDGE_WEIGHT_TYPE.")
            return False

    def generate_distance_matrix(self):
        """
        Generate the global distance matrix (dist_matrix) for the TSP.
        """
        global dist_matrix
        dist_matrix = [[INF for _ in range(self.n)] for _ in range(self.n)]
        start_node = list(self.problem.get_nodes())[0]

        # Adjust if the node indices start at 1 or 0
        if start_node == 0:
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        dist_matrix[i][j] = self.problem.get_weight(i, j)
        else:
            # If nodes start at 1 instead of 0
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        dist_matrix[i][j] = self.problem.get_weight(i + 1, j + 1)


def distance(i, j):
    """
    Returns the distance between city i and city j using the global distance matrix.
    """
    return dist_matrix[i][j]


def total_cost(route):
    """
    Evaluates the total cost of a TSP route (closed tour).
    """
    csum = 0
    for k in range(len(route) - 1):
        csum += distance(route[k], route[k + 1])
    # Add cost from last city back to the first city
    csum += distance(route[-1], route[0])
    return (csum,)


def nearest_neighbor(n):
    """
    Generates a TSP route using the Nearest Neighbor heuristic with probability 0.4,
    otherwise shuffles cities randomly.

    :param n: Number of cities.
    :return: A route (list or numpy array) representing a permutation of cities.
    """
    start = random.randrange(0, n)
    if random.uniform(0, 1) < 0.4:
        current = start
        route = [start]
        selected = [False] * n
        selected[current] = True

        while len(route) < n:
            min_dist = INF
            next_city = None
            for candidate in range(n):
                if not selected[candidate] and candidate != current:
                    cost_val = distance(current, candidate)
                    if cost_val < min_dist:
                        min_dist = cost_val
                        next_city = candidate

            route.append(next_city)
            selected[next_city] = True
            current = next_city
    else:
        route = list(range(n))
        random.shuffle(route)

    if use_numpy:
        return np.array(route)
    else:
        return route


def two_opt(route):
    """
    2-Opt local search: tries to improve the route by reversing segments.
    """
    n = len(route)
    improved = False
    best_delta = 0
    cut_count = 0

    # Shift the route starting point randomly
    k = random.randint(0, n - 1)
    if use_numpy:
        route = np.hstack((route[k:], route[:k]))  # rotate with numpy
    else:
        route = route[k:] + route[:k]

    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            old_cost = distance(route[i], route[i + 1]) + distance(route[j], route[j + 1])
            new_cost = distance(route[i], route[j]) + distance(route[i + 1], route[j + 1])
            delta = new_cost - old_cost

            if delta < best_delta:
                best_delta = delta
                min_i, min_j = i, j
                cut_count += 1
                # Only make one improving swap
                if cut_count == 1:
                    improved = True

        if improved:
            break

    if cut_count > 0:
        segment = route[min_i + 1: min_j + 1]
        route[min_i + 1: min_j + 1] = segment[::-1]


def perturbation_swap_two(route):
    """
    Perturbation: randomly swap two distinct cities in the route.
    """
    i, j = 0, 0
    n = len(route)
    while i == j:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)

    route[i], route[j] = route[j], route[i]


def perturbation_swap_neighbors(route):
    """
    Perturbation: choose one city at random and swap it with its immediate neighbor.
    """
    n = len(route)
    i = random.randint(0, n - 1)
    j = i + 1 if i < n - 1 else 0

    route[i], route[j] = route[j], route[i]


def perturbation_reverse_subroute(route):
    """
    Perturbation 2: choose two random points i, j (i < j) and reverse the subroute between them.
    """
    i, j = 0, 0
    n = len(route)
    while i >= j:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)

    route[i:j] = route[i:j][::-1]


def mutate(route):
    """
    Custom mutation strategy that applies a subroute reversal (2).
    Other perturbations are commented out but can be included if desired.
    """
    # Examples of different perturbations:
    # perturbation_swap_two(route)
    # perturbation_swap_neighbors(route)
    # two_opt(route)
    perturbation_reverse_subroute(route)
    return (route,)


def ga_simple(tsp_instance, seed):
    """
    Simple GA to solve TSP.
    :param tsp_instance: TSPInstance object.
    :param seed: Random seed to control reproducibility.
    """
    population_size = 50
    max_gens = 200
    cx_prob = 0.9
    mut_prob = 0.4
    tournament_size = 4
    n_cities = tsp_instance.n

    random.seed(seed)

    # Create the Fitness and Individual classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if use_numpy:
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    else:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Register functions
    toolbox.register("indices", nearest_neighbor, n_cities)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", total_cost)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", mutate)

    # Generate initial population
    pop = toolbox.population(n=population_size)
    if use_numpy:
        hof = tools.HallOfFame(1, similar=np.array_equal)
    else:
        hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    start_time = time.time()
    final_population, logbook = algorithms.eaSimple(
        pop, toolbox, cx_prob, mut_prob, max_gens,
        stats=stats, halloffame=hof
    )
    end_time = time.time()

    min_list, avg_list = logbook.select("min", "avg")

    print(f"Best route cost: {min(min_list)}")
    print(f"Execution time : {end_time - start_time}")

    if tsp_instance.plot_enabled:
        plot_evolution(min_list, avg_list)


def ga_advanced(tsp_instance, seed):
    """
    Advanced GA to solve TSP with custom selection, crossover, mutation, and partial 2-Opt local search steps.
    :param tsp_instance: TSPInstance object.
    :param seed: Random seed to control reproducibility.
    """
    population_size = 100
    max_gens = 1000
    cx_prob = 0.9
    mut_prob = 0.1
    tournament_size = 4
    n_cities = tsp_instance.n

    random.seed(seed)

    # Define Fitness and Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if use_numpy:
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    else:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("indices", nearest_neighbor, n_cities)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", total_cost)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", mutate)

    # Build initial population
    pop = toolbox.population(n=population_size)
    if use_numpy:
        hof = tools.HallOfFame(1, similar=np.array_equal)
    else:
        hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Evaluate initial population
    start_time = time.time()
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit_val in zip(pop, fitnesses):
        ind.fitness.values = fit_val

    gen = 0
    solutions_history = []
    costs_history = []

    record = stats.compile(pop)
    logbook.record(gen=gen, evals=len(pop), **record)
    print(logbook[-1]["gen"], logbook[-1]["avg"], logbook[-1]["min"])

    # Evolve
    while gen < max_gens:
        gen += 1

        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Re-evaluate mutated/crossover offsprings
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_inds)
        for ind, fit_val in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit_val

        # Replace population
        pop[:] = offspring
        hof.update(offspring)

        record = stats.compile(offspring)
        logbook.record(gen=gen, evals=len(offspring), **record)
        print(logbook[-1]["gen"], logbook[-1]["avg"], logbook[-1]["min"])

        # Store best route of this generation
        best_ind = tools.selBest(offspring, k=1)[0]
        solutions_history.append(best_ind)
        costs_history.append(int(logbook[-1]["min"]))

    end_time = time.time()

    print(f"Best route cost: {min(costs_history)}")
    print(f"Execution time : {end_time - start_time}")

    # If route plotting is enabled, animate the route
    if tsp_instance.plot_route:
        anim = AnimationTSP(solutions_history, tsp_instance.coord_x, tsp_instance.coord_y, costs_history)
        anim.animate_routes()

    # If cost plot is enabled, plot cost evolution
    if tsp_instance.plot_enabled:
        min_values, avg_values = logbook.select("min", "avg")
        plot_evolution(min_values, avg_values)


def plot_evolution(min_values, avg_values):
    """
    Plots the evolution of the best and average cost over generations.
    """
    plt.figure()
    plot1, = plt.plot(min_values, 'c-', label='Best Cost')
    plot2, = plt.plot(avg_values, 'b-', label='Average Cost')
    plt.legend(handles=[plot1, plot2], frameon=True)
    plt.ylabel('Cost')
    plt.xlabel('Generations')
    plt.title("Generations vs. Cost - TSP")
    plt.xlim((0, len(min_values)))
    plt.show()


def main():
    """
    Main function that uses a small menu to collect user input (rather than sys.argv).
    """
    print("=========================================")
    print("  TSP GA Solver - Interactive Menu")
    print("=========================================")

    # Gather user input
    plot_flag = int(input("Enable route plotting/animation? (0=No, 1=Yes): "))
    instance_file = input("Enter the file path of the TSP instance (e.g., kroA100.tsp): ")
    seed_value = int(input("Enter random seed (integer), e.g. 42: "))
    version = int(input("Choose GA version (0=Simple, 1=Advanced): "))
    
    global use_numpy
    use_numpy = int(input("Use numpy arrays for individuals? (0=No, 1=Yes): "))

    # Create TSP instance
    tsp_instance = TSPInstance(plot_flag, instance_file)
    tsp_instance.generate_distance_matrix()

    # Run the chosen GA version
    if version == 0:
        ga_simple(tsp_instance, seed_value)
    else:
        ga_advanced(tsp_instance, seed_value)


if __name__ == "__main__":
    main()
