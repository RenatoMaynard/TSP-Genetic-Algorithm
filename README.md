# TSP Genetic Algorithm in Python

This repository contains a **Genetic Algorithm (GA)** implementation for solving the **Traveling Salesman Problem (TSP)**. It uses the following Python libraries:

- [**tsplib95**](https://pypi.org/project/tsplib95/) to parse TSPLIB-compatible TSP instances.
- [**DEAP**](https://deap.readthedocs.io/en/master/) to build and evolve the population.
- [**matplotlib**](https://matplotlib.org/) (for plotting and optional animation).
- [**numpy**](https://numpy.org/) (optional, for representing routes).
- [**IPython**](https://ipython.org/) (needed if you want to see the animations inline in a Jupyter notebook).

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Code Overview](#code-overview)
   - [Global Variables](#global-variables)
   - [AnimationTSP Class](#animationtsp-class)
   - [TSPInstance Class](#tspinstance-class)
   - [Genetic Algorithm Functions](#genetic-algorithm-functions)
   - [Helper Functions](#helper-functions)
   - [Main Function](#main-function)
5. [Notes on Plotting and Animations](#notes-on-plotting-and-animations)
6. [License](#license)

---

## Features

- **Simple GA** and an **Advanced GA** mode:
  - *Simple GA* uses a basic approach for selection, crossover, and mutation.
  - *Advanced GA* includes larger population sizes, more generations, and can integrate local search steps (such as 2-Opt).
- **Multiple mutation/perturbation** methods are provided, including:
  - Swapping two random cities.
  - Swapping neighbors.
  - Reversing sub-routes.
  - 2-Opt local improvement (commented out but can be activated).
- **Visualization**:
  - Plots the best cost and average cost over generations.
  - Optionally animates the evolving route (in Jupyter/IPython environments).
- **Interactive menu** instead of command-line arguments, making it more straightforward to use within a Python environment or when running the script directly.

---

## Installation

1. **Clone** or **download** this repository.
2. Install the required libraries:
   ```bash
   pip install tsplib95 deap matplotlib numpy ipython

## Usage 

1. Run the script (e.g., python ga_interactive.py), or open it in an environment like Jupyter Notebook and run all cells.
2. You will see a **menu**:
```
=========================================
  TSP GA Solver - Interactive Menu
=========================================
Enable route plotting/animation? (`0=No`, `1=Yes`):
Enter the file path of the TSP instance (e.g., `kroA100.tsp`):
Enter random seed (integer), e.g. `42`:
Choose GA version (`0=Simple`, `1=Advanced`):
Use numpy arrays for individuals? (`0=No`, `1=Yes`):
```
3. Enter the requested inputs:
 - Enable route plotting/animation?
   - `1` enables a line plot of best/average costs and, if using a Jupyter environment, an animation of the route.
   - `0` disables plotting and animation.

- File path of the TSP instance: e.g., `kroA100.tsp`

- Random seed: Any integer to make results reproducible (e.g., `73`).

- GA version:
  - `0` runs the simple GA.
  - `1` runs the advanced GA (larger population, more generations, etc.).

- Use numpy arrays:
  - `1` to store individuals (routes) as numpy.ndarray
  - `0` to store individuals as a standard Python list.

4. The algorithm will begin executing. You will see logs of each generation (especially in the advanced version).

5. After finishing, it will output:
- Best route cost found.
- Execution time (in seconds).
- Plots of cost evolution (if plotting is enabled).
- An animation of the best route per generation (in Jupyter/IPython) if you chose `1` in the first question and you have the environment to display it.

## Code Overview

The code is structured as follows:

```python
ga_interactive.py
└── (entry point) main()
```


### Global Variables

- **INF**: A large constant (`9999999`) used to represent infinite distance in the distance matrix.
- **dist_matrix**: A global 2D list that holds pairwise distances between cities.
- **use_numpy**: A global integer set from the user’s input in the menu (`0` or `1`). If `1`, routes are stored as NumPy arrays.

### AnimationTSP Class

```python
class AnimationTSP:
    ...
```
- **Purpose:** Creates and manages an animation of the TSP route as it evolves.

- **Methods:**
  - **init(self, history, x_coords, y_coords, costs):** Receives the full solution history (each solution is a permutation of city indices), corresponding cost list, and the x/y coordinates of each city.
  - **init_animation(self):** Initializes the plot with city nodes.
  - **update_animation(self, frame):** Updates the route lines for each frame in the history.
  - **animate_routes(self):** Puts everything together and shows the animation in Jupyter using FuncAnimation.
 
### TSPInstance Class

```python
class TSPInstance:
    ...
```
- **Purpose**:
  - Loads a TSP instance from a TSPLIB file.
  - Extracts node coordinates if they exist (for 2D-based problems: `EUC_2D`, `GEO`, `ATT`).
  - Generates the global distance matrix (`dist_matrix`) to quickly retrieve distances.
- **Key methods**:
  - **init(self, plot_route, instance_file):** Sets up the plotting option and reads the TSP file. Checks if coordinates can be plotted.
  - **generate_distance_matrix(self):** Builds the matrix `dist_matrix[i][j]` = distance from city `i` to `j`.
    
### Genetic Algorithm Functions
1. **Fitness Evaluation**
```python
def total_cost(route):
    ...
```
  - Computes the total distance of the closed tour represented by `route`.
2. **Initialization**
```python
def nearest_neighbor(n):
    ...
```
  - Randomly chooses a start city, then with probability 0.4 applies a Nearest Neighbor approach; otherwise fully shuffles the cities.
  - Returns either a Python list or a NumPy array of the city permutation (depending on `use_numpy`).
3. **Mutation (and Perturbations)**
```python
def mutate(route):
    ...
```
  - Applies one of several custom perturbation methods (e.g., reversing a sub-route).
  - You can switch out different mutation methods (e.g., `perturbation_swap_two`, `perturbation_swap_neighbors`, `two_opt`).
4. **GA Routines**
    - ga_simple
        - Population size = 50
        - Generations = 200
        - Uses `eaSimple` from DEAP for a straightforward evolution loop.
    - ga_advanced
        - Larger population (100)
        - More generations (1000)
        - Custom loop (with optional local searches, more detailed logs, etc).
### Helper Functions
- `two_opt(route)`: An optional local search to reverse edges in the route if it yields improvement.
- `perturbation_swap_two`, `perturbation_swap_neighbors`, `perturbation_reverse_subroute`: Different ways to shuffle or reverse parts of the route.
- `plot_evolution(min_values, avg_values)`: Creates a line chart showing how the best cost and average cost evolve over generations.

### Main Function 
```python
def main():
    ...
```
- Presents a **menu** of questions to the user:
    1. Enable route plotting/animation? (`0` or `1`)
    2. TSP instance file path (e.g., `kroA100.tsp`)
    3. Seed (integer)
    4. Version of GA: `0 = simple`, `1 = advanced`
    5. use_numpy: `0` or `1`
- Creates a `TSPInstance` based on these answers, generates the distance matrix, and then calls `ga_simple` or `ga_advanced`.

### Notes on Plotting and Animations 
1. **Ploting:**  The script uses `matplotlib.pyplot` to show the best and average cost across generations.
  - If `plot_flag == 1`, a figure will pop up at the end of the run.
2. **Animation:**  The route animation (showing how the best route changes) works best in a **Jupyter/IPython** environment.
    - If run in a standard terminal, you might **not** see the inline animation (you can still save it to a file, if you modify the code).
    - The `FuncAnimation` calls `to_jshtml()`, so an inline notebook display is ideal.
   
### License
This repository is licensed under the MIT License. You are free to modify, share, and use this code for your own projects.

---

Enjoy solving TSP instances with this Genetic Algorithm! If you have any questions, feel free to open an issue or submit a pull request.
