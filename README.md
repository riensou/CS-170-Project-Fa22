# Fall 2022 CS 170 Project

How to replicate our results:

1. Install the following requirements:
    * networkx
    * numpy
    * tqdm
    * matplotlib
    * multiprocessing
    * cvxpy (only for testing sdp_algorithm.py)

2. Create a list of input files in a folder called 'input'. This can easily be done with 'random_solve' from main.ipynb

3. Generate some initial solutions to these input files (and put them in a folder called 'output') using the solvers in the following files:
    * greedy_algorithm.py
    * k_cut_algorithm.py
    * sdp_algorithm.py
    * main.ipynb

4. Run "python3 simulated_annealing.py" to generate .out files using the multiple search operator heuristic algorithm.


note: The code is structured in a way such that anytime a new best solution is found, it is saved. Thus, it is possible that certain results may not occur if only using the current setup. While the current setup is generally the best, it may be the case for specific test cases, previous iterations of the algorithm produced better results. The main differences in these previous iterations include different hyperparameter values, using a different k, or starting from a different initial solution.
