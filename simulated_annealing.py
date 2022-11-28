import random
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from starter import *


in_dir = "inputs"
out_dir = "outputs"

def simulated_annealing(file, overwrite=True):

    start_t = time.perf_counter()

    in_file = str(Path(in_dir) / file)
    out_file = str(Path(out_dir) / f"{file[:-len('.in')]}.out")

    initial_score = best_score = score(read_output(read_input(in_file), out_file))

    G = read_output(read_input(in_file), out_file)

    # Implement efficient simulated annealing algorithm
    change = list(G.nodes)
    for _ in range(30):
        output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
        teams, counts = np.unique(output, return_counts=True)
        k = np.max(teams)
        C = {}
        for i in G.nodes:
            C[i] = sum(d for u, v, d in G.edges(data='weight') if output[u] == output[v] and u == i)

        most_impact = max(change, key=lambda x: C[x])

        G.nodes[most_impact]['team'] = random.randint(1, k)

        current_score = score(G)

        if best_score > current_score:
            best_score = current_score
        else:
            change.remove(most_impact)
            G.nodes[most_impact]['team'] = output[most_impact]
    # Implement efficient simulated annealing algorithm


    if overwrite and initial_score > best_score:
        write_output(G, out_file, overwrite=True)

    end_t = time.perf_counter()

    return in_file, initial_score - best_score, end_t - start_t



# These operators are based off of the following:
# "A Multiple Search Operator Heuristic for the Max-k-cut Problem", by Fuda Ma and Jin-Kao Hao

def O1_operator(G: nx.Graph):
    """Applies the single-transfer move operation such that the induced move gain is maximum."""
    # TODO
    pass

def O2_operator(G: nx.Graph):
    """Applies the double-transfer move operation such that the induced move gain is maximum."""
    # TODO
    pass

# TODO: O3_operator
# TODO: O4_operator

def O5_operator(G: nx.Graph):
    """Assigns a random vertex to a random team such that the new team is different than the current team."""
    # TODO
    pass



if __name__ == '__main__':

    filenames = [x for x in os.listdir(in_dir) if x.endswith('.in')]

    with Pool() as pool:
        results = pool.imap_unordered(simulated_annealing, filenames)

        for filename, improvement, duration in results:
            print(f"{filename}: improved by {improvement:.2f} in {duration:.2f}s")