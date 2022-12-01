import random
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from starter import *


in_dir = "inputs"
out_dir = "outputs"

def simulated_annealing(file, overwrite=True):

    start_t = time.perf_counter()

    # Read in the file, and record the initial score
    in_file = str(Path(in_dir) / file)
    out_file = str(Path(out_dir) / f"{file[:-len('.in')]}.out")

    initial_score = best_score = score(read_output(read_input(in_file), out_file))

    G = read_output(read_input(in_file), out_file)
    #

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
    #

    # Write the new file if it is better than the initial score
    if overwrite and initial_score > best_score:
        write_output(G, out_file, overwrite=True)
    #

    end_t = time.perf_counter()

    return in_file, initial_score - best_score, end_t - start_t



# These operators are based off of the following:
# "A Multiple Search Operator Heuristic for the Max-k-cut Problem", by Fuda Ma and Jin-Kao Hao

def O1_operator(G: nx.Graph):
    """
    Selects the single-transfer move operation such that the induced move gain is maximum.
    Returns the move and the induced move gain. 
    """
    # TODO

    # Step 1: find the best single-transfer move operation
        # calculated changed score (use HW11 as a start) of moving each node to each subset
        # there are fixed k groups: 
            # find the partial score change (from group size) from moving something from each group 'a' to each group 'b'
            # search the better score changes first
        # use adjacency list representation of G to quickly find change of sum of weights inside each group


    # Step 2: return the move and the gain

    pass

def O2_operator(G: nx.Graph):
    """
    Selects the double-transfer move operation such that the induced move gain is maximum.
    Returns the move and the induced move gain.
    """
    # TODO

    # This seems hard to implement efficiently, but the paper gives some good advice on how we can do this.
    # (if we can't figure out a better way) we could have a fixed number of moves to search and pick the best of those moves

    pass

def O3_operator(G: nx.Graph):
    """
    Selects the best single-transfer move, like 01, that is not in the tabu list, H.
    Returns the move and the induced move gain, as well as appending the move to the tabu list H.
    """
    # TODO
    pass

def O4_operator(G: nx.Graph):
    """
    Selects the two best single-transfer moves for vertices going to randomly selected teams a and b.
    Returns the move and the induced move gain.
    """
    # TODO
    pass

def O5_operator(G: nx.Graph):
    """
    Selects an assignment of a random vertex to a random team such that the new team is different than the current team.
    Returns the move and the induced move gain.
    """
    # TODO
    pass



if __name__ == '__main__':

    filenames = [x for x in os.listdir(in_dir) if x.endswith('.in')]

    with Pool() as pool:
        results = pool.imap_unordered(simulated_annealing, filenames)

        for filename, improvement, duration in results:
            print(f"{filename}: improved by {improvement:.2f} in {duration:.2f}s")