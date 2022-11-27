import random
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from starter import *


in_dir = "inputs"
out_dir = "outputs"

def etl(file):

    in_file = str(Path(in_dir) / file)
    out_file = str(Path(out_dir) / f"{file[:-len('.in')]}.out")

    initial_score = best_score = score(read_output(read_input(in_file), out_file))

    start_t = time.perf_counter()

    G = read_output(read_input(in_file), out_file)

    for _ in range(5):
        output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
        teams, counts = np.unique(output, return_counts=True)
        k = np.max(teams)
        C = {}
        for i in G.nodes:
            C[i] = sum(d for u, v, d in G.edges(data='weight') if output[u] == output[v] and u == i)

        most_impact = max(G.nodes, key=lambda x: C[x])

        G.nodes[most_impact]['team'] = random.randint(1, k)

        current_score = score(G)

        if best_score > current_score:
            best_score = current_score
        else:
            G.nodes[most_impact]['team'] = output[most_impact]

    end_t = time.perf_counter()

    return initial_score - best_score, end_t - start_t
if __name__ == '__main__':
    filenames = [x for x in os.listdir(in_dir) if x.endswith('.in')]

    with Pool() as pool:
        results = pool.imap_unordered(etl, filenames)

        for improvement, duration in results:
            print(f"improved by {improvement:.2f} in {duration:.2f}s")


    # if initial_score > best_score:
    #     print("I would write this")
    #     #write_output(G, out_file, overwrite=True)
    # print(f"{str(in_file)}: cost", score(instance))