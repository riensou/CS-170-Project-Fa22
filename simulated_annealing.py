import random
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from starter import *

def helper(output_1, output_2, team):
    """
    Helps with figuring out C_w_partial
    """
    if team == output_2:
        return 1
    elif output_1 == output_2:
        return -1
    else:
        return 0
class Bucket_Structure():

    def __init__(self, G):

        self.G = G

        C_w, C_k, C_b = score(G, separated=True)

        output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
        teams, counts = np.unique(output, return_counts=True)

        self.k = np.max(teams)

        b = (counts / G.number_of_nodes()) - 1 / self.k
        b_norm = np.linalg.norm(b, 2)
        n = G.number_of_nodes()

        self.buckets = [{} for _ in range(self.k)]
        self.vertices_key = [{} for _ in range(G.number_of_nodes())] 

        C_team_size = [[None for _ in range(self.k)] for _ in range(self.k)]
        for v in G.nodes:
            for team in range(self.k):
                if G.nodes[v]['team'] == team + 1:
                    continue

                ##FOR TESTING
                # G_copy = G.copy()
                # G_copy.nodes[v]['team'] = team + 1
                # new_C_w, new_C_k, new_C_b = score(G_copy, separated=True)

                # print("C_w change:", new_C_w - C_w)
                # print("C_b change:", new_C_b - C_b)
                # print("Overall change:", (new_C_w - C_w) + (new_C_b - C_b))
                ##FOR TESTING

                C_w_partial = sum(d * helper(output[i], output[j], team + 1) for i, j, d in G.edges(v, data='weight') if i == v)
                
                if not C_team_size[output[v] - 1][team]:
                    inside_sqrt = b_norm ** 2 - b[output[v] - 1] ** 2 - b[team] ** 2 + (b[output[v] - 1] - 1 / n) ** 2 + (b[team] + 1 / n) ** 2
                    # if inside_sqrt < 0:
                    #     inside_sqrt = 0
                    C_team_size[output[v] - 1][team] = math.exp(B_EXP * np.sqrt(inside_sqrt)) - C_b
                gain = round(C_w_partial + C_team_size[output[v] - 1][team], 10)

                ##FOR TESTING
                # print("calculated C_w change:", C_w_partial)
                # print("calculated C_b change:", C_team_size[output[v] - 1][team])
                # print("calculated Overall change:", gain)
                ##FOR TESTING

                if gain in self.buckets[team]:
                    self.buckets[team][gain].append(v)
                else:
                    self.buckets[team][gain] = [v]

                self.vertices_key[v][team] = gain

    def update(self, move):

        self.G.nodes[move[0]]['team'] = move[1]

        G = self.G

        C_w, C_k, C_b = score(self.G, separated=True)

        output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
        teams, counts = np.unique(output, return_counts=True)

        b = (counts / G.number_of_nodes()) - 1 / self.k
        b_norm = np.linalg.norm(b, 2)
        n = len(G.nodes)

        C_team_size = [[None for _ in range(self.k)] for _ in range(self.k)]

        change_nodes = [move[0]]
        for i, j, d in G.edges(move[0], data='weight'):
            if i == move[0] and d > 0:
                change_nodes.append(j)

        for v in change_nodes:
            for team in self.vertices_key[v]:
                self.buckets[team][self.vertices_key[v][team]].remove(v)
                if not self.buckets[team][self.vertices_key[v][team]]:
                    self.buckets[team].pop(self.vertices_key[v][team])
            self.vertices_key[v].clear()


        for v in change_nodes:
            for team in range(self.k):
                if self.G.nodes[v]['team'] == team + 1:
                    continue

                ##FOR TESTING
                # G_copy = self.G.copy()
                # G_copy.nodes[v]['team'] = team + 1
                # new_C_w, new_C_k, new_C_b = score(G_copy, separated=True)

                # print("C_w change:", new_C_w - C_w)
                # print("C_b change:", new_C_b - C_b)
                # print("Overall change:", (new_C_w - C_w) + (new_C_b - C_b))
                ##FOR TESTING


                C_w_partial = sum(d * helper(output[i], output[j], team + 1) for i, j, d in G.edges(v, data='weight') if i == v)
                
                if not C_team_size[output[v] - 1][team]:
                    inside_sqrt = b_norm ** 2 - b[output[v] - 1] ** 2 - b[team] ** 2 + (b[output[v] - 1] - 1 / n) ** 2 + (b[team] + 1 / n) ** 2
                    # if inside_sqrt < 0:
                    #     inside_sqrt = 0
                    C_team_size[output[v] - 1][team] = math.exp(B_EXP * np.sqrt(inside_sqrt)) - C_b
                gain = round(C_w_partial + C_team_size[output[v] - 1][team], 10)

                ##FOR TESTING
                #print("calculated C_w change:", C_w_partial)
                #print("calculated C_b change:", C_team_size[output[v] - 1][team])
                #print("calculated Overall change:", gain)
                ##FOR TESTING

                if gain in self.buckets[team]:
                    self.buckets[team][gain].append(v)
                else:
                    self.buckets[team][gain] = [v]

                self.vertices_key[v][team] = gain

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
    bucket = Bucket_Structure(G.copy())

    for _ in range(300):

        move, gain = O1_operator(bucket)

        if gain < 0:

            G.nodes[move[0]]['team'] = move[1]

            #bucket = Bucket_Structure(G)
            bucket.update(move)

    #

    current_score = score(G)

    # Write the new file if it is better than the initial score
    if overwrite and initial_score > current_score:
        write_output(G, out_file, overwrite=True)
    #

    end_t = time.perf_counter()

    return in_file, initial_score - current_score, end_t - start_t



# These operators are based off of the following:
# "A Multiple Search Operator Heuristic for the Max-k-cut Problem", by Fuda Ma and Jin-Kao Hao

def O1_operator(bucket):
    """
    Selects the single-transfer move operation such that the induced move gain is maximum.
    Returns the move and the induced move gain. 
    """

    best_bucket = min([[b, min(bucket.buckets[b])] for b in range(bucket.k)], key=lambda x: x[1])

    return [random.choice(bucket.buckets[best_bucket[0]][best_bucket[1]]), best_bucket[0] + 1], min(bucket.buckets[best_bucket[0]])


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

def MOH_algorithm(file, overwrite=True):

    start_t = time.perf_counter()

    # Read in the file, and record the initial score
    in_file = str(Path(in_dir) / file)
    out_file = str(Path(out_dir) / f"{file[:-len('.in')]}.out")

    I = read_output(read_input(in_file), out_file)
    initial_score = score(I)
    #


    ### MOH algorithm ###

    I_best = I.copy() 
    local_optimum_score = initial_score
    best_score = initial_score
    c_non_impv = 0
    iter = 0

    # while not stop condition # here we can set a time limit for the algorithm to run depending on graph size or something
        
        ## Descent-based improvement phase (O1 and O2) ##
        # while I can be improved by O1 and O2

            # while O1 improves I
                # do O1 and update bucket data structure (what is this? read the paper)
                # iter += 1

            # while O2 improves I
                # do O2 and update bucket data structure (what is this? read the paper)
                # iter += 1
        ##
        
        # current_score = score(I)
        # local_optimum_score = current_score
        # if current_score < best_score
            # best_score = current_score
            # I_best = I.copy()
            # c_non_impv = 0
        # else
            # c_non_impv += 1

        ## Diversified improvement phase (O3 and O4) ##
        # c_div = 0

        # while c_div <= lower_omega and score(I) > local_optimum_score

            # if random(0, 1) < lower_rho:
                # apply O3 on I
            # else:
                # apply O4 on I
            # Update bucket data structure and H
            # iter += 1
            # c_div += 1
        ##

        ## Perturbation phase (O5) ##
        # if c_non_impv > lower_xi
            # apply O5 (gamma times)
            # c_non_impv
        ##

    ###
    
    # Write the new file if it is better than the initial score
    if overwrite and initial_score > best_score:
        write_output(I_best, out_file, overwrite=True)
    #

    end_t = time.perf_counter()

    return in_file, initial_score - best_score, end_t - start_t


if __name__ == '__main__':

    filenames = [x for x in os.listdir(in_dir) if x.endswith('.in')]

    with Pool() as pool:
        results = pool.imap_unordered(simulated_annealing, filenames)

        for filename, improvement, duration in results:
            print(f"{filename}: improved by {improvement:.2f} in {duration:.2f}s")