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

                # calculate move gain of moving v to team (use HW11)
                C_w_partial = sum(d * helper(output[i], output[j], team) for i, j, d in G.edges(v, data='weight') if i == v) # i should always equal v anyway

                if not C_team_size[output[v] - 1][team - 1]: # if we haven't yet calculated the change from moving something in team a to team b, use the formula from HW 11
                    C_team_size[output[v] - 1][team - 1] = math.exp(B_EXP * np.sqrt(b_norm ** 2 - b[output[v] - 1] ** 2 - b[team - 1] ** 2 + (b[output[v] - 1] - 1 / n) ** 2 + (b[team - 1] + 1 / n) ** 2))

                gain = C_w_partial + C_team_size[output[v] - 1][team - 1]

                if gain in self.buckets[team]:
                    self.buckets[team][gain].append(v)
                else:
                    self.buckets[team][gain] = [v]

                self.vertices_key[v][team] = gain # are vertices zero-indexed?
                    # n dictionaries where each dictionary has k keys (1, k), the values of the dict the gains

    def update(self, move):
        # I think we should store a move as a pair: (v, S) the changed node and the set it's going to

        self.G.nodes[move[0]]['team'] = move[1]

        output = [self.G.nodes[v]['team'] for v in range(self.G.number_of_nodes())]
        teams, counts = np.unique(output, return_counts=True)

        b = (counts / self.G.number_of_nodes()) - 1 / self.k
        b_norm = np.linalg.norm(b, 2)
        n = len(self.G.nodes)

        C_team_size = [[None for _ in range(self.k)] for _ in range(self.k)]

        # create a list of nodes that need to be changed
        # this list consists of the node that was moved, and the nodes that it was connected to

        change_nodes = [move[0]]
        for i, j, d in self.G.nodes(move[0], data='weight'):
            if i == move[0] and d > 0:
                change_nodes.append(j)

        # figure out which stuff needs to get deleted.
        # for v in change_nodes:
        #     for team in range(self.k):
        #         self.buckets[]

        for v in change_nodes:
            for team in range(self.k):
                if self.G.nodes[v]['team'] == team + 1:
                    continue

                # calculate move gain of moving v to team (use HW11)
                C_w_partial = sum(d * helper(output[i], output[j], team) for i, j, d in self.G.edges(v, data='weight') if i == v) # i should always equal v anyway

                if not C_team_size[output[v] - 1][team - 1]: # if we haven't yet calculated the change from moving something in team a to team b, use the formula from HW 11
                    C_team_size[output[v] - 1][team - 1] = math.exp(B_EXP * np.sqrt(b_norm ** 2 - b[output[v] - 1] ** 2 - b[team - 1] ** 2 + (b[output[v] - 1] - 1 / n) ** 2 + (b[team - 1] + 1 / n) ** 2))

                gain = C_w_partial + C_team_size[output[v] - 1][team - 1]

                if gain in self.buckets[team]:
                    self.buckets[team][gain].append(v)
                else:
                    self.buckets[team][gain] = [v]

                self.vertices_key[v][team] = gain # are vertices zero-indexed?
                    # n dictionaries where each dictionary has k keys (1, k), the values of the dict the gains

        pass


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