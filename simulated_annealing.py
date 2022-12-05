import random
import time
import copy
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from starter import *
from k_cut_algorithm import *

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

def random_solve(G: nx.Graph, k):
    """Randomly partitions G into k number of teams."""
    max_per_team = len(G.nodes) / k
    teams = [0 for _ in range(k)]
    for v in G.nodes:
        look_for_team = True
        while look_for_team:
            this_team = random.randint(1, k)
            if teams[this_team - 1] < max_per_team:
                G.nodes[v]['team'] = this_team
                teams[this_team - 1] += 1
                look_for_team = False

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
            for team in range(int(np.shape(b)[0])):
                if G.nodes[v]['team'] == team + 1:
                    continue

                ##FOR TESTING
                # G_copy = G.copy()
                # G_copy.nodes[v]['team'] = team + 1
                # new_C_w, new_C_k, new_C_b = score(G_copy, separated=True)

                # print("move:", [v, team + 1])

                # #print("C_w change:", new_C_w - C_w)
                # #print("C_b change:", new_C_b - C_b)
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

        self.k = np.max(teams)

        b = (counts / G.number_of_nodes()) - 1 / self.k
        b_norm = np.linalg.norm(b, 2)
        n = len(G.nodes)

        C_team_size = [[None for _ in range(self.k)] for _ in range(self.k)]

        # This is to change C_w, but doesn't account for C_b
        # change_nodes = [move[0]]
        # for i, j, d in G.edges(move[0], data='weight'):
        #     assert i == move[0]
        #     if i == move[0] and d > 0:
        #         change_nodes.append(j)
        # print("\nCHANGE NODES:", change_nodes)
        # print("G.edges(..)", G.edges(move[0], data='weight'), "\n")

        for v in G.nodes: #
            for team in self.vertices_key[v]:
                self.buckets[team][self.vertices_key[v][team]].remove(v)
                if not self.buckets[team][self.vertices_key[v][team]]:
                    self.buckets[team].pop(self.vertices_key[v][team])
            self.vertices_key[v].clear()


        for v in G.nodes: #
            for team in range(self.k):
                if self.G.nodes[v]['team'] == team + 1:
                    continue

                ##FOR TESTING
                #G_copy = self.G.copy()
                #G_copy.nodes[v]['team'] = team + 1
                #new_C_w, new_C_k, new_C_b = score(G_copy, separated=True)
                #print("move:", [v, team + 1])
                #print("C_w change:", new_C_w - C_w)
                #print("C_b change:", new_C_b - C_b)
                #print("Overall change:", (new_C_w - C_w) + (new_C_b - C_b))
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

def the_number_one_algorithm(file, overwrite=True):

    start_t = time.perf_counter()

    # Read in the file, and record the initial score
    in_file = str(Path(in_dir) / file)
    out_file = str(Path(out_dir) / f"{file[:-len('.in')]}.out")

    initial_score = score(read_output(read_input(in_file), out_file))

    G = read_output(read_input(in_file), out_file)
    #

    # Implement efficient simulated annealing algorithm
    bucket = Bucket_Structure(G.copy())

    current_k = bucket.k

    final_scores = {}
    
    for K in [current_k - l for l in [1, 0, -1] if current_k - l >= 2]:
        G_copy = G.copy()
        max_k_cut_solve(K)(G_copy)
        bucket = Bucket_Structure(G_copy)

        

        for _ in range(10):

            move, gain = O1_operator(bucket)

            if gain < 0:

                G_copy.nodes[move[0]]['team'] = move[1]

                #bucket = Bucket_Structure(G)
                bucket.update(move)

            else:

                moves, gain = O2_operator(G_copy)

                if gain < 0:

                    for move in moves:
                        G_copy.nodes[move[0]]['team'] = move[1]

                        bucket.update(move)






        #

        final_scores[K] = [score(G_copy), G_copy]

    #print(final_scores)
    
    best_k = min(final_scores, key=lambda x: final_scores[x][0])

    best_score = final_scores[best_k][0]
    best_G = final_scores[best_k][1]

    # Write the new file if it is better than the initial score
    #if overwrite and initial_score > best_score:
    #    write_output(best_G, out_file, overwrite=True)
    #

    end_t = time.perf_counter()

    return in_file, initial_score - best_score, end_t - start_t


# These operators are based off of the following:
# "A Multiple Search Operator Heuristic for the Max-k-cut Problem", by Fuda Ma and Jin-Kao Hao

def O1_operator(bucket):
    """
    Selects the single-transfer move operation such that the induced move gain is maximum.
    Returns the move and the induced move gain. 
    """

    best_bucket = min([[b, min(bucket.buckets[b])] for b in range(bucket.k)], key=lambda x: x[1])

    move = [random.choice(bucket.buckets[best_bucket[0]][best_bucket[1]]), best_bucket[0] + 1]

    gain = best_bucket[1]

    return move, gain


def O2_operator(bucket):
    """
    Selects the double-transfer move operation such that the induced move gain is maximum.
    Returns the move and the induced move gain.
    """

    G = bucket.G
    
    potential_sequences = {}

    output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
    teams, counts = np.unique(output, return_counts=True)
    k = np.max(teams)
    moves = []
    for i in (G.nodes):
        for j in range(1, k + 1):
            if output[i] != j:
                moves.append([i, j])
    
    C_w_1, C_k_1, C_b_1 = score(G, separated=True)

    b = (counts / G.number_of_nodes()) - 1 / k
    b_norm = np.linalg.norm(b, 2)
    n = G.number_of_nodes()

    sampled_moves = random.sample(moves, int(len(moves) / 32))
    for move in sampled_moves:
        current_output = output.copy()
        current_bucket = copy.deepcopy(bucket)

        C_w_partial = sum(d * helper(current_output[i], current_output[j], move[1]) for i, j, d in current_bucket.G.edges(move[0], data='weight') if i == move[0])
                
        inside_sqrt = (b_norm ** 2) - (b[current_output[move[0]] - 1] ** 2) - (b[move[1] - 1] ** 2) + ((b[current_output[move[0]] - 1] - 1 / n) ** 2) + ((b[move[1] - 1] + 1 / n) ** 2)
        C_b_change = math.exp(B_EXP * np.sqrt(inside_sqrt)) - C_b_1
        gain1 = round(C_w_partial + C_b_change, 10)

        ##FOR TESTING
        #print("calculated gain1", gain1)
        #print("calculated C_w_change:", C_w_partial)
        #print("calculated C_b_change:", C_b_change)
        ##FOR TESTING


        current_bucket.update(move)

        ##FOR TESTING
        #C_w_2, C_k_2, C_b_2 = score(current_bucket.G, separated=True)
        #print("true gain1:", score(current_bucket.G) - (C_w_1 + C_k_1 + C_b_1))
        #print("true C_w_change:", C_w_2 - C_w_1)
        #print("true C_b_change:", C_b_2 - C_b_1)
        ##FOR TESTING

        next_move, gain2 = O1_operator(current_bucket)

        total_gain = round(gain1 + gain2, 10)

        ##FOR TESTING
        # calculated_total_gain = total_gain
        # G_copy = G.copy()
        # cur_score = score(G_copy)
        # G_copy.nodes[move[0]]['team'] = move[1]
        # true_gain1 = score(G_copy) - cur_score
        # cur_score = score(G_copy)
        # G_copy.nodes[next_move[0]]['team'] = next_move[1]
        # true_gain2 = score(G_copy) - cur_score
        # true_total_gain = score(G_copy) - (C_w_1 + C_k_1 + C_b_1)

        # if round(calculated_total_gain, 2) != round(true_total_gain, 2):
        #     print("ERROR:\n")
        #     print("true:", true_total_gain)
        #     print("calculated:", calculated_total_gain)
        #     print("")
        #     print("CALCULATED:", "gain1:", gain1, "gain2:", gain2)
        #     print("TRUE:", "gain1:", true_gain1, "gain2:", true_gain2)
        ##FOR TESTING

        # gain2 is somehow being wrong ...



        if total_gain in list(potential_sequences.keys()):
            potential_sequences[total_gain].append([move, next_move])
        else:
            potential_sequences[total_gain] = [[move, next_move]]

    

    min_gain = min(potential_sequences.keys())

    #print(min_gain)
    #print(random.choice(potential_sequences[min_gain]))

    return random.choice(potential_sequences[min_gain]), min_gain

def O3_operator(bucket, H):
    """
    Selects the best single-transfer move, like 01, that is not in the tabu list, H.
    Returns the move and the induced move gain, as well as appending the move to the tabu list H.
    """
    def min_key(x):
        moves = []
        for v in bucket.buckets[x[0]][x[1]]: # is a list of vertices
            moves.append([v, x[0] + 1])

        if any([move not in H for move in moves]):
            return x[1]
        else:
            return float('inf')

    best_bucket = min([[b, min(bucket.buckets[b])] for b in range(bucket.k)], key=min_key)

    #move = [random.choice(bucket.buckets[best_bucket[0]][best_bucket[1]]), best_bucket[0] + 1]
    bruh = [[v, best_bucket[0] + 1] for v in bucket.buckets[best_bucket[0]][best_bucket[1]] if [v, best_bucket[0] + 1] not in H]
    if not bruh:
        # print("WHAT IS HAPPENING :(((((((((((((((((\n")
        # print(bruh)
        return None, None
    

    move = random.choice([[v, best_bucket[0] + 1] for v in bucket.buckets[best_bucket[0]][best_bucket[1]] if [v, best_bucket[0] + 1] not in H])

    gain = best_bucket[1]

    return move, gain

def O4_operator(bucket):
    """
    Selects the two best single-transfer moves for vertices going to randomly selected teams a and b.
    Returns the move and the induced move gain.
    """

    def min_key(team):
        def min_key_helper(x):
            if x[0] + 1 == team:
                return x[1]
            else:
                return float('inf')
        return min_key_helper
        
        
    
    G = bucket.G
    potential_sequences = {}

    output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
    teams, counts = np.unique(output, return_counts=True)
    k = np.max(teams)
    two_go_two = random.sample(range(1, k + 1), 2)
    #print("two_go_two:", two_go_two) #test
    moves = []
    for i in (G.nodes):
        for j in range(1, k + 1):
            if output[i] != j and (j in two_go_two):
                moves.append([i, j])
    
    C_w_1, C_k_1, C_b_1 = score(G, separated=True)

    b = (counts / G.number_of_nodes()) - 1 / k
    b_norm = np.linalg.norm(b, 2)
    n = G.number_of_nodes()


    sampled_moves = random.sample(moves, int(len(moves) / 16))
    for move in sampled_moves:
        current_output = output.copy()
        current_bucket = copy.deepcopy(bucket)

        C_w_partial = sum(d * helper(current_output[i], current_output[j], move[1]) for i, j, d in current_bucket.G.edges(move[0], data='weight') if i == move[0])
                
        inside_sqrt = (b_norm ** 2) - (b[current_output[move[0]] - 1] ** 2) - (b[move[1] - 1] ** 2) + ((b[current_output[move[0]] - 1] - 1 / n) ** 2) + ((b[move[1] - 1] + 1 / n) ** 2)
        C_b_change = math.exp(B_EXP * np.sqrt(inside_sqrt)) - C_b_1
        gain1 = round(C_w_partial + C_b_change, 10)
        
        current_bucket.update(move)

        if move[1] == two_go_two[0]:
            other_team = two_go_two[1]
        else:
            other_team = two_go_two[0]
        best_bucket = min([[b, min(bucket.buckets[b])] for b in range(bucket.k)], key=min_key(other_team))
        next_move = [random.choice(bucket.buckets[best_bucket[0]][best_bucket[1]]), best_bucket[0] + 1]
        gain2 = best_bucket[1]


        total_gain = round(gain1 + gain2, 10)

        if total_gain in list(potential_sequences.keys()):
            potential_sequences[total_gain].append([move, next_move])
        else:
            potential_sequences[total_gain] = [[move, next_move]]

    

    min_gain = min(potential_sequences.keys())

    moves = random.choice(potential_sequences[min_gain])

    #print(min_gain)
    #print(moves)

    return moves, min_gain

def O5_operator(bucket):
    """
    Selects an assignment of a random vertex to a random team such that the new team is different than the current team.
    Returns the move and the induced move gain.
    """
    G = bucket.G

    v = random.sample(G.nodes, 1)[0]
    team = random.sample([i for i in range(1, bucket.k + 1) if i != G.nodes[v]['team']], 1)[0]

    return [v, team]

def MOH_algorithm(file, overwrite=True):

    try:

        start_t = time.perf_counter()

        # Read in the file, and record the initial score
        in_file = str(Path(in_dir) / file)
        out_file = str(Path(out_dir) / f"{file[:-len('.in')]}.out")

        I = read_output(read_input(in_file), out_file)
        initial_score = score(I)
        #

        OMEGA = 25
        RHO = 1
        XI = 10
        GAMMA = int(I.number_of_nodes() / 10)
        ALPHA = 0

        ### MOH algorithm ###

        output = [I.nodes[v]['team'] for v in range(I.number_of_nodes())]
        teams, counts = np.unique(output, return_counts=True)

        k = np.max(teams)

        random_solve(I, k - 2)

        initial_random_score = current_score = score(I)

        bucket = Bucket_Structure(I)
        H = []

        I_best = I.copy() 
        local_optimum_score = initial_random_score
        best_score = initial_score
        c_non_impv = 0
        iter = 0

        # while not stop condition # here we can set a time limit for the algorithm to run depending on graph size or something
        for _ in range(5):
            
            #print(in_file, current_score - initial_score, _ + 1, time.perf_counter() - start_t)
            
            ## Descent-based improvement phase (O1 and O2) ##
            moveO1, gainO1 = O1_operator(bucket)
            gainO2 = 0
            # movesO2, gainO2 = O2_operator(bucket)
            while gainO1 < 0 or gainO2 < 0:
                while gainO1 < 0:
                    I.nodes[moveO1[0]]['team'] = moveO1[1]
                    bucket.update(moveO1)
                    moveO1, gainO1 = O1_operator(bucket)
                    iter += 1

                if random.uniform(0, 1) < ALPHA:
                    movesO2, gainO2 = O2_operator(bucket)
                    if gainO2 > 0:
                        for move in movesO2:
                            I.nodes[move[0]]['team'] = move[1]
                            bucket.update(move)

                # while gainO2 < 0:
                #     for move in movesO2:
                #         I.nodes[move[0]]['team'] = move[1]
                #         bucket.update(move)
                #     movesO2, gainO2 = O2_operator(bucket)
                #     iter += 1
                # moveO1, gainO1 = O1_operator(bucket)
            # Out of phase 1 (O1 and O2)

            current_score = score(I)
            local_optimum_score = current_score
            if current_score < best_score:
                best_score = current_score
                I_best = I.copy()
                c_non_impv = 0
            else:
                c_non_impv += 1

            ## Diversified improvement phase (O3 and O4) ##
            c_div = 0

            while c_div <= OMEGA and current_score >= local_optimum_score:
                LAMBDA = random.randint(3, int(I.number_of_nodes() / 10))

                gain = 0

                if random.uniform(0, 1) < RHO:
                    for tabu in H:
                        if tabu[2] <= iter:
                            H.remove(tabu)
                    move, gain = O3_operator(bucket, [m[0:2] for m in H])
                    if not move: # if O3 fails
                        continue
                    H.append([move[0], I.nodes[move[0]]['team'], iter + LAMBDA])
                    bucket.update(move)
                else:
                    moves, gain = O4_operator(bucket)
                    for move in moves:
                        I.nodes[move[0]]['team'] = move[1]
                        bucket.update(move)
                iter += 1
                c_div += 1

                current_score += gain
            ##

            ## Perturbation phase (O5) ##
            if c_non_impv > XI:
                for _ in range(GAMMA):
                    move = O5_operator(bucket)
                    I.nodes[move[0]]['team'] = move[1]
                    bucket.update(move)
                c_non_impv = 0
            ##

        ###
        
        # Write the new file if it is better than the initial score
        if overwrite and initial_score > best_score:
            write_output(I_best, out_file, overwrite=True)
        #

        end_t = time.perf_counter()

        return in_file, best_score - initial_score, end_t - start_t

    except:

        return "error", 0, 0


if __name__ == '__main__':


    filenames = [x for x in os.listdir(in_dir) if x.endswith('.in')]
    random.shuffle(filenames)

    with Pool() as pool:
        results = pool.imap_unordered(MOH_algorithm, filenames)

        for filename, improvement, duration in results:
            print(f"{filename}: improved by {improvement:.2f} in {duration:.2f}s")