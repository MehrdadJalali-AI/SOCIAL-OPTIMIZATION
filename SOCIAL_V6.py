import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import ranksums
import itertools
from datetime import datetime
import os
import json
import pickle
from scipy.stats import entropy
import community as community_louvain

# --- Composite Centrality ---
def composite_centrality(G, t):
    deg = nx.degree_centrality(G)
    close = nx.closeness_centrality(G)
    eig = G.graph.get('eig', nx.eigenvector_centrality(G, max_iter=1500))  # Use cached eigenvector centrality
    return {n: 0.5*(1-t)*deg[n] + 0.3*(1-t)*close[n] +
               (0.2+0.3*t)*eig[n] for n in G}

# --- Configuration (Hyperparameters) ---
class Config:
    DIM = 30
    NUM_NODES = 300
    ITERATIONS = 2500
    NUM_RUNS = 20
    K = 5
    P_BASE = 0.1
    ALPHA_INIT = 0.5
    BETA_INIT = 0.5
    GAMMA = 0.2
    DELTA = 0.2
    MUTATION_RATE_INIT = 0.2
    MUTATION_STRENGTH_BASE = 0.1
    MUTATION_STRENGTH_MIN = 0.01
    SYNC_WEIGHT_INIT = 0.03
    SUCCESS_THRESHOLD = 1e-8
    TRACKED_DIM = 0
    LOTUS_R0 = 2.0
    LOTUS_BETA_DROPS = 3
    LOTUS_PIT_CONST = 40
    LOTUS_LOCAL_FREQ = 10

# --- Benchmark Functions with Optima ---
FUNCTIONS = {
    'Sphere': (lambda x: np.sum(x**2), [-100, 100], 0.0),
    'Schwefel_2_22': (lambda x: np.sum(np.abs(x)) + np.prod(np.abs(x)), [-10, 10], 0.0),
    'Schwefel_1_2': (lambda x: np.sum([np.sum(x[:i+1])**2 for i in range(len(x))]), [-100, 100], 0.0),
    'Schwefel_2_21': (lambda x: np.max(np.abs(x)), [-100, 100], 0.0),
    'Rosenbrock': (lambda x: np.sum([100 * (x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1)]), [-30, 30], 0.0),
    'Step': (lambda x: np.sum(np.floor(x+0.5)**2), [-100, 100], 0.0),
    'Quartic': (lambda x: np.sum([(i+1)*xi**4 for i, xi in enumerate(x)]) + np.random.uniform(0, 1), [-1.28, 1.28], 0.0),
    'Schwefel_2_26': (lambda x: 418.9829*len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))), [-500, 500], 0.0),
    'Rastrigin': (lambda x: 10 * len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)), [-5.12, 5.12], 0.0),
    'Ackley': (lambda x: -20 * np.exp(-0.2*np.sqrt(np.sum(x**2)/len(x))) - np.exp(np.sum(np.cos(2*np.pi*x))/len(x)) + 20 + np.e, [-32, 32], 0.0),
    'Griewank': (lambda x: np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1)))) + 1, [-600,600], 0.0),
    'Penalized': (lambda x: (np.pi/len(x))*(10*np.sin(np.pi*(1+(x[0]+1)/4))**2 +
                               np.sum([((1+(x[i]+1)/4)-1)**2 * (1+10*np.sin(np.pi*(1+(x[i+1]+1)/4))**2)
                                       for i in range(len(x)-1)]) +
                               ((1+(x[-1]+1)/4)-1)**2) +
                              np.sum([100*(xi-10)**4 if xi>10 else (-10-xi)**4 if xi < -10 else 0 for xi in x]),
                  [-50,50], 0.0),
    'Penalized2': (lambda x: 0.1*(np.sin(3*np.pi*x[0])**2 +
                                 np.sum([(x[i]-1)**2 * (1+np.sin(3*np.pi*x[i+1])**2)
                                         for i in range(len(x)-1)]) +
                                 (x[-1]-1)**2 * (1+np.sin(2*np.pi*x[-1])**2)) +
                                np.sum([0.1*(xi-5)**4 if xi>5 else (-5-xi)**4 if xi<-5 else 0 for xi in x]),
                   [-50,50], 0.0)
}

def foxholes(x):
    x = x[:2]
    a = np.array([[4.0]*25, np.linspace(0, 12, 25)])
    denom = 1/500.0
    for j in range(25):
        sum_term = (x[0] - a[0, j])**6 + (x[1] - a[1, j])**6
        denom += 1.0 / (j + 1 + sum_term)
    return 1.0 / denom

def kowalik(x):
    x = x[:4]
    a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844,
                  0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    b = np.array([4, 2, 1, 0.5, 0.25,
                  0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625])
    s = 0.0
    for i in range(11):
        s += (a[i] - (x[0]*(b[i]**2 + b[i]*x[1]) / (b[i]**2 + b[i]*x[2] + x[3]*x[2])))**2
    return s

def camel_back(x):
    x = x[:2]
    return 4*x[0]**2 - 2.1*x[0]**4 + (1/3)*x[0]**6 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4

def branin(x):
    x = x[:2]
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)
    return (x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s

def goldstein_price(x):
    x = x[:2]
    term1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[1] - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
    term2 = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[1] + 48*x[0] - 36*x[0]*x[1] + 27*x[1]**2)
    return term1 * term2

def hartman3(x):
    x = x[:3]
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673],
                         [4699, 4387, 7470],
                         [1091, 8732, 5547],
                         [381, 5743, 8828]])
    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i] * ((x - P[i])**2))
        outer += alpha[i] * np.exp(-inner)
    return -outer

def hartman6(x):
    x = x[:6]
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])
    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i] * ((x - P[i])**2))
        outer += alpha[i] * np.exp(-inner)
    return -outer

def shekel5(x):
    x = x[:4]
    m = 5
    C = 0.1 * np.ones(m)
    A = np.array([[4, 4, 4, 4],
                  [1, 1, 1, 1],
                  [8, 8, 8, 8],
                  [6, 6, 6, 6],
                  [3, 7, 3, 7]])
    sum_val = 0.0
    for i in range(m):
        diff = x - A[i]
        sum_val += 1.0 / (np.sum(diff**2) + C[i])
    return -sum_val

def shekel7(x):
    x = x[:4]
    m = 7
    C = 0.1 * np.ones(m)
    A = np.array([[4, 4, 4, 4],
                  [1, 1, 1, 1],
                  [8, 8, 8, 8],
                  [6, 6, 6, 6],
                  [3, 7, 3, 7],
                  [2, 9, 2, 9],
                  [5, 5, 3, 3]])
    sum_val = 0.0
    for i in range(m):
        diff = x - A[i]
        sum_val += 1.0 / (np.sum(diff**2) + C[i])
    return -sum_val

def shekel10(x):
    x = x[:4]
    m = 10
    C = 0.1 * np.ones(m)
    A = np.array([[4, 4, 4, 4],
                  [1, 1, 1, 1],
                  [8, 8, 8, 8],
                  [6, 6, 6, 6],
                  [3, 7, 3, 7],
                  [2, 9, 2, 9],
                  [5, 5, 3, 3],
                  [8, 1, 8, 1],
                  [6, 2, 6, 2],
                  [7, 3.6, 7, 3.6]])
    sum_val = 0.0
    for i in range(m):
        diff = x - A[i]
        sum_val += 1.0 / (np.sum(diff**2) + C[i])
    return -sum_val

FUNCTIONS.update({
    'Foxholes': (foxholes, [-65.536, 65.536], 0.998),
    'Kowalik': (kowalik, [-5, 5], 0.0003075),
    'Camel-Back': (camel_back, [-5, 5], -1.0316),
    'Branin': (branin, [-5, 5], 0.398),
    'Goldstein-Price': (goldstein_price, [-2, 2], 3.0),
    'Hartman': (hartman3, [0, 1], -3.86),
    'Shekel1': (hartman6, [0, 1], -3.322),
    'Shekel2': (shekel5, [0, 10], -10.1532),
    'Shekel3': (shekel7, [0, 10], -10.4028),
    'Shekel4': (shekel10, [0, 10], -10.5363)
})

# --- Initialize Graph and Population ---
def initialize_population(num_nodes, dim, bounds, k, p):
    G = nx.watts_strogatz_graph(num_nodes, k=k, p=p)
    for node in G.nodes:
        G.nodes[node]['position'] = np.random.uniform(bounds[0], bounds[1], dim)
        G.nodes[node]['fitness'] = None
    return G

# --- Evaluate Fitness ---
def evaluate_fitness(G, func):
    for node in G.nodes:
        pos = G.nodes[node]['position']
        G.nodes[node]['fitness'] = func(pos)

# --- Compute Population Entropy (Degree-Based) ---
def compute_population_entropy(G):
    deg_vals = np.array([d for _, d in G.degree()])
    p = deg_vals / deg_vals.sum()
    return entropy(p + 1e-12)

# --- Select Top-K Nodes by PageRank ---
def pick_topk_nodes(G, k):
    pagerank = nx.pagerank(G, alpha=0.85)
    sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes[:k]]

# --- Lotus Shrink Step ---
def lotus_shrink_step(G, elite_pos, bounds, iteration, max_iter, cfg):
    R = cfg.LOTUS_R0 * np.exp(- (4 * iteration / max_iter) ** 2)
    top_k_nodes = pick_topk_nodes(G, k=int(0.1 * len(G)))
    for node in top_k_nodes:
        pos = G.nodes[node]['position']
        new_pos = pos + R * (elite_pos - pos)
        G.nodes[node]['position'] = np.clip(new_pos, bounds[0], bounds[1])

# --- Lotus Reinforcement (Droplet-Overflow) ---
def lotus_reinforcement(G, func, bounds, cfg):
    fits = np.array([G.nodes[n]['fitness'] for n in G.nodes])
    c_max, c_min = fits.max(), fits.min()
    caps = (np.abs(fits - c_max) / (np.abs(c_min - c_max) + 1e-9)) * cfg.LOTUS_PIT_CONST

    pagerank = nx.pagerank(G, alpha=0.85)
    epsilon = 1e-6
    for i, node in enumerate(G.nodes):
        caps[i] /= (pagerank[node] + epsilon)

    partition = community_louvain.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)

    droplets = []
    for i, node in enumerate(G.nodes):
        if caps[i] < 1.0:
            neighbors = list(G.neighbors(node))
            if neighbors:
                neighbor_pos = G.nodes[np.random.choice(neighbors)]['position']
                v = (neighbor_pos - G.nodes[node]['position']) * 0.01
            else:
                v = np.random.randn(cfg.DIM) * 0.01
            droplets.append((node, v))

    for node, v in droplets:
        pos = G.nodes[node]['position'].copy()
        node_comm = partition[node]
        for _ in range(cfg.LOTUS_BETA_DROPS):
            v = 0.9 * v
            if np.random.rand() < 0.8:
                candidates = communities[node_comm]
                if candidates:
                    target_node = np.random.choice(candidates)
                    target_pos = G.nodes[target_node]['position']
                    pos = pos + v * (target_pos - pos) / (np.linalg.norm(target_pos - pos) + 1e-6)
            else:
                pos = pos + v
            pos = np.clip(pos, bounds[0], bounds[1])
        new_fit = func(pos)
        if new_fit < G.nodes[node]['fitness']:
            G.nodes[node]['position'] = pos
            G.nodes[node]['fitness'] = new_fit

# --- Diffusion with Schwefel Adaptations ---
def diffuse(G, gbest_pos, elite_pos, config, bounds, iteration, max_iterations, enable_mutation=True):
    t = iteration / max_iterations
    # Cache eigenvector centrality at iteration 0 and every 50 iterations
    if iteration == 0 or iteration % 50 == 0:
        G.graph['eig'] = nx.eigenvector_centrality(G, max_iter=1500)  # Increased max_iter for robustness
    cent = composite_centrality(G, t)
    fitness_values = np.array([G.nodes[n]['fitness'] for n in G.nodes])
    log_influence = np.log1p(np.abs(fitness_values) + 1e-6)
    influence = 1 - (log_influence / (np.max(log_influence) + 1e-6))

    alpha_t = config.ALPHA_INIT * np.exp(-5 * t) + 0.1
    beta_t = config.BETA_INIT * np.exp(-5 * t) + 0.1
    gamma_t = config.GAMMA * t
    delta_t = config.DELTA * t
    sync_weight_t = config.SYNC_WEIGHT_INIT * (1 - t)

    dimension_mean = np.mean([G.nodes[n]['position'] for n in G.nodes], axis=0)
    new_positions = {}

    # Community-aware weak-ties
    if iteration % 25 == 0:
        part = community_louvain.best_partition(G)
        comm_best = {}
        for node, comm in part.items():
            if comm not in comm_best or G.nodes[node]['fitness'] < G.nodes[comm_best[comm]]['fitness']:
                comm_best[comm] = node
        for c, src in comm_best.items():
            dst = np.random.choice([comm_best[d] for d in comm_best if d != c])
            G.add_edge(src, dst, tmp=True)

    # Cache personalized PageRank, refresh every 100 iterations during exploitation
    pr_cache = G.graph.get('pr_cache')
    elite_idx = min(G.nodes, key=lambda n: G.nodes[n]['fitness'])
    if gamma_t + delta_t > alpha_t + beta_t and (pr_cache is None or iteration % 100 == 0):
        pr_cache = nx.pagerank(G, alpha=0.85, personalization={elite_idx: 1})
        G.graph['pr_cache'] = pr_cache

    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if not neighbors:
            new_positions[node] = G.nodes[node]['position']
            continue

        if pr_cache is not None:
            influence = np.array([pr_cache[n] for n in neighbors])

        weights = [alpha_t * cent[n] + beta_t * influence[i] for i, n in enumerate(neighbors)]
        weights = np.array(weights)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        neighbor_positions = [G.nodes[n]['position'] for n in neighbors]
        neighbor_contribution = np.average(neighbor_positions, weights=weights, axis=0)

        new_pos = (G.nodes[node]['position'] * (1 - alpha_t - beta_t - gamma_t - delta_t) +
                   neighbor_contribution * (alpha_t + beta_t) +
                   gbest_pos * gamma_t +
                   elite_pos * delta_t)
        new_pos = new_pos * (1 - sync_weight_t) + dimension_mean * sync_weight_t
        new_positions[node] = np.clip(new_pos, bounds[0], bounds[1])

    # Remove temporary weak-tie edges and clear PageRank cache if graph rewires
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d.get('tmp')])

    # Compute entropy for rewiring and mutation
    entropy_val = compute_population_entropy(G)
    normalized_entropy = entropy_val / np.log(config.NUM_NODES)

    if enable_mutation:
        mutation_rate_t = config.MUTATION_RATE_INIT * (1 - t) + 0.01
        mutation_strength = config.MUTATION_STRENGTH_BASE * (1 - t)
        median_fitness = np.median(fitness_values)

        # Apply entropy-controlled mutation rate boost once
        if normalized_entropy < 0.4:
            mutation_rate_t += 0.15 * (0.4 - normalized_entropy)

        for node in G.nodes:
            pos = new_positions[node]
            if G.nodes[node]['fitness'] > median_fitness and np.random.rand() < mutation_rate_t:
                sigma_i = mutation_strength * (1 - cent[node]**1.5) + config.MUTATION_STRENGTH_MIN
                perturb = np.random.uniform(-sigma_i * (bounds[1] - bounds[0]), sigma_i * (bounds[1] - bounds[0]), len(pos))
                pos = np.clip(pos + perturb, bounds[0], bounds[1])
            if iteration % 10 == 0 and np.random.rand() < 0.05:
                pos += np.random.uniform(-1, 1, len(pos)) * 0.5
            G.nodes[node]['position'] = pos
    else:
        for node, pos in new_positions.items():
            G.nodes[node]['position'] = pos

    # Sparse in-place rewiring
    if iteration % 75 == 0 and normalized_entropy < 0.5:
        nx.double_edge_swap(G, nswap=int(0.1 * G.number_of_edges()), max_tries=1000)
        # Clear cached eigenvector centrality and PageRank after rewiring
        G.graph.pop('eig', None)
        G.graph.pop('pr_cache', None)

    return G

# --- SOCIAL Optimizer with Data Collection ---
def social_optimize(func_name, config=Config(), enable_mutation=True):
    func, bounds, optimum = FUNCTIONS[func_name]
    G = initialize_population(config.NUM_NODES, config.DIM, bounds, config.K, config.P_BASE)

    search_history = []
    first_agent_trajectory = []
    avg_fitness_history = []
    convergence_history = []

    evaluate_fitness(G, func)
    fitness_values = [G.nodes[n]['fitness'] for n in G.nodes]
    gbest_idx = np.argmin(fitness_values)
    gbest_pos = G.nodes[gbest_idx]['position'].copy()
    gbest_fitness = fitness_values[gbest_idx]
    elite_pos = gbest_pos.copy()
    elite_fitness = gbest_fitness

    for iteration in range(config.ITERATIONS):
        evaluate_fitness(G, func)
        fitness_values = [G.nodes[n]['fitness'] for n in G.nodes]

        positions = [G.nodes[n]['position'][config.TRACKED_DIM] for n in G.nodes]
        search_history.append(positions)
        first_agent_trajectory.append(G.nodes[0]['position'][config.TRACKED_DIM])
        avg_fitness = np.mean(fitness_values)
        avg_fitness_history.append(avg_fitness)
        convergence_history.append(elite_fitness)

        min_idx = np.argmin(fitness_values)
        if fitness_values[min_idx] < gbest_fitness:
            gbest_pos = G.nodes[min_idx]['position'].copy()
            gbest_fitness = fitness_values[min_idx]

        if gbest_fitness < elite_fitness:
            elite_pos = gbest_pos.copy()
            elite_fitness = gbest_fitness

        G = diffuse(G, gbest_pos, elite_pos, config, bounds, iteration, config.ITERATIONS, enable_mutation)
        lotus_shrink_step(G, elite_pos, bounds, iteration, config.ITERATIONS, config)
        if iteration % config.LOTUS_LOCAL_FREQ == 0:
            lotus_reinforcement(G, func, bounds, config)

    return {
        'search_history': search_history,
        'first_agent_trajectory': first_agent_trajectory,
        'avg_fitness_history': avg_fitness_history,
        'convergence_history': convergence_history
    }, elite_pos, elite_fitness, G

# --- Wilcoxon Rank-Sum Test ---
def wilcoxon_test(function_results):
    wilcoxon_results = []
    function_names = list(function_results.keys())
    for func1, func2 in itertools.combinations(function_names, 2):
        fitness1 = function_results[func1]
        fitness2 = function_results[func2]
        stat, p_value = ranksums(fitness1, fitness2)
        wilcoxon_results.append({
            "Function 1": func1,
            "Function 2": func2,
            "Statistic": stat,
            "p-value": p_value,
            "Significant (p<0.05)": p_value < 0.05
        })
    return wilcoxon_results

# --- Run, Compute Metrics, and Save to CSV ---
def run_all_functions(config=Config()):
    temp_dir = "temp_results"
    output_dir = "qualitative_results"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    current_date = datetime.now().strftime("%Y-%m-%d")
    results_filename = f'social_results_{current_date}.csv'
    wilcoxon_filename = f'wilcoxon_results_{current_date}.csv'

    checkpoint_file = os.path.join(temp_dir, "checkpoint.json")
    checkpoint_data = {'completed_functions': [], 'run_progress': {}}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            checkpoint_data.setdefault('completed_functions', [])
            checkpoint_data.setdefault('run_progress', {})
        except:
            print("Error reading checkpoint file. Starting from scratch.")

    completed_functions = checkpoint_data['completed_functions']
    run_progress = checkpoint_data['run_progress']

    results = []
    function_results = {}
    temp_fitness_file = os.path.join(temp_dir, "temp_fitness_values.pkl")
    if os.path.exists(temp_fitness_file):
        try:
            with open(temp_fitness_file, 'rb') as f:
                function_results = pickle.load(f)
        except:
            print("Error reading temporary fitness file. Starting fresh.")

    for name in completed_functions:
        temp_result_file = os.path.join(temp_dir, f"temp_results_{name}.csv")
        if os.path.exists(temp_result_file):
            try:
                temp_df = pd.read_csv(temp_result_file)
                results.append(temp_df.to_dict('records')[0])
            except:
                print(f"Error reading {temp_result_file}. Skipping {name}.")

    for name, (func, bounds, optimum) in FUNCTIONS.items():
        if name in completed_functions:
            print(f"Skipping {name} (already completed)")
            continue

        print(f"Processing {name}...")
        best_fits = function_results.get(name, [])
        all_histories = []
        final_populations = []

        func_dir = os.path.join(output_dir, name)
        os.makedirs(func_dir, exist_ok=True)

        start_run = run_progress.get(name, 0)
        for run in range(start_run, config.NUM_RUNS):
            print(f"  Run {run+1}/{config.NUM_RUNS}")
            data, _, best_fit, G = social_optimize(name, config)
            best_fits.append(best_fit)
            all_histories.append(data['convergence_history'])
            final_populations.append(G)

            if run == 0:
                search_df = pd.DataFrame(
                    data['search_history'],
                    columns=[f'Agent_{i}' for i in range(config.NUM_NODES)],
                    index=[f'Iteration_{i}' for i in range(config.ITERATIONS)]
                )
                search_df.to_csv(os.path.join(func_dir, 'search_history.csv'))

                trajectory_df = pd.DataFrame({
                    'Iteration': range(config.ITERATIONS),
                    'Position': data['first_agent_trajectory']
                })
                trajectory_df.to_csv(os.path.join(func_dir, 'first_agent_trajectory.csv'), index=False)

                avg_fitness_df = pd.DataFrame({
                    'Iteration': range(config.ITERATIONS),
                    'Average_Fitness': data['avg_fitness_history']
                })
                avg_fitness_df.to_csv(os.path.join(func_dir, 'avg_fitness.csv'), index=False)

                convergence_df = pd.DataFrame({
                    'Iteration': range(config.ITERATIONS),
                    'Best_Fitness': data['convergence_history']
                })
                convergence_df.to_csv(os.path.join(func_dir, 'convergence.csv'), index=False)

            run_progress[name] = run + 1
            checkpoint_data = {'completed_functions': completed_functions, 'run_progress': run_progress}
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)

        completed_functions.append(name)
        if name in run_progress:
            del run_progress[name]
        checkpoint_data = {'completed_functions': completed_functions, 'run_progress': run_progress}
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        function_results[name] = best_fits

        best_fits = np.array(best_fits)
        mean_fit = np.mean(best_fits)
        std_fit = np.std(best_fits)
        robustness = std_fit**2
        diversity = np.mean([np.mean([np.std([G.nodes[n]['position'][i] for n in G.nodes])
                                      for i in range(config.DIM)]) for G in final_populations])
        convergence_speed = next((i for i, fit in enumerate(all_histories[0]) if fit < mean_fit + std_fit), config.ITERATIONS)
        success_rate = np.mean([1 if abs(bf - optimum) < config.SUCCESS_THRESHOLD else 0 for bf in best_fits])

        result = {
            "Function": name,
            "Best Fitness": np.min(best_fits),
            "Worst Fitness": np.max(best_fits),
            "Mean Fitness": mean_fit,
            "Std Dev": std_fit,
            "Robustness": robustness,
            "Diversity": diversity,
            "Conv. Speed": convergence_speed,
            "SR": success_rate
        }
        metrics_file = os.path.join(func_dir, 'metrics.csv')
        pd.DataFrame([result]).to_csv(metrics_file, index=False)

        temp_result_file = os.path.join(temp_dir, f"temp_results_{name}.csv")
        pd.DataFrame([result]).to_csv(temp_result_file, index=False)

        results.append(result)

        with open(temp_fitness_file, 'wb') as f:
            pickle.dump(function_results, f)

        print(f"{name} - Best: {np.min(best_fits):.6f}, Mean: {mean_fit:.6f}, SR: {success_rate:.2f}")

    df = pd.DataFrame(results)
    df.to_csv(results_filename, index=False)
    print(f"\nResults saved to '{results_filename}'")

    wilcoxon_results = wilcoxon_test(function_results)
    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    wilcoxon_df.to_csv(wilcoxon_filename, index=False)
    print(f"Wilcoxon test results saved to '{wilcoxon_filename}'")

    print("\nSignificant Wilcoxon Test Results (p < 0.05):")
    for result in wilcoxon_results:
        if result["Significant (p<0.05)"]:
            print(f"{result['Function 1']} vs {result['Function 2']}: p-value = {result['p-value']:.6f}")

    for name in FUNCTIONS.keys():
        temp_result_file = os.path.join(temp_dir, f"temp_results_{name}.csv")
        if os.path.exists(temp_result_file):
            os.remove(temp_result_file)
    if os.path.exists(temp_fitness_file):
        os.remove(temp_fitness_file)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)

    return results

# --- Execute ---
if __name__ == "__main__":
    config = Config()
    results = run_all_functions(config)