


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
import uuid

# --- Configuration ---
class Config:
    def __init__(self):
        self.DIM = 30
        self.NUM_NODES = 300
        self.ITERATIONS = 3500
        self.NUM_RUNS = 20
        self.K = 5
        self.P_BASE = 0.1
        self.ALPHA_INIT = 0.5
        self.BETA_INIT = 0.5
        self.GAMMA = 0.2
        self.DELTA = 0.2
        self.MUTATION_RATE_INIT = 0.1
        self.MUTATION_STRENGTH_BASE = 0.05
        self.MUTATION_STRENGTH_MIN = 0.01
        self.SYNC_WEIGHT_INIT = 0.03
        self.SUCCESS_THRESHOLD = 1e-8
        self.TRACKED_DIM = 0
        self.LOTUS_R0 = 2.0
        self.LOTUS_BETA_DROPS = 3
        self.LOTUS_PIT_CONST = 40
        self.LOTUS_LOCAL_FREQ = 10
        self.SEED_SET_SIZE = int(0.1 * self.NUM_NODES)

# --- Benchmark Functions ---
class BenchmarkFunctions:
    def __init__(self):
        self.functions = {
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
                         [-50,50], 0.0),
            'Foxholes': (self.foxholes, [-65.536, 65.536], 0.998),
            'Kowalik': (self.kowalik, [-5, 5], 0.0003075),
            'Camel-Back': (self.camel_back, [-5, 5], -1.0316),
            'Branin': (self.branin, [-5, 5], 0.398),
            'Goldstein-Price': (self.goldstein_price, [-2, 2], 3.0),
            'Hartman': (self.hartman3, [0, 1], -3.86),
            'Shekel1': (self.hartman6, [0, 1], -3.322),
            'Shekel2': (self.shekel5, [0, 10], -10.1532),
            'Shekel3': (self.shekel7, [0, 10], -10.4028),
            'Shekel4': (self.shekel10, [0, 10], -10.5363)
        }

    def foxholes(self, x):
        x = x[:2]
        a = np.array([[4.0]*25, np.linspace(0, 12, 25)])
        denom = 1/500.0
        for j in range(25):
            sum_term = (x[0] - a[0, j])**6 + (x[1] - a[1, j])**6
            denom += 1.0 / (j + 1 + sum_term)
        return 1.0 / denom

    def kowalik(self, x):
        x = x[:4]
        a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844,
                      0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
        b = np.array([4, 2, 1, 0.5, 0.25,
                      0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625])
        s = 0.0
        for i in range(11):
            s += (a[i] - (x[0]*(b[i]**2 + b[i]*x[1]) / (b[i]**2 + b[i]*x[2] + x[3]*x[2])))**2
        return s

    def camel_back(self, x):
        x = x[:2]
        return 4*x[0]**2 - 2.1*x[0]**4 + (1/3)*x[0]**6 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4

    def branin(self, x):
        x = x[:2]
        a = 1.0
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8 * np.pi)
        return (x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s

    def goldstein_price(self, x):
        x = x[:2]
        term1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[1] - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
        term2 = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[1] + 48*x[0] - 36*x[0]*x[1] + 27*x[1]**2)
        return term1 * term2

    def hartman3(self, x):
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

    def hartman6(self, x):
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

    def shekel5(self, x):
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

    def shekel7(self, x):
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

    def shekel10(self, x):
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

# --- Graph Management ---
class GraphManager:
    def __init__(self, config):
        self.config = config

    def initialize_population(self, dim, bounds, k, p, graph_type='watts_strogatz'):
        if graph_type == 'watts_strogatz':
            G = nx.watts_strogatz_graph(self.config.NUM_NODES, k=k, p=p)
        elif graph_type == 'erdos_renyi':
            G = nx.erdos_renyi_graph(self.config.NUM_NODES, p=p)
        elif graph_type == 'barabasi_albert':
            G = nx.barabasi_albert_graph(self.config.NUM_NODES, k // 2)
        else:
            raise ValueError(f"Unsupported graph_type: {graph_type}")

        for node in G.nodes:
            G.nodes[node]['position'] = np.random.uniform(bounds[0], bounds[1], dim)
            G.nodes[node]['fitness'] = None
        return G

    def evaluate_fitness(self, G, func):
        for node in G.nodes:
            pos = G.nodes[node]['position']
            G.nodes[node]['fitness'] = func(pos)

    def compute_population_entropy(self, G):
        if not G.nodes or G.number_of_edges() == 0:
            return 0.0
        deg_vals = np.array([d for _, d in G.degree()])
        if deg_vals.sum() == 0:
            return 0.0
        p = deg_vals / deg_vals.sum()
        return entropy(p + 1e-12)

    def pick_topk_nodes(self, G, k):
        if not G.nodes:
            return []
        try:
            pagerank = nx.pagerank(G, alpha=0.85)
        except nx.NetworkXError:
            return list(G.nodes)[:min(k, len(G.nodes))]
        sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:k]]

# --- Adaptive Graph ---
class AdaptiveGraph:
    def __init__(self, config):
        self.config = config
        self.best_fitness_archive = []

    def select_seed_set(self, G):
        if not G.nodes:
            raise ValueError("Graph is empty in select_seed_set.")
        fitness_values = np.array([G.nodes[n]['fitness'] for n in G.nodes])
        if np.any(fitness_values == None):
            raise ValueError("Fitness values contain None in select_seed_set.")
        sorted_indices = np.argsort(fitness_values)[:self.config.SEED_SET_SIZE]
        seed_nodes = np.random.choice(sorted_indices, size=self.config.SEED_SET_SIZE, replace=False)
        return [list(G.nodes)[i] for i in seed_nodes]

    def adapt_graph(self, G, iteration, func, bounds):
        if not G.nodes:
            raise ValueError("Graph is empty in adapt_graph.")
        seed_nodes = self.select_seed_set(G)
        best_fitness = min(G.nodes[n]['fitness'] for n in G.nodes)
        self.best_fitness_archive.append(best_fitness)

        # Rewire edges to connect seed nodes
        G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d.get('tmp')])
        for i in range(len(seed_nodes)):
            for j in range(i + 1, len(seed_nodes)):
                if np.random.rand() < 0.3:
                    G.add_edge(seed_nodes[i], seed_nodes[j], tmp=True)

        # Inject random edges to prevent local clustering
        if iteration % 10 == 0:
            nodes = list(G.nodes)
            num_random_edges = int(0.05 * self.config.NUM_NODES)
            for _ in range(num_random_edges):
                u, v = np.random.choice(nodes, size=2, replace=False)
                if not G.has_edge(u, v):
                    G.add_edge(u, v, tmp=True)

        # Remove low-fitness nodes and add new ones
        if iteration % 25 == 0:
            fitness_values = np.array([G.nodes[n]['fitness'] for n in G.nodes])
            nodes_to_remove = np.argsort(fitness_values)[-int(0.05 * self.config.NUM_NODES):]
            nodes_to_remove = [list(G.nodes)[i] for i in nodes_to_remove]
            G.remove_nodes_from(nodes_to_remove)
            max_node_id = max(G.nodes) if G.nodes else -1
            new_nodes = list(range(max_node_id + 1, max_node_id + 1 + len(nodes_to_remove)))
            G.add_nodes_from(new_nodes)
            for node in new_nodes:
                G.nodes[node]['position'] = np.random.uniform(bounds[0], bounds[1], self.config.DIM)
                G.nodes[node]['fitness'] = func(G.nodes[node]['position'])
                if seed_nodes:
                    G.add_edge(node, np.random.choice(seed_nodes), tmp=True)
                else:
                    existing_nodes = list(G.nodes - {node})
                    if existing_nodes:
                        G.add_edge(node, np.random.choice(existing_nodes), tmp=True)
            self.config.NUM_NODES = len(G.nodes)

            # Ensure minimum connectivity
            if G.number_of_edges() < max(2, int(0.1 * len(G.nodes))):
                nodes = list(G.nodes)
                for i in range(len(nodes)):
                    for j in range(i + 1, min(i + 3, len(nodes))):
                        if np.random.rand() < 0.5 and not G.has_edge(nodes[i], nodes[j]):
                            G.add_edge(nodes[i], nodes[j], tmp=True)

            G.graph.pop('eig', None)
            G.graph.pop('pr_cache', None)

# --- Optimizer ---
class Optimizer:
    def __init__(self, config):
        self.config = config
        self.graph_manager = GraphManager(config)
        self.adaptive_graph = AdaptiveGraph(config)

    def composite_centrality(self, G, t):
        deg = nx.degree_centrality(G)
        close = nx.closeness_centrality(G)
        try:
            eig = nx.eigenvector_centrality(G, max_iter=1500)
            G.graph['eig'] = eig
        except nx.NetworkXError:
            eig = {n: 0.0 for n in G.nodes}
        return {n: 0.5*(1-t)*deg[n] + 0.3*(1-t)*close[n] + (0.2+0.3*t)*eig[n] for n in G}

    def lotus_shrink_step(self, G, elite_pos, bounds, iteration, max_iter):
        R = self.config.LOTUS_R0 * np.exp(- (4 * iteration / max_iter) ** 2)
        top_k_nodes = self.graph_manager.pick_topk_nodes(G, k=int(0.1 * len(G)))
        for node in top_k_nodes:
            pos = G.nodes[node]['position']
            new_pos = pos + R * (elite_pos - pos)
            G.nodes[node]['position'] = np.clip(new_pos, bounds[0], bounds[1])

    def lotus_reinforcement(self, G, func, bounds):
        fits = np.array([G.nodes[n]['fitness'] for n in G.nodes])
        if np.any(fits == None):
            raise ValueError("Fitness values contain None in lotus_reinforcement.")
        c_max, c_min = fits.max(), fits.min()
        caps = (np.abs(fits - c_max) / (np.abs(c_min - c_max) + 1e-9)) * self.config.LOTUS_PIT_CONST

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
                    v = np.random.randn(self.config.DIM) * 0.01
                droplets.append((node, v))

        for node, v in droplets:
            pos = G.nodes[node]['position'].copy()
            node_comm = partition[node]
            for _ in range(self.config.LOTUS_BETA_DROPS):
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

    def diffuse(self, G, gbest_pos, elite_pos, bounds, iteration, max_iterations, func, enable_mutation=True):
        if not G.nodes:
            raise ValueError("Graph is empty in diffuse.")
        t = iteration / max_iterations
        if iteration == 0 or iteration % 50 == 0:
            try:
                G.graph['eig'] = nx.eigenvector_centrality(G, max_iter=1500)
            except nx.NetworkXError:
                G.graph['eig'] = {n: 0.0 for n in G.nodes}
        cent = self.composite_centrality(G, t)
        fitness_values = np.array([G.nodes[n]['fitness'] for n in G.nodes])
        if np.any(fitness_values == None):
            raise ValueError("Fitness values contain None in diffuse.")
        log_influence = np.log1p(np.abs(fitness_values) + 1e-6)
        influence = 1 - (log_influence / (np.max(log_influence) + 1e-6))

        alpha_t = self.config.ALPHA_INIT * np.exp(-5 * t) + 0.1
        beta_t = self.config.BETA_INIT * np.exp(-5 * t) + 0.1
        gamma_t = self.config.GAMMA * t
        delta_t = self.config.DELTA * t
        sync_weight_t = self.config.SYNC_WEIGHT_INIT * (1 - t)

        dimension_mean = np.mean([G.nodes[n]['position'] for n in G.nodes], axis=0)
        new_positions = {}

        if iteration % 25 == 0:
            part = community_louvain.best_partition(G)
            comm_best = {}
            for node, comm in part.items():
                if comm not in comm_best or G.nodes[node]['fitness'] < G.nodes[comm_best[comm]]['fitness']:
                    comm_best[comm] = node
            for c, src in comm_best.items():
                dst = np.random.choice([comm_best[d] for d in comm_best if d != c])
                G.add_edge(src, dst, tmp=True)

        pr_cache = G.graph.get('pr_cache')
        elite_idx = min(G.nodes, key=lambda n: G.nodes[n]['fitness'])
        if gamma_t + delta_t > alpha_t + beta_t and (pr_cache is None or iteration % 100 == 0):
            try:
                pr_cache = nx.pagerank(G, alpha=0.85, personalization={elite_idx: 1})
                G.graph['pr_cache'] = pr_cache
            except nx.NetworkXError:
                pr_cache = {n: 1.0 / len(G.nodes) for n in G.nodes}
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

        G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d.get('tmp')])

        entropy_val = self.graph_manager.compute_population_entropy(G)
        normalized_entropy = entropy_val / np.log(self.config.NUM_NODES) if self.config.NUM_NODES > 1 else 0.0

        if enable_mutation:
            mutation_rate_t = self.config.MUTATION_RATE_INIT * (1 - t) + 0.01
            mutation_strength = self.config.MUTATION_STRENGTH_BASE * (1 - t)
            median_fitness = np.median(fitness_values)

            if normalized_entropy < 0.4:
                mutation_rate_t += 0.15 * (0.4 - normalized_entropy)

            for node in G.nodes:
                pos = new_positions[node]
                if G.nodes[node]['fitness'] > median_fitness and np.random.rand() < mutation_rate_t:
                    sigma_i = mutation_strength * (1 - cent[node]**1.5) + self.config.MUTATION_STRENGTH_MIN
                    perturb = np.random.uniform(-sigma_i * (bounds[1] - bounds[0]), sigma_i * (bounds[1] - bounds[0]), len(pos))
                    pos = np.clip(pos + perturb, bounds[0], bounds[1])
                if iteration % 10 == 0 and np.random.rand() < 0.05:
                    pos += np.random.uniform(-1, 1, len(pos)) * 0.5
                G.nodes[node]['position'] = pos
        else:
            for node, pos in new_positions.items():
                G.nodes[node]['position'] = pos

        if iteration % 75 == 0 and normalized_entropy < 0.5 and G.number_of_edges() >= 2:
            try:
                nx.double_edge_swap(G, nswap=int(0.1 * G.number_of_edges()), max_tries=1000)
            except nx.NetworkXError:
                pass
            G.graph.pop('eig', None)
            G.graph.pop('pr_cache', None)

        self.adaptive_graph.adapt_graph(G, iteration, func, bounds)

        return G

    def social_optimize(self, func_name, functions, enable_mutation=True, graph_type='watts_strogatz'):
        func, bounds, optimum = functions.functions[func_name]
        G = self.graph_manager.initialize_population(self.config.DIM, bounds, self.config.K, self.config.P_BASE, graph_type=graph_type)

        search_history = []
        first_agent_trajectory = []
        avg_fitness_history = []
        convergence_history = []
        self.adaptive_graph.best_fitness_archive = []

        self.graph_manager.evaluate_fitness(G, func)
        fitness_values = [G.nodes[n]['fitness'] for n in G.nodes]
        gbest_idx = np.argmin(fitness_values)
        gbest_pos = G.nodes[gbest_idx]['position'].copy()
        gbest_fitness = fitness_values[gbest_idx]
        elite_pos = gbest_pos.copy()
        elite_fitness = gbest_fitness

        for iteration in range(self.config.ITERATIONS):
            self.graph_manager.evaluate_fitness(G, func)
            fitness_values = [G.nodes[n]['fitness'] for n in G.nodes]

            positions = [G.nodes[n]['position'][self.config.TRACKED_DIM] for n in G.nodes]
            search_history.append(positions)
            first_node = next(iter(G.nodes)) if G.nodes else None
            if first_node is not None:
                first_agent_trajectory.append(G.nodes[first_node]['position'][self.config.TRACKED_DIM])
            else:
                first_agent_trajectory.append(np.nan)
            avg_fitness = np.mean(fitness_values)
            avg_fitness_history.append(avg_fitness)
            convergence_history.append(elite_fitness)

            node_ids = list(G.nodes)
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < gbest_fitness:
                gbest_pos = G.nodes[node_ids[min_idx]]['position'].copy()
                gbest_fitness = fitness_values[min_idx]

            if gbest_fitness < elite_fitness:
                elite_pos = gbest_pos.copy()
                elite_fitness = gbest_fitness

            G = self.diffuse(G, gbest_pos, elite_pos, bounds, iteration, self.config.ITERATIONS, func, enable_mutation)
            self.lotus_shrink_step(G, elite_pos, bounds, iteration, self.config.ITERATIONS)
            if iteration % self.config.LOTUS_LOCAL_FREQ == 0:
                self.lotus_reinforcement(G, func, bounds)

        return {
            'search_history': search_history,
            'first_agent_trajectory': first_agent_trajectory,
            'avg_fitness_history': avg_fitness_history,
            'convergence_history': convergence_history,
            'best_fitness_archive': self.adaptive_graph.best_fitness_archive
        }, elite_pos, elite_fitness, G

# --- Result Processor ---
class ResultProcessor:
    def __init__(self, config):
        self.config = config

    def wilcoxon_test(self, function_results):
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

    def run_all_functions(self, functions, optimizer, function_names=['Sphere']):
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

        for name in function_names:
            if name not in functions.functions:
                print(f"Function {name} not found in benchmark functions. Skipping.")
                continue
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
            for run in range(start_run, self.config.NUM_RUNS):
                print(f"  Run {run+1}/{self.config.NUM_RUNS}")
                data, _, best_fit, G = optimizer.social_optimize(name, functions)
                best_fits.append(best_fit)
                all_histories.append(data['convergence_history'])
                final_populations.append(G)

                if run == 0:
                    search_df = pd.DataFrame(
                        data['search_history'],
                        columns=[f'Agent_{i}' for i in range(self.config.NUM_NODES)],
                        index=[f'Iteration_{i}' for i in range(self.config.ITERATIONS)]
                    )
                    search_df.to_csv(os.path.join(func_dir, 'search_history.csv'))

                    trajectory_df = pd.DataFrame({
                        'Iteration': range(self.config.ITERATIONS),
                        'Position': data['first_agent_trajectory']
                    })
                    trajectory_df.to_csv(os.path.join(func_dir, 'first_agent_trajectory.csv'), index=False)

                    avg_fitness_df = pd.DataFrame({
                        'Iteration': range(self.config.ITERATIONS),
                        'Average_Fitness': data['avg_fitness_history']
                    })
                    avg_fitness_df.to_csv(os.path.join(func_dir, 'avg_fitness.csv'), index=False)

                    convergence_df = pd.DataFrame({
                        'Iteration': range(self.config.ITERATIONS),
                        'Best_Fitness': data['convergence_history']
                    })
                    convergence_df.to_csv(os.path.join(func_dir, 'convergence.csv'), index=False)

                    best_fitness_df = pd.DataFrame({
                        'Iteration': range(self.config.ITERATIONS),
                        'Best_Fitness': data['best_fitness_archive']
                    })
                    best_fitness_df.to_csv(os.path.join(func_dir, 'best_fitness_archive.csv'), index=False)

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
                                          for i in range(self.config.DIM)]) for G in final_populations])
            convergence_speed = next((i for i, fit in enumerate(all_histories[0]) if fit < mean_fit + std_fit), self.config.ITERATIONS)
            success_rate = np.mean([1 if abs(bf - functions.functions[name][2]) < self.config.SUCCESS_THRESHOLD else 0 for bf in best_fits])

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

        if len(function_results) > 1:
            wilcoxon_results = self.wilcoxon_test(function_results)
            wilcoxon_df = pd.DataFrame(wilcoxon_results)
            wilcoxon_df.to_csv(wilcoxon_filename, index=False)
            print(f"Wilcoxon test results saved to '{wilcoxon_filename}'")

            print("\nSignificant Wilcoxon Test Results (p < 0.05):")
            for result in wilcoxon_results:
                if result["Significant (p<0.05)"]:
                    print(f"{result['Function 1']} vs {result['Function 2']}: p-value = {result['p-value']:.6f}")
        else:
            print("\nWilcoxon test skipped (only one function processed).")

        for name in function_names:
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
    functions = BenchmarkFunctions()
    optimizer = Optimizer(config)
    processor = ResultProcessor(config)
   # results = processor.run_all_functions(functions, optimizer, function_names=['Sphere'])
    results = processor.run_all_functions(functions, optimizer, function_names=list(functions.functions.keys()))