# """CPU structural-importance computation shared by training and preprocessing.

# All measures use the same unweighted, undirected simple graph, including
# isolated entities and excluding self-loops. No alignment labels are used.
# """
# import hashlib
# import json
# import os
# from pathlib import Path
# import platform
# import time

# import networkx as nx
# import numpy as np

# SCHEMA_VERSION = 1
# METHODS = ("betweenness", "degree", "pagerank", "kshell", "none")


# def array_digest(array):
#     array = np.ascontiguousarray(array)
#     digest = hashlib.sha256()
#     digest.update(str(array.dtype).encode())
#     digest.update(str(array.shape).encode())
#     digest.update(memoryview(array).cast("B"))
#     return digest.hexdigest()


# def build_graph(num_nodes, edges):
#     if num_nodes < 1:
#         raise ValueError("The graph must contain at least one entity.")
#     edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
#     if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
#         raise ValueError("Graph endpoints do not match the entity ID range.")
#     edges = np.sort(edges, axis=1)
#     edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
#     graph = nx.Graph()
#     graph.add_nodes_from(range(num_nodes))
#     graph.add_edges_from(edges.tolist())
#     digest = hashlib.sha256()
#     digest.update(str(num_nodes).encode())
#     digest.update(edges.astype("<i8", copy=False).tobytes())
#     return graph, digest.hexdigest()


# def read_benchmark_graph(dataset_dir):
#     """Read the same entity IDs and triple endpoints used by get_adjr()."""
#     dataset_dir = Path(dataset_dir)
#     entity_ids = []
#     names = set()
#     for side in (1, 2):
#         with (dataset_dir / f"ent_ids_{side}").open(encoding="utf-8") as stream:
#             for line in stream:
#                 entity_id, name = line.rstrip("\n").split("\t", 1)
#                 entity_ids.append(int(entity_id))
#                 names.add(name)
#     if sorted(entity_ids) != list(range(len(entity_ids))) or len(names) != len(entity_ids):
#         raise ValueError("Expected unique entity names and contiguous joint entity IDs.")
#     edges = []
#     for side in (1, 2):
#         with (dataset_dir / f"triples_{side}").open(encoding="utf-8") as stream:
#             for line in stream:
#                 head, _, tail = map(int, line.rstrip("\n").split("\t"))
#                 edges.append((head, tail))
#     return len(entity_ids), np.asarray(edges, dtype=np.int64).reshape(-1, 2)


# def normalize_scores(raw_scores):
#     scores = np.asarray(raw_scores, dtype=np.float32)
#     if not np.isfinite(scores).all():
#         raise ValueError("Centrality computation returned nonfinite scores.")
#     lower, upper = scores.min(), scores.max()
#     if upper > lower:
#         scores = (scores - lower) / (upper - lower + 1e-9)
#     else:
#         scores = np.ones_like(scores)
#     return scores + np.float32(1e-3)


# def compute_scores(graph, method, sample_size=1000, seed=42,
#                    pagerank_alpha=0.85, pagerank_tol=1e-8, pagerank_max_iter=1000):
#     if method not in METHODS:
#         raise ValueError(f"Unknown importance measure: {method}")
#     if sample_size < 1 or not 0 < pagerank_alpha < 1 or pagerank_tol <= 0 or pagerank_max_iter < 1:
#         raise ValueError("Invalid centrality parameters.")
#     if method == "betweenness":
#         values = nx.betweenness_centrality(
#             graph, k=min(sample_size, len(graph)), normalized=True, weight=None, seed=seed)
#     elif method == "degree":
#         values = nx.degree_centrality(graph)
#     elif method == "pagerank":
#         values = nx.pagerank(graph, alpha=pagerank_alpha, tol=pagerank_tol,
#                              max_iter=pagerank_max_iter, weight=None)
#     elif method == "kshell":
#         values = nx.core_number(graph)
#     else:
#         values = dict.fromkeys(graph, 1.0)
#     raw = np.asarray([values[node] for node in range(len(graph))], dtype=np.float32)
#     return normalize_scores(raw)


# def load_or_compute_importance(num_nodes, edges, cache_dir, method="betweenness",
#                                sample_size=1000, seed=42, pagerank_alpha=0.85,
#                                pagerank_tol=1e-8, pagerank_max_iter=1000, force=False):
#     started = time.perf_counter()
#     graph, graph_hash = build_graph(num_nodes, edges)
#     graph_seconds = time.perf_counter() - started
#     parameters = {"schema_version": SCHEMA_VERSION, "method": method,
#                   "graph_sha256": graph_hash, "num_nodes": num_nodes,
#                   "num_edges": graph.number_of_edges(), "networkx": nx.__version__,
#                   "normalization": "float32_minmax_eps1e-9_plus1e-3",
#                   "graph_type": "unweighted_undirected_simple_no_self_loops"}
#     if method == "betweenness":
#         parameters.update(sample_size=min(sample_size, num_nodes), sampling_seed=seed)
#     elif method == "pagerank":
#         parameters.update(alpha=pagerank_alpha, tol=pagerank_tol, max_iter=pagerank_max_iter)
#     if method not in METHODS:
#         raise ValueError(f"Unknown importance measure: {method}")
#     key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:20]
#     cache_path = Path(cache_dir) / f"importance_v{SCHEMA_VERSION}_{method}_{key}.npz"
#     cache_hit = cache_path.is_file() and not force and method != "none"
#     operation_started = time.perf_counter()
#     if cache_hit:
#         with np.load(cache_path, allow_pickle=False) as data:
#             scores = data["scores"].copy()
#             stored = json.loads(str(data["metadata"].item()))
#         if stored["parameters"] != parameters:
#             raise ValueError(f"Importance cache metadata mismatch: {cache_path}")
#         if scores.shape != (num_nodes,) or scores.dtype != np.float32 or not np.isfinite(scores).all():
#             raise ValueError(f"Invalid importance cache scores: {cache_path}")
#         if stored["scores_sha256"] != array_digest(scores):
#             raise ValueError(f"Importance cache checksum mismatch: {cache_path}")
#         computation_seconds = stored["computation_seconds"]
#         computed_environment = stored["computed_environment"]
#     else:
#         scores = compute_scores(graph, method, sample_size, seed, pagerank_alpha,
#                                 pagerank_tol, pagerank_max_iter)
#         computation_seconds = time.perf_counter() - operation_started
#         computed_environment = {"python": platform.python_version(),
#                                 "platform": platform.platform(),
#                                 "processor": platform.processor(),
#                                 "logical_cpus": os.cpu_count(), "numpy": np.__version__,
#                                 "networkx": nx.__version__}
#         stored = {"parameters": parameters, "computation_seconds": computation_seconds,
#                   "scores_sha256": array_digest(scores),
#                   "computed_environment": computed_environment}
#         if method != "none":
#             cache_path.parent.mkdir(parents=True, exist_ok=True)
#             temporary = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
#             with temporary.open("wb") as stream:
#                 np.savez_compressed(stream, scores=scores,
#                                     metadata=np.asarray(json.dumps(stored, sort_keys=True)))
#             os.replace(temporary, cache_path)
#     report = {**stored, "cache_hit": cache_hit,
#               "cache_path": str(cache_path) if method != "none" else None,
#               "graph_build_and_hash_seconds": graph_seconds,
#               "load_or_compute_seconds": time.perf_counter() - operation_started,
#               "total_current_call_seconds": time.perf_counter() - started}
#     return scores, report
