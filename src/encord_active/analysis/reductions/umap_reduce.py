import dataclasses
from base64 import b85decode, b85encode
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from numpy.random import RandomState
from pydantic import BaseModel
from pynndescent import NNDescent
from pynndescent.distances import fast_distance_alternatives
from pynndescent.rp_trees import FlatTree
from scipy.sparse import csr_matrix
from umap import UMAP, distances, sparse


class UMAPDistanceFunc(Enum):
    METRIC = "metric"
    SPARSE_NAMED_DISTANCES = "sparse_named_distances"
    NAMED_DISTANCES = "named_distances"


def _func_to_distance_func(func: Any, metric: str) -> UMAPDistanceFunc:
    if func == metric:
        return UMAPDistanceFunc.METRIC
    elif func == sparse.sparse_named_distances[metric]:
        return UMAPDistanceFunc.SPARSE_NAMED_DISTANCES
    elif func == distances.named_distances[metric]:
        return UMAPDistanceFunc.NAMED_DISTANCES
    else:
        raise ValueError("Unknown distance func")


def _metric_to_distance_func(func: UMAPDistanceFunc, metric: str) -> Any:
    if func == UMAPDistanceFunc.METRIC:
        return metric
    elif func == UMAPDistanceFunc.SPARSE_NAMED_DISTANCES:
        return sparse.sparse_named_distances[metric]
    elif func == UMAPDistanceFunc.NAMED_DISTANCES:
        return distances.named_distances[metric]
    else:
        raise ValueError("Unknown distance func")


class ShapedBytes(BaseModel):
    data: bytes
    shape: Tuple[int, ...]


class FlatTreeSerialized(BaseModel):
    hyperplanes: ShapedBytes  # float32
    offsets: ShapedBytes  # float32
    children: ShapedBytes  # int32
    indices: ShapedBytes  # int32
    leaf_size: int


class PyNNDescentSerialized(BaseModel):
    angular_trees: bool
    n_jobs: int
    is_sparse: bool
    tree_init: bool
    n_neighbors: int
    search_forest_trees: List[FlatTreeSerialized]

    # FIXME: may be the same value as umap - if so do not set.
    raw_data: ShapedBytes
    vertex_order: ShapedBytes  # FIXME: default self._search_forest[0].indices

    # FIXME: change to better format!
    search_graph_array: ShapedBytes
    search_graph_shape: Tuple[int, ...]


class UMAPSerialized(BaseModel):
    # Constructor arguments
    n_neighbors: int = 15
    n_components: int = 2
    metric: Literal["euclidean"] = "euclidean"
    metric_kwds: Optional[Dict[str, Union[str, int, float]]] = None
    output_metric: Literal["euclidean"] = "euclidean"
    output_metric_kwds: None = None
    n_epochs: Optional[int] = None
    learning_rate: float = 1.0
    init: Literal["spectral"] = "spectral"
    min_dist: float = 0.1
    spread: float = 1.0
    low_memory: bool = True
    n_jobs: int = -1
    set_op_mix_ratio: float = 1.0
    local_connectivity: float = 1.0
    repulsion_strength: float = 1.0
    negative_sample_rate: int = 5
    transform_queue_size: float = 4.0
    a: Optional[float] = None
    b: Optional[float] = None
    random_state: Literal[0] = 0  # Explicit
    angular_rp_forest: bool = False
    target_n_neighbors: int = -1
    target_metric: Literal["categorical"] = "categorical"
    target_metric_kwds: None = None
    target_weight: float = 0.5
    transform_seed: int = 42
    transform_mode: Literal["embedding"] = "embedding"
    force_approximation_algorithm = False
    verbose: Literal[False] = False
    tqdm_kwds: Dict[str, Union[bool, str]] = {
        "desc": "Epochs completed",
        "bar_format": "{desc}: {percentage:3.0f}%| {bar} {n_fmt}/{total_fmt} [{elapsed}]",
        "disable": True,
    }
    unique: bool = False
    densmap: bool = False
    dens_lambda: float = 2.0
    dens_frac: float = 0.3
    dens_var_shift: float = 0.1
    output_dens: bool = False
    disconnection_distance: Optional[float] = None
    precomputed_knn: Tuple[None, None, None] = (None, None, None)
    # Internal state
    embedding: ShapedBytes
    i_metric_kwds: Dict[str, Union[str, int, float]] = {}
    i_raw_data: ShapedBytes
    i_small_data: bool = False
    i_sparse_data: bool = False
    i_input_distance_func: UMAPDistanceFunc = UMAPDistanceFunc.NAMED_DISTANCES
    i_n_neighbors: int = 15
    i_a: float
    i_b: float
    i_disconnection_distance: Optional[float] = float("inf")
    i_initial_alpha: float = 1.0
    i_knn_search_index: Optional[PyNNDescentSerialized] = None
    i_knn_dists: None = None


def serialize_np_array(arr: np.ndarray, t: Type[np.number] = np.float32) -> ShapedBytes:
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Wrong type: attempting to serialize np.ndarray: {type(arr)}")
    if arr.dtype != t:
        raise ValueError(f"Unsupported np serialize: {arr.dtype}")
    return ShapedBytes(data=b85encode(arr.tobytes(), pad=False), shape=arr.shape)


def serialize_pynn_descent(nn_descent: NNDescent, metric: str) -> PyNNDescentSerialized:
    # Sanity checking
    if not isinstance(nn_descent.verbose, bool) or nn_descent.verbose:
        raise ValueError(f"Verbose umap: {nn_descent.verbose}")

    print(f"Debug keys: {nn_descent.__dict__.keys()}")
    search_graph = getattr(nn_descent, "_search_graph")
    if isinstance(search_graph, csr_matrix):
        print(search_graph.__dict__)
        search_graph_array = search_graph.toarray()  # FIXME: serialize better!!
        search_graph_shape = search_graph.shape
        if search_graph_array.dtype != np.uint8:
            raise ValueError("Unsupported pynn descent index")
    else:
        raise ValueError("Unsupported csr matrix type")

    search_forest = getattr(nn_descent, "_search_forest")
    search_forest_trees = [
        FlatTreeSerialized(
            hyperplanes=serialize_np_array(hyperplanes),
            offsets=serialize_np_array(offsets),
            children=serialize_np_array(children, np.int32),
            indices=serialize_np_array(indices, np.int32),
            leaf_size=leaf_size,
        )
        for hyperplanes, offsets, children, indices, leaf_size in search_forest
    ]

    if getattr(nn_descent, "_distance_func") != fast_distance_alternatives[metric]["dist"]:
        raise ValueError("Unsupported nn_descent _distance_func")
    if getattr(nn_descent, "_distance_correction") != fast_distance_alternatives[metric]["correction"]:
        raise ValueError("Unsupported nn_descent _distance_correction")

    return PyNNDescentSerialized(
        angular_trees=getattr(nn_descent, "_angular_trees"),
        n_jobs=nn_descent.n_jobs,
        is_sparse=getattr(nn_descent, "_is_sparse"),
        tree_init=nn_descent.tree_init,
        n_neighbors=nn_descent.n_neighbors,
        search_forest_trees=search_forest_trees,
        search_graph_array=ShapedBytes(
            data=b85encode(search_graph_array.tobytes(), pad=True), shape=search_graph_array.shape
        ),
        search_graph_shape=search_graph_shape,
        raw_data=serialize_np_array(getattr(nn_descent, "_raw_data")),
        vertex_order=serialize_np_array(getattr(nn_descent, "_vertex_order"), np.int32),
    )


def serialize_umap(umap: UMAP) -> UMAPSerialized:
    # Sanity checking
    if not isinstance(umap.verbose, bool) or umap.verbose:
        raise ValueError(f"Verbose umap: {umap.verbose}")

    return UMAPSerialized(
        n_neighbors=umap.n_neighbors,
        n_components=umap.n_components,
        metric=umap.metric,
        metric_kwds=umap.metric_kwds,
        output_metric=umap.output_metric,
        output_metric_kwds=umap.output_metric_kwds,
        n_epochs=umap.n_epochs,
        learning_rate=umap.learning_rate,
        init=umap.init,
        min_dist=umap.min_dist,
        spread=umap.spread,
        low_memory=umap.low_memory,  # FIXME: execution only?
        n_jobs=umap.n_jobs,  # FIXME: execution only?
        set_op_mix_ratio=umap.set_op_mix_ratio,
        local_connectivity=umap.local_connectivity,
        repulsion_strength=umap.repulsion_strength,
        negative_sample_rate=umap.negative_sample_rate,
        transform_queue_size=umap.transform_queue_size,
        a=umap.a,
        b=umap.b,
        random_state=umap.random_state,
        angular_rp_forest=umap.angular_rp_forest,
        target_n_neighbors=umap.target_n_neighbors,
        target_metric=umap.target_metric,
        target_metric_kwds=umap.target_metric_kwds,
        target_weight=umap.target_weight,
        transform_seed=umap.transform_seed,
        transform_mode=umap.transform_mode,
        force_approximation_algorithm=umap.force_approximation_algorithm,
        verbose=umap.verbose,
        tqdm_kwds=umap.tqdm_kwds,
        unique=umap.unique,
        densmap=umap.densmap,
        dens_lambda=umap.dens_lambda,
        dens_frac=umap.dens_frac,
        dens_var_shift=umap.dens_var_shift,
        output_dens=umap.output_dens,
        disconnection_distance=umap.disconnection_distance,
        precomputed_knn=umap.precomputed_knn,
        # Internal State
        embedding=serialize_np_array(umap.embedding_),  # FIXME: when is this needed?
        i_raw_data=serialize_np_array(getattr(umap, "_raw_data")),  # FIXME: when is this needed?
        i_small_data=getattr(umap, "_small_data"),
        i_sparse_data=getattr(umap, "_sparse_data"),
        i_input_distance_func=_func_to_distance_func(getattr(umap, "_input_distance_func"), umap.metric),
        i_metric_kwds=getattr(umap, "_metric_kwds"),
        i_n_neighbors=getattr(umap, "_n_neighbors"),
        i_disconnection_distance=getattr(umap, "_disconnection_distance"),
        i_a=getattr(umap, "_a"),
        i_b=getattr(umap, "_b"),
        i_initial_alpha=getattr(umap, "_initial_alpha"),
        i_knn_search_index=None
        if getattr(umap, "_knn_search_index", None) is None
        else serialize_pynn_descent(getattr(umap, "_knn_search_index"), umap.metric),
        i_knn_dists=getattr(umap, "knn_dists", None),
    )


@dataclasses.dataclass(frozen=True)
class _RawData:
    shape: Any


def deserialize_np_array(serialized: ShapedBytes, t: Type[np.number] = np.float32) -> np.ndarray:
    return np.frombuffer(b85decode(serialized.data), dtype=t).reshape(serialized.shape)


def deserialize_pynn_descent(serialized: PyNNDescentSerialized, metric: str) -> NNDescent:
    nn_descent = NNDescent.__new__(NNDescent)
    to_del = {attr_name for attr_name in nn_descent.__dict__.keys() if not attr_name.startswith("_")}
    for attr_name in to_del:
        delattr(nn_descent, attr_name)

    # Assign attributes.
    setattr(nn_descent, "_angular_trees", serialized.angular_trees)
    setattr(nn_descent, "_is_sparse", serialized.is_sparse)
    setattr(nn_descent, "verbose", False)
    setattr(nn_descent, "tree_init", serialized.tree_init)
    setattr(nn_descent, "n_neighbors", serialized.n_neighbors)

    setattr(
        nn_descent,
        "_search_graph",
        csr_matrix(
            np.frombuffer(b85decode(serialized.search_graph_array.data), dtype=np.uint8).reshape(
                serialized.search_graph_array.shape
            ),
            serialized.search_graph_shape,
        ),
    )

    search_forest = [
        FlatTree(
            deserialize_np_array(forest.hyperplanes),
            deserialize_np_array(forest.offsets),
            deserialize_np_array(forest.children, np.int32),
            deserialize_np_array(forest.indices, np.int32),
            forest.leaf_size,
        )
        for forest in serialized.search_forest_trees
    ]
    setattr(nn_descent, "_search_forest", search_forest)
    setattr(nn_descent, "parallel_batch_queries", False)  # FIXME: make deserialize input

    setattr(nn_descent, "_raw_data", deserialize_np_array(serialized.raw_data).copy())
    setattr(nn_descent, "_distance_func", fast_distance_alternatives[metric]["dist"])
    setattr(nn_descent, "_distance_correction", fast_distance_alternatives[metric]["correction"])

    visited = np.zeros((serialized.raw_data.shape[0] // 8) + 1, dtype=np.uint8, order="C")
    setattr(nn_descent, "_visited", visited)
    setattr(nn_descent, "_vertex_order", deserialize_np_array(serialized.vertex_order, np.int32))

    # Random state
    INT32_MIN = np.iinfo(np.int32).min + 1
    INT32_MAX = np.iinfo(np.int32).max - 1
    random_state = RandomState(0)
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    search_rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    setattr(nn_descent, "rng_state", rng_state)
    setattr(nn_descent, "search_rng_state", search_rng_state)

    return nn_descent


def deserialize_umap(serialized: UMAPSerialized) -> UMAP:
    # print(f"DEBUG: {serialized}")
    print(f"Debug map modes: {serialized.i_small_data}, {serialized.densmap}, {serialized.i_sparse_data}")
    # Create empty UMAP, remove all attributes.
    deserialized = UMAP(
        n_neighbors=serialized.n_neighbors,
        n_components=serialized.n_components,
        metric=serialized.metric,
        metric_kwds=serialized.metric_kwds,
        output_metric=serialized.output_metric,
        output_metric_kwds=serialized.output_metric_kwds,
        n_epochs=serialized.n_epochs,
        learning_rate=serialized.learning_rate,
        init=serialized.init,
        min_dist=serialized.min_dist,
        spread=serialized.spread,
        low_memory=serialized.low_memory,  # FIXME: execution only?
        n_jobs=serialized.n_jobs,  # FIXME: execution only?
        set_op_mix_ratio=serialized.set_op_mix_ratio,
        local_connectivity=serialized.local_connectivity,
        repulsion_strength=serialized.repulsion_strength,
        negative_sample_rate=serialized.negative_sample_rate,
        transform_queue_size=serialized.transform_queue_size,
        a=serialized.a,
        b=serialized.b,
        random_state=serialized.random_state,
        angular_rp_forest=serialized.angular_rp_forest,
        target_n_neighbors=serialized.target_n_neighbors,
        target_metric=serialized.target_metric,
        target_metric_kwds=serialized.target_metric_kwds,
        target_weight=serialized.target_weight,
        transform_seed=serialized.transform_seed,
        transform_mode=serialized.transform_mode,
        force_approximation_algorithm=serialized.force_approximation_algorithm,
        verbose=serialized.verbose,
        tqdm_kwds=serialized.tqdm_kwds,
        unique=serialized.unique,
        densmap=serialized.densmap,
        dens_lambda=serialized.dens_lambda,
        dens_frac=serialized.dens_frac,
        dens_var_shift=serialized.dens_var_shift,
        output_dens=serialized.output_dens,
        disconnection_distance=serialized.disconnection_distance,
        precomputed_knn=serialized.precomputed_knn,
    )

    # Start assigning attributes from the serialized state.

    # Return the umap object
    setattr(deserialized, "_raw_data", deserialize_np_array(serialized.i_raw_data))
    setattr(deserialized, "_input_hash", None)
    setattr(deserialized, "_small_data", serialized.i_small_data)
    setattr(deserialized, "_sparse_data", serialized.i_sparse_data)
    setattr(
        deserialized,
        "_input_distance_func",
        _metric_to_distance_func(serialized.i_input_distance_func, serialized.metric),
    )
    setattr(deserialized, "_metric_kwds", serialized.i_metric_kwds)
    setattr(deserialized, "_n_neighbors", serialized.i_n_neighbors)
    setattr(deserialized, "_disconnection_distance", serialized.i_disconnection_distance)
    setattr(deserialized, "embedding_", deserialize_np_array(serialized.embedding))
    setattr(deserialized, "_a", serialized.i_a)
    setattr(deserialized, "_b", serialized.i_b)
    setattr(deserialized, "_initial_alpha", serialized.i_initial_alpha)
    if serialized.i_knn_search_index is not None:
        setattr(
            deserialized,
            "_knn_search_index",
            deserialize_pynn_descent(serialized.i_knn_search_index, serialized.metric),
        )
    setattr(deserialized, "knn_dists", serialized.i_knn_dists)
    return deserialized


# FIXME: this is test code - the serialize / deserialize pipeline in this file should be considered
# experimental and untested.
#  FIXME: convert to proper testing pytest or similar
#  FIXME: get to pass where full reconstruction works (after call to fit reconstructs some internals).
#  FIXME: get pynn implementation to return exact-same result (or as close as possible - may be harder due
#   to needing to reset the random state used).
def _test():
    for n in [35, 3500, 35000, 95000]:
        print(f"\nRunning: {n}")
        test_embeddings = np.array([[float(j) / 4142.532 for j in range(i, i + 5)] for i in range(3, n, 4)])
        reducer = UMAP(random_state=0)
        _ignore = reducer.fit_transform(test_embeddings)
        reducer._input_hash = None
        test_reduced = reducer.transform(test_embeddings)

        serialized = serialize_umap(reducer)
        if n <= 3500:
            print(f"Serialized: {serialized.json(exclude_defaults=True)}")
        new_umap = deserialize_umap(serialized)
        match_reduced = new_umap.transform(test_embeddings)
        print(f"DEBUG Equivalence Max: {np.max(np.abs(test_reduced - match_reduced))}")
        print(f"DEBUG Equivalence Avg: {np.mean(np.abs(test_reduced - match_reduced))}")

        def _neq(a, b) -> bool:
            if type(a) != type(b):  # pylint: disable=unidiomatic-typecheck
                return True
            if isinstance(a, np.ndarray):
                if a.shape != b.shape:
                    return True
                if a.dtype != b.dtype:
                    return True
                return (a != b).any()
            try:
                return a != b
            except Exception:
                print(f"t: {type(a)}, {type(b)}")
                return True

        umap_differences = {
            k
            for k, v in reducer.__dict__.items()
            if hasattr(new_umap, k) and _neq(getattr(new_umap, k), v) and k != "_knn_search_index"
        }
        missing_umap_differences = {
            k for k, v in reducer.__dict__.items() if not hasattr(new_umap, k) and k != "_knn_search_index"
        }

        print(f"Differences (if attr not missing): {umap_differences}")
        print(f"Differences (missing): {missing_umap_differences}")
        reducer_knn = getattr(reducer, "_knn_search_index", NNDescent.__new__(NNDescent))
        new_umap_knn = getattr(new_umap, "_knn_search_index", NNDescent.__new__(NNDescent))
        if type(reducer_knn) != type(new_umap_knn):  # pylint: disable=unidiomatic-typecheck
            print("Different knn type")
        try:
            knn_differences = {
                k
                for k, v in reducer_knn.__dict__.items()
                if hasattr(new_umap_knn, k) and _neq(getattr(new_umap_knn, k), v)
            }
        except Exception:
            print("Exception in knn differences")
            knn_differences = {}
        missing_knn_differences = {k for k, v in reducer_knn.__dict__.items() if not hasattr(new_umap_knn, k)}
        print(f"Differences KNN (if attr not missing): {knn_differences}")
        print(f"Differences KNN (missing): {missing_knn_differences}")


if __name__ == "__main__":
    _test()
