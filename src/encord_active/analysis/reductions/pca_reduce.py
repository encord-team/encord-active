import numpy as np
from sklearn.decomposition import PCA


def serialize_pca(pca: PCA) -> bytes:
    mean = pca.mean_
    components = pca.components_
    variance = pca.explained_variance_
    n_features = pca.n_features_in_

    n_feature_bytes = n_features.to_bytes(length=8, byteorder="little", signed=True)
    if tuple(mean.shape) != (n_features,) and mean.dtype != np.float64:
        raise ValueError(f"PCA has invalid formatting: {mean.shape} / {mean.dtype}")
    mean_bytes = mean.tobytes(order="C")
    if tuple(components.shape) != (n_features, n_features) and components.dtype != np.float64:
        raise ValueError(f"PCA has invalid formatting: {components.shape} / {components.dtype}")
    components_bytes = components.tobytes(order="C")
    if tuple(variance.shape) != (n_features,) and variance.dtype != np.float64:
        raise ValueError(f"PCA has invalid formatting: {variance.shape} / {variance.dtype}")
    variance_bytes = variance.tobytes(order="C")
    return b"".join(
        [
            n_feature_bytes,
            mean_bytes,
            components_bytes,
            variance_bytes,
        ]
    )


def deserialize_pca(pca: bytes) -> PCA:
    n_features = int.from_bytes(pca[:8], byteorder="little", signed=True)
    n_feature_bytes = n_features * 8
    pca_rest = pca[8:]
    if len(pca_rest) != n_feature_bytes + n_feature_bytes + (n_feature_bytes * n_features):
        raise ValueError(f"Serialized PCA: {pca} is invalid")
    mean = np.frombuffer(pca_rest[:n_feature_bytes], dtype=np.float64)
    components = np.frombuffer(pca_rest[n_feature_bytes:-n_feature_bytes], dtype=np.float64)
    variance = np.frombuffer(pca_rest[-n_feature_bytes:], dtype=np.float64)

    pca = PCA(random_state=0, n_components=2)
    pca.n_features_in_ = n_features
    pca.mean_ = mean
    pca.components_ = components
    pca.explained_variance_ = variance
    return pca
