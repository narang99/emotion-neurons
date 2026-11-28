import pytest
import torch
from sklearn.decomposition import PCA

from emotion_neurons.vectors import neutral_pcs

torch.manual_seed(0)


@pytest.mark.parametrize("share", [0.3, 0.5, 0.9])
def test_neutral_pcs_match_sklearn_pca(share):
    """Same subspace and explained variance. The signs of the directions are free."""
    x = torch.randn(40, 12, dtype=torch.float64) * torch.linspace(3, 0.1, 12, dtype=torch.float64)
    pcs, explained = neutral_pcs(x, share)
    pca = PCA(n_components=share).fit(x.numpy())
    components = torch.from_numpy(pca.components_)
    assert pcs.shape == components.shape
    torch.testing.assert_close(pcs.T @ pcs, components.T @ components)
    assert explained == pytest.approx(pca.explained_variance_ratio_.sum())
