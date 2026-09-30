import numpy as np

from qcv.covariance import fit_residual_covariance


def test_ledoit_wolf_covariance_is_positive_definite() -> None:
    generator = np.random.default_rng(7)
    residuals = generator.normal(size=(128, 8))
    residuals[:, 1] = residuals[:, 0] * 0.9 + generator.normal(scale=0.05, size=128)
    model = fit_residual_covariance("sig", ["a", "b"], residuals)
    eigenvalues = np.linalg.eigvalsh(model.covariance)
    assert np.min(eigenvalues) > 0
    identity = model.whitening @ model.covariance @ model.whitening.T
    assert np.allclose(identity, np.eye(8), atol=1e-8)
    assert model.alpha >= 1 / 128
