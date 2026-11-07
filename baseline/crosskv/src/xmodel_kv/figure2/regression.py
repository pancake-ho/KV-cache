from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Moments:
    centered: object
    mean: np.ndarray
    covariance: np.ndarray | None
    variance: np.ndarray


class Statistics:
    """Centered sufficient statistics; CUDA float32 or offline CPU float64 reference."""
    def __init__(self, device="cuda:0", dtype="float32"):
        self.device = device
        self.dtype = dtype
        self.torch = None
        if device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required; CPU fallback is disabled")
            torch.backends.cuda.matmul.allow_tf32 = False
            self.torch = torch
        elif device != "cpu":
            raise ValueError("device must be cpu or cuda:N")

    def moments(self, array, *, covariance):
        if array.ndim != 3 or array.shape[0] < 2:
            raise ValueError("expected at least two observations in [N,H,D]")
        if self.torch is None:
            tensor = np.array(array, dtype=self.dtype, copy=True).transpose(1, 0, 2)
            mean = tensor.mean(axis=1)
            centered = tensor - mean[:, None]
            variance = np.mean(centered**2, axis=1)
            cov = (centered.transpose(0, 2, 1) @ centered) / tensor.shape[1] if covariance else None
        else:
            torch = self.torch
            array_copy = np.array(array, dtype=self.dtype, copy=True)
            tensor = torch.from_numpy(array_copy).to(self.device).permute(1, 0, 2).contiguous()
            del array_copy
            mean = tensor.mean(dim=1)
            centered = tensor - mean[:, None]
            variance = centered.square().mean(dim=1)
            cov = torch.bmm(centered.transpose(1, 2), centered) / tensor.shape[1] if covariance else None
            mean = mean.cpu().double().numpy()
            variance = variance.cpu().double().numpy()
            cov = cov.cpu().double().numpy() if cov is not None else None
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ValueError("nonfinite activation statistics")
        return Moments(centered, np.asarray(mean, dtype=np.float64), cov,
                       np.asarray(variance, dtype=np.float64))

    def cross(self, source, target):
        x, y = source.centered, target.centered
        if x.shape[:2] != y.shape[:2]:
            raise ValueError("matched observations and KV heads are required")
        if self.torch is None:
            value = x.transpose(0, 2, 1) @ y / x.shape[1]
        else:
            value = self.torch.bmm(x.transpose(1, 2), y) / x.shape[1]
            value = value.cpu().double().numpy()
        return np.asarray(value, dtype=np.float64)


def ols_inverse(covariance, rcond):
    """PSD pseudoinverse, not ridge: only numerical null-space directions are removed."""
    if not 0 < rcond < 1:
        raise ValueError("rcond must be in (0,1)")
    sym = (covariance + covariance.transpose(0, 2, 1)) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(sym.astype(np.float64))
    maximum = np.maximum(eigenvalues[:, -1:], np.finfo(np.float64).tiny)
    keep = eigenvalues > maximum * rcond
    reciprocal = np.zeros_like(eigenvalues)
    np.divide(1.0, eigenvalues, out=reciprocal, where=keep)
    inverse = (eigenvectors * reciprocal[:, None, :]) @ eigenvectors.transpose(0, 2, 1)
    return inverse, keep.sum(axis=1), eigenvalues[:, 0]


def score(source, target, cross, weight, bias):
    if source.covariance is None:
        raise ValueError("source covariance required")
    linear = np.einsum("hdo,hdo->ho", weight, cross)
    quadratic = np.einsum("hdo,hde,heo->ho", weight, source.covariance, weight)
    shift = target.mean - np.einsum("hd,hdo->ho", source.mean, weight) - bias
    mse_raw = target.variance - 2 * linear + quadratic + shift**2
    if not np.isfinite(mse_raw).all():
        raise ValueError("nonfinite residual statistics")
    tolerance = 1e-4 * np.maximum(target.variance, np.finfo(np.float64).tiny)
    if np.any(mse_raw < -tolerance):
        raise ValueError("materially negative SSE from statistics; rerun with float64 covariance in a new run directory")
    # Tiny negative SSE is possible from roundoff in sufficient-statistic arithmetic.
    mse = np.maximum(mse_raw, 0.0)
    valid = target.variance > 0
    r2 = np.full_like(mse, np.nan)
    np.divide(mse, target.variance, out=r2, where=valid)
    r2[valid] = 1.0 - r2[valid]
    valid_counts = valid.sum(axis=1)
    if np.any(valid_counts == 0):
        raise ValueError("a complete target head has zero variance")
    uniform_head = np.nansum(r2, axis=1) / valid_counts
    weighted_head = 1.0 - (mse * valid).sum(axis=1) / (target.variance * valid).sum(axis=1)
    return {"uniform": uniform_head, "weighted": weighted_head,
            "valid_channels": valid_counts, "roundoff_clamped_channels": (mse_raw < 0).sum(axis=1)}


def fit_pair(engine, source_train, target_train, source_test, target_test, inverse):
    train_cross = engine.cross(source_train, target_train)
    weight = inverse @ train_cross
    bias = target_train.mean - np.einsum("hd,hdo->ho", source_train.mean, weight)
    train = score(source_train, target_train, train_cross, weight, bias)
    test = score(source_test, target_test, engine.cross(source_test, target_test), weight, bias)
    return train, test
