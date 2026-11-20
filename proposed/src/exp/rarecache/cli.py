"""RaReCache fit and evaluation entry point."""
from ..mapper_optim.cli import run
from .rank import disagreement, fit_basis, select_positions
from .recompute import recompute


if __name__ == "__main__":
    run("rarecache", basis=fit_basis, rank_score=disagreement,
        replay=(select_positions, recompute))
