"""CacheBridge v1 fit and evaluation entry point."""
from ..mapper_optim.cli import run
from .attention import QueryCapture, boundaries, shrinkage


if __name__ == "__main__":
    run("cachebridge", attention=(QueryCapture, boundaries, shrinkage))
