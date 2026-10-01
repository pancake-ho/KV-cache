"""Read-only checks for using the existing lab environment with Figure 2.

No package installation, pip invocation, model download, or login-node GPU
allocation. CUDA/BF16 device checks remain in the allocated Slurm jobs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--env', required=True)
    parser.add_argument('--output', help='Optional JSON provenance file, written only on success.')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    env = Path(args.env).expanduser().resolve()
    errors: list[str] = []
    report: dict = {
        'checked_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'python': platform.python_version(), 'executable': sys.executable,
        'prefix': sys.prefix, 'expected_prefix': str(env), 'root': str(root),
        'required_packages': {}, 'torch': None, 'cuda_build': None,
    }
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Environment: {sys.prefix}")
    if Path(sys.prefix).resolve() != env:
        errors.append(f"Wrong interpreter: expected {env}; got {sys.prefix}")
    if sys.version_info < (3, 10):
        errors.append('Python >=3.10 is required.')
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError as exc:
        raise SystemExit(f"packaging cannot be imported: {exc}. No packages changed.") from exc

    required_files = (
        'requirements-figure2.txt', 'configs/figure2_qwen3_1p7b_4b.json',
        'jobs/figure2/01_source.sh', 'jobs/figure2/02_target.sh',
        'jobs/figure2/03_probe_plot.sh',
        'tests/test_figure2.py', 'tests/test_figure2_tiny_qwen.py',
    )
    for relative in required_files:
        if not (root / relative).is_file():
            errors.append(f"Required file missing: {root / relative}")
    requirements = root / 'requirements-figure2.txt'
    if requirements.is_file():
        for line in requirements.read_text().splitlines():
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            required = Requirement(line)
            if required.marker and not required.marker.evaluate():
                continue
            try:
                version = metadata.version(required.name)
                report['required_packages'][required.name] = version
                print(f"{required.name}: {version} (required: {required.specifier})")
                if version not in required.specifier:
                    errors.append(f"Version mismatch: {required.name}=={version}; expected {required}")
                importlib.import_module(required.name.replace('-', '_'))
            except Exception as exc:
                errors.append(f"Dependency {required.name}: {type(exc).__name__}: {exc}")

    try:
        import torch
        import numpy as np
        report['torch'] = torch.__version__
        report['cuda_build'] = torch.version.cuda
        print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
        if Version(torch.__version__.split('+')[0]) < Version('2.4'):
            errors.append('PyTorch >=2.4 is required.')
        if torch.version.cuda is None:
            errors.append('CUDA-enabled PyTorch is required; this build is CPU-only.')
        torch.from_numpy(np.zeros(1, dtype=np.float32))
    except Exception as exc:
        errors.append(f"PyTorch/numpy import or bridge: {type(exc).__name__}: {exc}")

    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        print('Qwen3 model/config imports: OK')
    except Exception as exc:
        errors.append(f"Qwen3 import: {type(exc).__name__}: {exc}")
    config_path = root / 'configs/figure2_qwen3_1p7b_4b.json'
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text())
            pair = (config['source_model'], config['target_model'])
            if pair != ('Qwen/Qwen3-1.7B', 'Qwen/Qwen3-4B'):
                errors.append(f"Wrong Figure 2 model pair: {pair}")
            report['model_pair'] = list(pair)
        except Exception as exc:
            errors.append(f"Figure 2 config: {type(exc).__name__}: {exc}")
    sys.path.insert(0, str(root / 'src'))
    for module in ('prepare', 'extract', 'probe', 'plot'):
        try:
            importlib.import_module(f'xmodel_kv.figure2.{module}')
        except Exception as exc:
            errors.append(f"Figure 2 {module}: {type(exc).__name__}: {exc}")
    if errors:
        print('\nEnvironment check failed; no packages were installed or changed.', file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print('Resolve the reported differences before submitting jobs.', file=sys.stderr)
        raise SystemExit(2)
    if args.output:
        report['installed_packages'] = sorted(
            [{'name': distribution.metadata.get('Name', ''), 'version': distribution.version}
             for distribution in metadata.distributions()],
            key=lambda item: (item['name'].lower(), item['version']),
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('\nEnvironment ready. GPU availability/BF16 will be checked inside the allocated job.')


if __name__ == '__main__':
    main()
