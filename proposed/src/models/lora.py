"""Load immutable LoRA identities and select one adapter per serialized request."""

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from threading import RLock


class LoRARegistry:
    def __init__(self, models):
        self.models = models
        self._peft = {}
        self._fingerprints = {}
        self._lock = RLock()

    def load(self, model_id, adapter_id, directory):
        if not adapter_id:
            raise ValueError("a LoRA adapter ID is required")
        directory = Path(directory).resolve()
        config_file = directory / "adapter_config.json"
        weights_file = directory / "adapter_model.safetensors"
        if not config_file.is_file() or not weights_file.is_file():
            raise FileNotFoundError(f"expected local LoRA config and safetensors in {directory}")

        from peft import PeftConfig, PeftModel, PeftType

        with self._lock:
            key = (model_id, adapter_id)
            if key in self._fingerprints:
                raise ValueError(f"duplicate LoRA registration: {key}")
            base = self.models[model_id]
            config = PeftConfig.from_pretrained(directory, local_files_only=True)
            if config.peft_type != PeftType.LORA or config.base_model_name_or_path != model_id:
                raise ValueError(f"LoRA base mismatch for {directory}, expected {model_id}")
            base_revision = getattr(base.config, "_commit_hash", None)
            if config.revision is not None and config.revision != base_revision:
                raise ValueError(f"LoRA base revision mismatch for {directory}")
            digest = sha256()
            for path in (config_file, weights_file):
                digest.update(path.name.encode())
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
            kwargs = dict(adapter_name=adapter_id, is_trainable=False, local_files_only=True)
            if model_id in self._peft:
                self._peft[model_id].load_adapter(directory, **kwargs)
            else:
                self._peft[model_id] = PeftModel.from_pretrained(base, directory, **kwargs)
            self._peft[model_id].eval()
            self._fingerprints[key] = digest.hexdigest()
            return {
                "model_id": model_id, "adapter_id": adapter_id, "path": str(directory),
                "adapter_sha256": digest.hexdigest(), "adapter_base_revision": config.revision,
            }

    @contextmanager
    def activate(self, model_id, adapter_id=None):
        # The active adapter changes mutable PEFT state and is held for the entire request.
        with self._lock:
            model = self.models[model_id]
            peft = self._peft.get(model_id)
            if adapter_id is not None:
                key = (model_id, adapter_id)
                if key not in self._fingerprints:
                    raise ValueError(f"unknown LoRA adapter: {key}")
                peft.set_adapter(adapter_id)
                yield peft.get_base_model(), self._fingerprints[key]
            elif peft is not None:
                with peft.disable_adapter():
                    yield peft.get_base_model(), None
            else:
                yield model, None
