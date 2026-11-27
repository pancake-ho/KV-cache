"""A directed cross-model KV mapper bound to two LoRA checkpoints."""


class AdapterPairKVTransfer:
    def __init__(self, mapper, source, target):
        self.mapper = mapper
        self.source = self._key(source)
        self.target = self._key(target)
        if (self.source[0] == self.target[0]
                or not all(endpoint[1] and endpoint[2] for endpoint in (self.source, self.target))
                or mapper.pair != (self.source[0], self.target[0])):
            raise ValueError("Expected two fingerprinted adapters on the mapper's directed model pair")

    @staticmethod
    def _key(adapter):
        return adapter["model_id"], adapter["adapter_id"], adapter["adapter_sha256"]

    def supports(self, source, target):
        return False  # This mapper cannot authorize a base-model route.

    def supports_adapter_route(self, source, target, *, source_sha256, target_sha256):
        return ((source.model_id, source.adapter_id, source_sha256) == self.source
                and (target.model_id, target.adapter_id, target_sha256) == self.target)

    def map_cache(self, layers, *, source, target, source_model, target_model, stage_events=None):
        if ((source.model_id, source.adapter_id) != self.source[:2]
                or (target.model_id, target.adapter_id) != self.target[:2]):
            raise ValueError("Request differs from the registered LoRA route")
        return self.mapper.map_prefix(
            layers, source.token_ids, source_model=source_model, target_model=target_model,
            **({"stage_events": stage_events} if stage_events is not None else {}),
        )
