"""Route base KV to fitted mappers by their declared model pair."""


class BaseModelKVTransfer:
    def __init__(self, mapper, *additional_mappers):
        self.mappers = {}
        for artifact in (mapper, *additional_mappers):
            pair = (artifact.config.source_model, artifact.config.target_model)
            if pair in self.mappers:
                raise ValueError(f"duplicate fitted mapper for {pair}")
            self.mappers[pair] = artifact

    def supports(self, source, target):
        return (source.adapter_id is None and target.adapter_id is None
                and (source.model_id, target.model_id) in self.mappers)

    def map_cache(self, layers, *, source, target, source_model, target_model):
        if not self.supports(source, target):
            raise ValueError("no fitted mapper for the requested model and adapter pair")
        mapper = self.mappers[(source.model_id, target.model_id)]
        return mapper.map_cache(layers, source_model=source_model, target_model=target_model)
