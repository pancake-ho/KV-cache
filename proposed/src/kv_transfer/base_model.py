"""Route base KV to fitted mappers by their declared model pair."""


class BaseModelKVTransfer:
    def __init__(self, mapper=None, *additional_mappers, prefix_mappers=()):
        self.mappers = {}
        self.prefix_mappers = {}
        legacy = (mapper, *additional_mappers) if mapper is not None else additional_mappers
        for artifact in legacy:
            pair = (artifact.config.source_model, artifact.config.target_model)
            if pair in self.mappers:
                raise ValueError(f"duplicate fitted mapper for {pair}")
            self.mappers[pair] = artifact
        for artifact in prefix_mappers:
            if artifact.pair in self.mappers or artifact.pair in self.prefix_mappers:
                raise ValueError(f"duplicate fitted mapper for {artifact.pair}")
            self.prefix_mappers[artifact.pair] = artifact

    def supports(self, source, target):
        pair = (source.model_id, target.model_id)
        return (source.adapter_id is None and target.adapter_id is None
                and (pair in self.mappers or pair in self.prefix_mappers))

    def map_cache(self, layers, *, source, target, source_model, target_model):
        if not self.supports(source, target):
            raise ValueError("no fitted mapper for the requested model and adapter pair")
        pair = (source.model_id, target.model_id)
        if pair in self.prefix_mappers:
            return self.prefix_mappers[pair].map_prefix(
                layers, source.token_ids, source_model=source_model, target_model=target_model,
            )
        return self.mappers[pair].map_cache(layers, source_model=source_model, target_model=target_model)
