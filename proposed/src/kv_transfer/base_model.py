"""Use an existing trained Ridge or neural mapper for its declared base pair."""


class BaseModelKVTransfer:
    def __init__(self, mapper):
        self.mapper = mapper
        self.pair = (mapper.config.source_model, mapper.config.target_model)

    def supports(self, source, target):
        return (source.adapter_id is None and target.adapter_id is None
                and (source.model_id, target.model_id) == self.pair)

    def map_cache(self, layers, *, source, target, source_model, target_model):
        if not self.supports(source, target):
            raise ValueError("no fitted mapper for the requested model and adapter pair")
        return self.mapper.map_cache(layers, source_model=source_model, target_model=target_model)
