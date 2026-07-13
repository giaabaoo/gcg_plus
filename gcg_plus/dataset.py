import json


class HarmfulPromptsDataset:
    """
    Load a JSON prompt list: [{"id": int, "prompt": str}, ...]

    Config fields (dataset.*):
        data_path   : path to JSON file
        num_samples : first-N cap (0 = all)
        prompt_ids  : list of specific IDs to run ([] = all, takes priority over num_samples)
        shard_idx   : 0-based shard index for parallel runs (requires num_shards)
        num_shards  : total number of shards
    """

    def __init__(self, config):
        with open(config.dataset.data_path) as f:
            raw = json.load(f)

        ids = getattr(config.dataset, 'prompt_ids', [])
        if ids:
            raw = [item for item in raw if item['id'] in ids]
        else:
            num = getattr(config.dataset, 'num_samples', 0)
            if num and num > 0:
                raw = raw[:num]

        num_shards = getattr(config.dataset, 'num_shards', 1)
        shard_idx  = getattr(config.dataset, 'shard_idx',  0)
        if num_shards > 1:
            raw = [item for i, item in enumerate(raw) if i % num_shards == shard_idx]

        self.items = raw

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
