import math
import random
from typing import Optional

from torch.utils.data import RandomSampler, Sampler


class BucketBatchSampler(Sampler):
    def __init__(
        self,
        bucket_limits: list[int],
        lengths: list[int],
        batch_cost: float,
        bucket_costs: Optional[list[float]] = None,
        drop_last: bool = True,
        round_batch_to_8: bool = False,
    ):

        # Modern GPUs can be more efficient when data is provided as a multiple of 8 (for 16-bit training)
        self.round_batch_to_8 = round_batch_to_8
        self.drop_last = drop_last

        if bucket_costs is not None and len(bucket_costs) != len(bucket_limits):
            raise ValueError("The number of costs and buckets must be the same.")

        if max(lengths) > max(bucket_limits):
            raise ValueError("Largest length cannot be larger than largest bucket limit.")

        bucket_limits = sorted(bucket_limits)

        # Use a constant bucket cost by default
        bucket_costs = [1] * len(bucket_limits) if bucket_costs is None else bucket_costs

        # Add indices to correct bucket based on seq length
        buckets = [[] for _ in range(len(bucket_limits))]
        for seq_idx, length in enumerate(lengths):
            for b_idx, limit in enumerate(bucket_limits):
                if limit >= length:
                    buckets[b_idx].append(seq_idx)
                    break

        # TODO allow non-shuffled sampling
        samplers = [RandomSampler(idxs, replacement=False) if len(idxs) > 0 else None for idxs in buckets]
        bucket_batch_sizes = [self._round_batch_size(batch_cost / cost) for cost in bucket_costs]

        batches_per_bucket = []
        for bucket, batch_size in zip(buckets, bucket_batch_sizes):
            n_batches = int(len(bucket) // batch_size)
            if not drop_last and n_batches * batch_size != len(bucket):
                n_batches += 1

            batches_per_bucket.append(n_batches)

        print()
        print("items per bucket", [len(idxs) for idxs in buckets])
        print("bucket batch sizes", bucket_batch_sizes)
        print("batches per bucket", batches_per_bucket)

        self.buckets = buckets
        self.samplers = samplers
        self.bucket_batch_sizes = bucket_batch_sizes
        self.batches_per_bucket = batches_per_bucket

    def __len__(self):
        return sum(self.batches_per_bucket)

    def __iter__(self):
        iters = [iter(sampler) if sampler is not None else None for sampler in self.samplers]
        remaining_batches = self.batches_per_bucket[:]
        remaining_items = [len(items) for items in self.buckets]

        while sum(remaining_batches) > 0:
            b_idx = random.choices(range(len(remaining_batches)), weights=remaining_batches, k=1)[0]
            if remaining_batches[b_idx] > 1 or self.drop_last:
                batch_size = self.bucket_batch_sizes[b_idx]
            else:
                batch_size = remaining_items[b_idx]

            batch_idxs = [next(iters[b_idx]) for _ in range(batch_size)]

            # Samplers will produce indices into the list, so look up dataset indices using sampled bucket indices
            batch = [self.buckets[b_idx][idx] for idx in batch_idxs]

            remaining_batches[b_idx] -= 1
            remaining_items[b_idx] -= batch_size

            yield batch

    def _round_batch_size(self, batch_size):
        if not self.round_batch_to_8:
            bs = math.floor(batch_size)
        else:
            bs = 8 * round(batch_size / 8)

        bs = 1 if bs == 0 else bs
        return bs
