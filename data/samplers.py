import random
from torch.utils.data import Sampler
from typing import List, Iterator, Optional


class VideoBatchSampler(Sampler[List[int]]):
    """
    Samples consecutive or shuffled batches.
    """
    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(self.num_samples))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class BucketSampler(Sampler[List[int]]):
    """
    Makes batches with samples of similar lengths reducing VRAM usage
    caused by padding length discrepancy.
    """
    def __init__(
        self,
        sequence_lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_boundaries: Optional[List[int]] = None,
        drop_last: bool = False
    ):
        self.sequence_lengths = sequence_lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_boundaries = bucket_boundaries or [8, 16, 32, 64, 100]

        self._build_buckets()

    def _build_buckets(self):
        """Assegna l'indice di ciascun campione al bucket corrispondente in base alla sua lunghezza."""
        self.buckets = [[] for _ in range(len(self.bucket_boundaries) + 1)]
        for idx, length in enumerate(self.sequence_lengths):
            bucket_idx = 0
            while bucket_idx < len(self.bucket_boundaries) and length > self.bucket_boundaries[bucket_idx]:
                bucket_idx += 1
            self.buckets[bucket_idx].append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        batches = []
        for bucket in self.buckets:
            if not bucket:
                continue
            
            if self.shuffle:
                random.shuffle(bucket)

            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        # Shuffle again to avoid training on the same length interval.
        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        total_batches = 0
        for bucket in self.buckets:
            if not bucket:
                continue
            if self.drop_last:
                total_batches += len(bucket) // self.batch_size
            else:
                total_batches += (len(bucket) + self.batch_size - 1) // self.batch_size
        return total_batches