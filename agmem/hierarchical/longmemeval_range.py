"""Read selected LongMemEval-M rows with their original bank and parent addresses."""
import pyarrow.parquet as pq

from ..core import digest
from .adapters import _source, _normalize_history


def range_banks(path, sha256, split, first, last, *, count_tokens, wanted=None):
    parquet = pq.ParquetFile(path)
    start = 0
    for group in range(parquet.num_row_groups):
        end = start + parquet.metadata.row_group(group).num_rows
        if start >= last:
            break
        if end >= first and (wanted is None or any(start <= i < end for i in wanted)):
            index = start
            for batch in parquet.iter_batches(batch_size=2, row_groups=[group]):
                left, right = max(0, first - 1 - index), min(len(batch), last - index)
                if right > left:
                    for offset in range(index + left, index + right):
                        if wanted is not None and offset not in wanted:
                            continue
                        row = batch.slice(offset - index, 1).to_pylist()[0]
                        source = _source(path, sha256, split, offset, row, "longmemeval_parquet")
                        name = "split_longmemeval_m"
                        bank_id = "bank_" + digest([name, split, sha256, offset])[:24]
                        bank = _normalize_history(name, row, source, bank_id, count_tokens)
                        if wanted is not None:
                            parents = wanted[offset]
                            bank["sessions"] = [s for s in bank["sessions"] if s["parent_id"] in parents]
                            if {s["parent_id"] for s in bank["sessions"]} != parents:
                                raise ValueError(f"Failed to normalize required source sessions in {split} row {offset + 1}")
                            bank["source_map"] = [s for s in bank["source_map"] if s["parent_id"] in parents]
                        yield bank
                index += len(batch)
                if index >= last:
                    break
        start = end

