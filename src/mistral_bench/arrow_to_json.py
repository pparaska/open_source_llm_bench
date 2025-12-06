# arrow_any_to_json.py
import sys, os, json, glob
from typing import Iterable, Optional

import pyarrow as pa
import pyarrow.ipc as ipc

try:
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except Exception:
    _HAS_PARQUET = False

def read_arrow_file(path: str) -> pa.Table:
    # Arrow "file" format
    try:
        with pa.memory_map(path, "r") as source:
            reader = ipc.open_file(source)
            return reader.read_all()
    except Exception:
        pass
    # Arrow "stream" format
    try:
        with pa.memory_map(path, "r") as source:
            reader = ipc.open_stream(source)
            batches = [rb for rb in reader]
            if not batches:
                return pa.table({})
            return pa.Table.from_batches(batches)
    except Exception:
        pass
    # Parquet (fallback)
    if _HAS_PARQUET:
        try:
            return pq.read_table(path)
        except Exception:
            pass
    raise ValueError(f"{path!r} is not a readable Arrow IPC file/stream or Parquet.")

def iter_tables(path: str):
    if os.path.isdir(path):
        pats = sorted(
            glob.glob(os.path.join(path, "*.arrow"))
            + glob.glob(os.path.join(path, "*.parquet"))
            + glob.glob(os.path.join(path, "data-*.arrow"))
            + glob.glob(os.path.join(path, "part-*.parquet"))
        )
        if not pats:
            raise ValueError(f"No .arrow/.parquet files under {path}")
        for f in pats:
            yield read_arrow_file(f)
    else:
        yield read_arrow_file(path)

def to_json(path_in: str, path_out: str, jsonl: bool = True, limit: Optional[int] = None):
    n = 0
    if jsonl:
        with open(path_out, "w", encoding="utf-8") as f:
            for table in iter_tables(path_in):
                for row in table.to_pylist():
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
                    if limit and n >= limit:
                        return n
        return n
    else:
        rows = []
        for table in iter_tables(path_in):
            part = table.to_pylist()
            rows.extend(part)
            if limit and len(rows) >= limit:
                rows = rows[:limit]
                break
        with open(path_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        return len(rows)

def main():
    if len(sys.argv) < 3:
        print("Usage: python arrow_any_to_json.py <input.arrow|dir> <output.jsonl|json> [--json] [--limit N]")
        sys.exit(1)
    inp = sys.argv[1]
    outp = sys.argv[2]
    jsonl = "--json" not in sys.argv
    limit = None
    if "--limit" in sys.argv:
        i = sys.argv.index("--limit")
        if i + 1 < len(sys.argv):
            limit = int(sys.argv[i+1])
    written = to_json(inp, outp, jsonl=jsonl, limit=limit)
    print(f"Done. Wrote {written} records to {outp}")

if __name__ == "__main__":
    main()
