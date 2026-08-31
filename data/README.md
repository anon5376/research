# Local data

Large datasets live here and are not committed. Stream when you can; download subsets, not the whole Tahoe-100M dump (~429 GB), unless you have the disk.

```bash
# example: stream Tahoe-100M (do not download by default)
# python -c "from datasets import load_dataset; ds = load_dataset('tahoebio/Tahoe-100M', streaming=True, split='train'); print(next(iter(ds)))"
```
