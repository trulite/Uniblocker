import sys
from pathlib import Path
import types
import importlib.util
import unittest

# create minimal package structure to load module without executing src.__init__
utils_pkg = types.ModuleType("utils")
utils_pkg.__path__ = []
nnblocker_pkg = types.ModuleType("utils.nnblocker")
nnblocker_pkg.__path__ = []
indexers_pkg = types.ModuleType("utils.nnblocker.indexers")
indexers_pkg.__path__ = []
sys.modules.setdefault("utils", utils_pkg)
sys.modules.setdefault("utils.nnblocker", nnblocker_pkg)
sys.modules.setdefault("utils.nnblocker.indexers", indexers_pkg)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
base = Path(__file__).resolve().parents[1] / "src" / "utils" / "nnblocker" / "indexers"
spec = importlib.util.spec_from_file_location(
    "utils.nnblocker.indexers.indexer", base / "indexer.py"
)
indexer_module = importlib.util.module_from_spec(spec)
sys.modules["utils.nnblocker.indexers.indexer"] = indexer_module
spec.loader.exec_module(indexer_module)  # type: ignore

spec = importlib.util.spec_from_file_location(
    "utils.nnblocker.indexers.fastkmeans_indexer", base / "fastkmeans_indexer.py"
)
fast_module = importlib.util.module_from_spec(spec)
sys.modules["utils.nnblocker.indexers.fastkmeans_indexer"] = fast_module
spec.loader.exec_module(fast_module)  # type: ignore
FastKMeansIndexer = fast_module.FastKMeansIndexer


class FastKMeansIndexerTest(unittest.TestCase):
    def test_fastkmeans_indexer_shapes(self):
        data = [[float(i + j) for j in range(4)] for i in range(20)]
        queries = data[:5]
        indexer = FastKMeansIndexer(n_clusters=4, random_state=0)
        indexer.build_index(data)
        result = indexer.batch_search(queries, k=3)
        self.assertEqual(len(result.batch_indices), len(queries))
        self.assertTrue(all(len(idx) == 3 for idx in result.batch_indices))


if __name__ == "__main__":
    unittest.main()
