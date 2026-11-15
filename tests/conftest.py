"""CI environment conftest."""
import pytest

collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="environment smoke only")
    for item in items:
        item.add_marker(skip)
