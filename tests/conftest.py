"""pytest support: the end-to-end grid tests take a temporary directory argument named `tmp`."""
import pytest


@pytest.fixture
def tmp(tmp_path):
    return tmp_path
