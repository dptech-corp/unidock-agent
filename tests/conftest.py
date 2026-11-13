"""Shared fixtures and configuration for integration tests.

Run with:  pytest tests/ -v
"""
import os
import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FF_DATA_DIR = os.path.join(
    REPO_ROOT, "unidock", "unidock_engine", "test", "data", "ff_minimize", "tyk2"
)
BACE_DIR = os.path.join(
    REPO_ROOT, "examples", "constraint_docking", "automatic_atom_mapping"
)
FREE_DOCK_DIR = os.path.join(
    REPO_ROOT, "examples", "free_docking", "molecular_docking"
)
BATCH_LIGAND_DIR = os.path.join(
    REPO_ROOT, "unidock", "unidock_processing", "test", "data", "batch_ligand"
)
RESCORE_TEST_DIR = os.path.join(REPO_ROOT, "rescore_test")
UD2ALIGN_DATA = os.environ.get("UD2ALIGN_DATA")
COVALENT_DATA_DIR = os.path.join(
    REPO_ROOT, "unidock", "unidock_processing", "test", "data", "covalent_docking", "1EWL"
)
PROTAC_DIR = os.path.join(
    REPO_ROOT, "examples", "protac_docking"
)
PROTAC_MZ1_DIR = os.path.join(PROTAC_DIR, "MZ1-BRD4-VHL")


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")


# ---------------------------------------------------------------------------
# Data-path fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tyk2_ligand_dms():
    p = os.path.join(FF_DATA_DIR, "ligand_ejm42.dms")
    if not os.path.isfile(p):
        pytest.skip(f"Missing test data: {p}")
    return p


@pytest.fixture(scope="session")
def tyk2_protein_dms():
    p = os.path.join(FF_DATA_DIR, "protein.dms")
    if not os.path.isfile(p):
        pytest.skip(f"Missing test data: {p}")
    return p


@pytest.fixture(scope="session")
def bace_receptor():
    p = os.path.join(BACE_DIR, "Bace.pdb")
    if not os.path.isfile(p):
        pytest.skip(f"Missing test data: {p}")
    return p


@pytest.fixture(scope="session")
def bace_reference():
    p = os.path.join(BACE_DIR, "reference.sdf")
    if not os.path.isfile(p):
        pytest.skip(f"Missing test data: {p}")
    return p


@pytest.fixture(scope="session")
def bace_cat4a():
    p = os.path.join(BACE_DIR, "CAT-4a.sdf")
    if not os.path.isfile(p):
        pytest.skip(f"Missing test data: {p}")
    return p


@pytest.fixture(scope="session")
def bace_cat13a():
    p = os.path.join(BACE_DIR, "CAT-13a.sdf")
    if not os.path.isfile(p):
        pytest.skip(f"Missing test data: {p}")
    return p


@pytest.fixture(scope="session")
def cdk4_receptor():
    p = os.path.join(RESCORE_TEST_DIR, "7sj3_prep.pdb")
    if not os.path.isfile(p):
        pytest.skip(f"Missing test data: {p}")
    return p


@pytest.fixture(scope="session")
def cdk4_3lig():
    for d in (BATCH_LIGAND_DIR, RESCORE_TEST_DIR):
        p = os.path.join(d, "cdk4_3lig.sdf")
        if os.path.isfile(p):
            return p
    pytest.skip("Missing test data: cdk4_3lig.sdf")


@pytest.fixture(scope="session")
def ewl_receptor():
    p = os.path.join(COVALENT_DATA_DIR, "1EWL_prepared.pdb")
    if not os.path.isfile(p):
        pytest.skip(f"Missing test data: {p}")
    return p


@pytest.fixture(scope="session")
def ewl_covalent_ligand():
    p = os.path.join(COVALENT_DATA_DIR, "covalent_mol.sdf")
    if not os.path.isfile(p):
        pytest.skip(f"Missing test data: {p}")
    return p


@pytest.fixture(scope="session")
def protac_mz1_dir():
    """MZ1-BRD4-VHL PROTAC example data directory."""
    required = ["brd_lig.sdf", "vhl_lig.sdf", "protac_random.sdf",
                "BRD4.pdb", "VHL.pdb"]
    for fname in required:
        p = os.path.join(PROTAC_MZ1_DIR, fname)
        if not os.path.isfile(p):
            pytest.skip(f"Missing PROTAC test data: {p}")
    return PROTAC_MZ1_DIR


@pytest.fixture(scope="session")
def protac_scripts_dir():
    """Directory containing protac_sampling.py and run_protac_docking.py."""
    return PROTAC_DIR
