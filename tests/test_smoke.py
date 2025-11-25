import importlib


def test_smoke_imports():
    """Simple smoke test to ensure core modules import correctly.

    This keeps CI fast while checking the main project modules load without
    executing training or app runtime code (those live behind __main__ guards).
    """
    modules = [
        'preprocessing',
        'feature_engineering',
        'train',
        'app',
    ]
    for m in modules:
        importlib.import_module(m)

