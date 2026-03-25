from pathlib import Path


def test_required_top_level_dirs_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / 'configs',
        root / 'data',
        root / 'models',
        root / 'outputs',
        root / 'scripts',
        root / 'src',
        root / 'tests',
    ]
    for p in required:
        assert p.exists() and p.is_dir(), f'missing dir: {p}'


def test_data_layering_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / 'data' / 'raw').is_dir()
    assert (root / 'data' / 'processed').is_dir()


def test_utils_layering_exists():
    root = Path(__file__).resolve().parents[1]
    utils = root / 'src' / 'utils'
    for sub in ['config', 'data', 'fourier', 'geometry', 'io', 'viz']:
        assert (utils / sub).is_dir(), f'missing utils subdir: {sub}'


def test_outputs_checkpoint_layering_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / 'outputs' / 'checkpoints').is_dir()
    assert (root / 'outputs' / 'exports').is_dir()
