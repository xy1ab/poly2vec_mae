from pathlib import Path


def test_pretrain_defaults_use_layered_paths():
    root = Path(__file__).resolve().parents[1]
    content = (root / 'configs' / 'mae' / 'pretrain.yaml').read_text(encoding='utf-8')
    assert './data/processed/polygon_triangles_normalized.pt' in content
    assert './outputs/checkpoints/' in content
    assert './outputs/exports/' in content
    assert 'train_type: mae' in content
    assert 'precision: bf16' in content
    assert 'checkpoint_dtype: bf16' in content


def test_eval_defaults_use_outputs_checkpoints():
    root = Path(__file__).resolve().parents[1]
    content = (root / 'configs' / 'mae' / 'eval.yaml').read_text(encoding='utf-8')
    assert './outputs/checkpoints/eval' in content


def test_recons_defaults_enable_bf16():
    root = Path(__file__).resolve().parents[1]
    content = (root / 'configs' / 'downstream' / 'recons.yaml').read_text(encoding='utf-8')
    assert 'precision: bf16' in content
