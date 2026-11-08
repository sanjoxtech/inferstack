from pathlib import Path
from inferstack import __version__
from inferstack.config import load_profile
from inferstack.foundry import FileSystemFoundryRepository,FoundryModelLoader


def test_version_and_default_fusion_profile():
    from importlib.metadata import version
    assert __version__==version('inferstack')
    c=load_profile('configs/profiles/production-v1.6.0.toml')
    assert c.scheduler.decode_priority and c.native.live_batch_refill
    assert c.native.forward_fusion and c.native.fuse_mlp_gate_up and c.native.fuse_gdn_projections


def test_all_v160_profiles_load():
    for n in ('production-v1.6.0','production-v1.6.0-baseline','production-v1.6.0-norm','production-v1.6.0-mlp','production-v1.6.0-gdn','production-v1.6.0-attention','production-v1.6.0-profile'):
        c=load_profile(f'configs/profiles/{n}.toml');assert c.name==n


def test_foundry_candidate_resolution_and_plan():
    c=load_profile('configs/profiles/production-v1.6.0.toml');r=FileSystemFoundryRepository('examples/model_foundry');m,src,p=FoundryModelLoader(r,'cpu').inspect('foundry://factorylm-2b@candidate',c)
    assert m['model_id']=='factorylm-2b' and src['uri']=='Qwen/Qwen3.5-2B-Base' and p.quantization=='none'


def test_ablation_profiles_isolate_one_forward_group():
    root=Path('configs/profiles')
    b=load_profile(root/'production-v1.6.0-baseline.toml');n=load_profile(root/'production-v1.6.0-norm.toml');m=load_profile(root/'production-v1.6.0-mlp.toml');g=load_profile(root/'production-v1.6.0-gdn.toml');a=load_profile(root/'production-v1.6.0-attention.toml');f=load_profile(root/'production-v1.6.0.toml')
    assert not b.native.forward_fusion
    assert n.native.forward_fusion and n.native.fuse_rmsnorm and not n.native.fuse_mlp_gate_up and not n.native.fuse_gdn_projections
    assert m.native.forward_fusion and m.native.fuse_mlp_gate_up and not m.native.fuse_rmsnorm and not m.native.fuse_gdn_projections
    assert g.native.forward_fusion and g.native.fuse_gdn_projections and g.native.grouped_gdn_decode and not g.native.fuse_mlp_gate_up
    assert a.native.forward_fusion and a.native.fuse_attention_qkv and not a.native.fuse_mlp_gate_up and not a.native.fuse_gdn_projections
    assert all((f.native.fuse_rmsnorm,f.native.fuse_add_rmsnorm,f.native.fuse_gated_rmsnorm,f.native.fuse_mlp_gate_up,f.native.fuse_gdn_projections,f.native.grouped_gdn_decode,f.native.fuse_attention_qkv))


def test_loader_keeps_hardened_weight_only_quantization_code_for_future_experiments():
    text=Path('src/inferstack/foundry/loader.py').read_text()
    assert 'Int8WeightOnlyConfig()' in text
    assert 'Int8DynamicActivationInt8WeightConfig' not in text
    assert "int4_packing_format='tile_packed_to_4d'" in text


def test_cli_defaults_to_v160():
    text=Path('src/inferstack/cli.py').read_text()
    assert text.count('configs/profiles/production-v1.6.0.toml')>=3
