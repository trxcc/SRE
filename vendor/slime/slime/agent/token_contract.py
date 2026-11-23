"""Fresh ResearchClaw six-candidate authority; exact token validation is retained below."""
import hashlib
import json
import math
import os
from pathlib import Path



def _explicit_hf_resume_contract():
    name = os.environ.get('RESEARCHCLAW_HF_RESUME_CONTRACT_MODULE', '')
    seal = os.environ.get('RESEARCHCLAW_HF_RESUME_CONTRACT_SHA', '')
    if name != 'runtime.portable_contract' or not seal:
        raise ValueError('Use the packaged launcher and sealed portable contract')
    path = Path(os.environ['BUNDLE_ROOT']).resolve() / 'runtime/portable_contract.py'
    manifest = json.loads(Path(os.environ['GEMMA_SELFREWARD_MANIFEST']).read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != seal or manifest.get('portable_contract_sha256') != seal:
        raise ValueError('Portable contract source mismatch')
    from importlib import import_module
    module = import_module(name)
    if Path(module.__file__).resolve() != path:
        raise ValueError('Wrong portable contract imported')
    return module


def validate_objective(args):
    if getattr(args, 'use_tis', None) is not False:
        raise ValueError('User requires TIS disabled')
    if getattr(args, 'use_kl_loss', None) is not False:
        raise ValueError('User requires KL disabled')
    for name in ('kl_loss_coef', 'kl_coef'):
        value = getattr(args, name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != 0:
            raise ValueError('User requires zero KL coefficient: ' + name)


def validate_recipe(args):
    validate_objective(args)
    recovery=_explicit_hf_resume_contract()
    if recovery is not None:return recovery.validate_recipe(args)
    manifest = json.loads(Path(os.environ['GEMMA_SELFREWARD_MANIFEST']).read_text())
    exact = dict(train_backend='nemo_megatron', advantage_estimator='grpo',
                 start_rollout_id=0, num_rollout=113, rollout_batch_size=16,
                 n_samples_per_prompt=6, global_batch_size=96,
                 num_steps_per_rollout=1, use_tis=False, use_rollout_logprobs=False,
                 use_kl_loss=False, kl_loss_coef=0.0, kl_coef=0.0,
                 normalize_advantages=False, colocate=False, use_critic=False,
                 no_load_optim=True, no_load_rng=True, update_weights_interval=1,
                 update_weight_transport='nccl', update_weight_mode='full',
                 actor_num_nodes=1, actor_num_gpus_per_node=4, rollout_num_gpus=4, num_gpus_per_node=2,
                 tensor_model_parallel_size=4, context_parallel_size=1,
                 pipeline_model_parallel_size=1, rollout_num_gpus_per_engine=1)
    for name, expected in exact.items():
        if getattr(args,name,None) != expected:
            raise ValueError('ResearchClaw fresh self-reward recipe mismatch: '+name)
    for name, expected in [('lr',1e-6),('rollout_temperature',.7),('rollout_top_p',1.),('tis_clip',2.),('tis_clip_low',0.)]:
        value=getattr(args,name,None)
        if not isinstance(value,(int,float))or not math.isfinite(value)or value!=expected:
            raise ValueError('ResearchClaw numeric recipe mismatch: '+name)
    for name, expected in [('rollout_top_k',-1),('eval_interval',20),('rollout_max_context_len',73728),('rollout_max_response_len',16384),('eval_max_context_len',81920),('eval_max_response_len',16384)]:
        if getattr(args,name,None)!=expected:raise ValueError('ResearchClaw native profile mismatch: '+name)
    if args.skip_eval_before_train:raise ValueError('new base run requires its own initial evaluation')
    if args.eval_function_path!='direct1.aligned_eval_recovery.generate_rollout':raise ValueError('native evaluator required')
    if args.data_source_path!='clawgym_nanobot.polar_data.FiniteSequentialRolloutDataSource':raise ValueError('finite one-pass datasource required')
    base=Path(manifest['original_checkpoint']).resolve()
    if Path(args.hf_checkpoint).resolve()!=base or Path(args.ref_load).resolve()!=base:raise ValueError('fresh actor/reference must use original Gemma base')
    if getattr(args,'load',None)and Path(args.load).resolve()!=base:raise ValueError('fresh base run must not load a trained checkpoint')
    if (base/'latest_checkpointed_iteration.txt').exists():raise ValueError('base source contains training-state marker')
    return base


def validate_fresh_args(args):
    validate_objective(args)
    recovery=_explicit_hf_resume_contract()
    if recovery is not None:return recovery.validate_recovery_args(args)
    base=validate_recipe(args)
    mp=Path(os.environ['GEMMA_SELFREWARD_MANIFEST']);raw=mp.read_bytes();m=json.loads(raw)
    probe_path=os.environ.get('GEMMA_RESEARCHCLAW_GPU_PROBE')
    if probe_path:
        probe=Path(probe_path);binding=m.get('gpu_probe_authority')or{}
        if m['status']!='prepared_not_ready'or str(probe)!=binding.get('path')or hashlib.sha256(probe.read_bytes()).hexdigest()!=binding.get('sha256'):raise ValueError('unbound one-use GPU probe')
        authorization=json.loads(probe.read_text())
        if authorization.get('status')!='ready_for_single_probe'or authorization.get('formal_training_launched')is not False or authorization.get('synthetic_only')is not True:raise ValueError('invalid synthetic probe authorization')
        if not(probe.parent/'framework-dispatched.json').is_file()or(probe.parent/'process-exit.json').exists():raise ValueError('GPU probe is not the current unique native framework driver')
    elif m['status']!='ready_for_single_launch':raise ValueError('new runtime has not passed launch preflight')
    run=Path(m['run']).resolve()
    authority=run/'authority.json'
    if hashlib.sha256(authority.read_bytes()).hexdigest()!=m['authority_sha256']:raise ValueError('new authority changed')
    a=json.loads(authority.read_text())
    if a['owner']!='parent'or a['run']!=str(run)or a['candidate_count']!=6:raise ValueError('new authority mismatch')
    old_owner=json.loads(Path(a['prior_owner_file']).read_text())
    if old_owner['owner']!='parent'or old_owner['run']!=a['prior_run']:raise ValueError('allocation ownership changed')
    if Path(a['prior_active_run_pointer']).read_text().strip()!=a['prior_run']or Path(a['prior_active_operator_pointer']).read_text().strip()!=a['prior_operator']:raise ValueError('allocation active pointers changed')
    for item in m['files'].values():
        p=Path(item['path'])
        if not p.resolve().is_relative_to(run)or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('new source/config seal mismatch')
    if Path(m['files']['token_contract']['path']).resolve()!=Path(__file__).resolve():raise ValueError('wrong contract imported')
    cfg=json.loads(Path(os.environ['CLAWGYM_SELF_REWARD_CONFIG']).read_text())
    if not cfg['enabled']or cfg['candidate_count']!=6 or cfg['mode']!='self_reward_5plus1_train5'or cfg['profile']!={'dataset':'researchclaw','model':'gemma4-e4b-it'}:raise ValueError('six-candidate config mismatch')
    for key in ['queue_root','snapshot_root']:
        if not Path(cfg[key]).resolve().is_relative_to(run):raise ValueError('shared prior reward state')
    expected={'OPENCLAW_TRAIN_ORDER':'sequential','OPENCLAW_TRAIN_START_INDEX':'0','CLAWGYM_SELF_REWARD_GROUP_SIZE':'6','SLIME_ASYNC_TRAINING':'1','USE_POLAR':'1','POLAR_SYNCHRONOUS_POLICY_VERSIONS':'0','POLAR_MAX_ASYNC_LEVEL':'1','POLAR_MAX_OFF_POLICY_STEPS':'2','CLAWGYM_NANOBOT_REQUIRE_EXACT_TOKENS':'1','CLAWGYM_NANOBOT_REASONING_CONTEXT_POLICY':'preserve','NANOBOT_STRIP_REASONING_CONTEXT':'0','CLAWGYM_NANOBOT_ENABLE_REALIGN':'1','CLAWGYM_ALLOW_MULTI_TOOL_CALLS':'1','NANOBOT_MICROCOMPACT_ENABLED':'0'}
    for name,value in expected.items():
        if os.environ.get(name)!=value:raise ValueError('new environment mismatch: '+name)
    if Path(os.environ['OPENCLAW_TRAIN_TASKS']).resolve()!=Path(m['train_view']).resolve():raise ValueError('wrong fixed training view')
    return {'validated':True,'base':str(base),'candidate_count':6}


def validate_reused_initial_eval(manifest):
    raise ValueError('fresh ResearchClaw base run does not reuse prior benchmark evaluation')


def validate_sample(sample):
    tokens = list(sample.tokens)
    length = int(sample.response_length)
    mask = list(sample.loss_mask or [])
    probs = sample.rollout_log_probs
    if length < 0 or length > len(tokens) or len(mask) != length:
        raise ValueError('response/mask length mismatch')
    if any(v not in (0, 1) for v in mask):
        raise ValueError('loss mask must be binary')
    if sample.remove_sample:
        if any(mask):
            raise ValueError('excluded sample still has trainable tokens')
        return {'excluded': True, 'trainable_tokens': 0}
    if probs is None or len(probs) != length:
        raise ValueError('missing/misaligned generation logprobs; never backfill for TIS')
    if any(not math.isfinite(float(p)) for p in probs):
        raise ValueError('nonfinite generation logprob')
    contexts = (sample.metadata or {}).get('clawgym_nanobot', {}).get('opsd_turns', [])
    covered = set()
    response_start = len(tokens) - length
    for turn in contexts:
        start, end = int(turn['output_start']), int(turn['output_end'])
        if not 0 <= start < end <= length:
            raise ValueError('invalid output span')
        positions = set(range(start, end))
        if positions & covered or any(mask[i] != 1 for i in positions):
            raise ValueError('duplicate output or tool/context token marked as output')
        if tokens[:response_start + start] != list(turn['prompt_ids']):
            raise ValueError('training context differs from actual generation input')
        if tokens[response_start + start:response_start + end] != list(turn['output_ids']):
            raise ValueError('training output differs from sampled token ids')
        expected = turn.get('output_log_probs')
        if expected is None or len(expected) != end-start or any(not math.isfinite(float(v)) for v in expected):
            raise ValueError('missing per-turn original output logprobs')
        if any(abs(float(a)-float(b)) > 1e-6 for a, b in zip(probs[start:end], expected)):
            raise ValueError('training logprobs differ from sampled token logprobs')
        covered |= positions
    if covered != {i for i, value in enumerate(mask) if value}:
        raise ValueError('trainable token lacks exact generation provenance')
    return {'excluded': False, 'trainable_tokens': len(covered), 'turns': len(contexts)}
