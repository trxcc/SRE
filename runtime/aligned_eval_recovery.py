"""User policy: failed evaluation tasks score zero; keep failure provenance and continue."""
import asyncio,json,math,os
from pathlib import Path

def apply_failure_zero(output):
    if set(output.data)!={'eval-200'}:raise RuntimeError('expected planned eval-200 dataset')
    z=output.data['eval-200'];flags=z['raw_skipped'];assert len(flags)==200 and all(type(v)is bool for v in flags)
    assert len(z['rewards'])==len(z['raw_rewards'])==len(z['code_verifier_rewards'])==len(z['raw_code_verifier_rewards'])==200
    failed=[i for i,v in enumerate(flags)if v]
    for i in failed:
        for key in ['rewards','raw_rewards','code_verifier_rewards','raw_code_verifier_rewards']:z[key][i]=0.0
    assert all(isinstance(v,(int,float))and math.isfinite(v)for v in z['rewards']+z['code_verifier_rewards'])
    # Keep raw_skipped, n_collected and n_skipped truthful; denominators include failures.
    z['failed_task_score_policy']='zero';z['n_scored']=200;z['n_failed_scored_zero']=len(failed)
    m=output.metrics;m['eval/eval-200']=m['eval/eval-200-reward_avg']=sum(z['rewards'])/200
    m['eval/eval-200-code_verifier_avg']=sum(z['code_verifier_rewards'])/200
    m['eval/eval-200-n_scored']=200;m['eval/eval-200-n_failed_scored_zero']=len(failed)
    return output

def generate_rollout(args,rollout_id,data_source,evaluation=False):
    if not evaluation:raise ValueError('evaluation-only adapter')
    from clawgym_nanobot import eval_rollout as native
    m=json.loads(Path(os.environ['GEMMA_SELFREWARD_MANIFEST']).read_text())
    if m.get('eval_failure_policy')!='score_zero_and_continue':raise RuntimeError('missing user zero policy')
    return apply_failure_zero(asyncio.run(native._run_eval(args,rollout_id)))
