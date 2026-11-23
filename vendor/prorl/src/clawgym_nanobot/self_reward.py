"""One GT anchor plus seven frozen-policy self-reward candidates (SRE, K=7).

Only training submissions opt into capture. Eval retains its configured evaluator.
No pending or invalid reward may contribute tokens or group-normalization statistics.
"""
from __future__ import annotations
import asyncio,hashlib,json,math,os,shutil,time,uuid
from pathlib import Path
from typing import Any
from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult

STREAM_PROTOCOL = "self_reward_stream_v1"
GROUP_SIZE_BY_MODE = {
    'self_reward_7plus1_train7': 8,
}

def configured_group_size(cfg):
    expected = GROUP_SIZE_BY_MODE.get(cfg['mode'])
    if expected is None:
        raise ValueError('Unknown self-reward mode')
    count = cfg.get('candidate_count', expected)
    if type(count) is not int or count != expected:
        raise ValueError('self-reward mode and candidate_count disagree')
    declared = os.environ.get('CLAWGYM_SELF_REWARD_GROUP_SIZE')
    if declared is not None and declared != str(count):
        raise ValueError('self-reward environment and configuration group sizes disagree')
    return count

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False));tmp.chmod(0o600);tmp.replace(path)

def _scheduler_fields(*metadata_rows):
    result={}
    for metadata in metadata_rows:
        if not isinstance(metadata,dict):continue
        stack=[metadata]
        while stack:
            row=stack.pop()
            for key in ('rollout_step','policy_version','group_id','task_index'):
                if key in row and key not in result:result[key]=row[key]
            stack.extend(value for value in row.values() if isinstance(value,dict))
    return result

def configuration():
    name=os.environ.get('CLAWGYM_SELF_REWARD_CONFIG')
    if not name:return None
    cfg=json.loads(Path(name).read_text())
    if not cfg.get('enabled',False):return None
    if cfg.get('mode') not in GROUP_SIZE_BY_MODE:
        raise ValueError('Unknown self-reward mode')
    configured_group_size(cfg)
    return cfg

def _stream_group_identity(cfg, *, rollout_step, group_index):
    payload = {
        'protocol': STREAM_PROTOCOL,
        'model': cfg['profile']['model'],
        'snapshot_root': cfg['snapshot_root'],
        'rollout_step': int(rollout_step),
        'group_index': int(group_index),
    }
    # Preserve legacy eight-slot group identities; isolate every smaller group size.
    if configured_group_size(cfg) != 8:
        payload['candidate_count'] = configured_group_size(cfg)
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return key, payload

def publish_candidate(
    *,
    base_sample,
    task_id,
    task_dir,
    session_id,
    candidate_snapshot=None,
    valid,
    status,
):
    """Publish one completed trajectory without waiting for sibling trajectories.

    The stable group key uses scheduler identity rather than candidate contents,
    so every rollout worker can independently append its candidate.  The CPU
    reward service may therefore start anchor GT/rubric and candidate GT/self
    scoring while other trajectories are still rolling out.
    """
    cfg = configuration()
    if cfg is None:
        return None
    sample_index = int(getattr(base_sample, 'index'))
    group_index = int(getattr(base_sample, 'group_index'))
    metadata = getattr(base_sample, 'metadata', None)
    if not isinstance(metadata, dict) or metadata.get('rollout_step') is None:
        raise ValueError(
            'streaming self-reward requires the outer rollout_step stamped by the rollout scheduler'
        )
    rollout_step = int(metadata['rollout_step'])
    group_size = configured_group_size(cfg)
    candidate_id = sample_index % group_size
    key, identity = _stream_group_identity(
        cfg, rollout_step=rollout_step, group_index=group_index,
    )
    dest = Path(cfg['queue_root']) / key
    group = {
        **identity,
        'group_id': key,
        'task_id': str(task_id),
        'task_dir': str(task_dir),
        'profile': cfg['profile'],
        'candidate_count': group_size,
    }
    group_path = dest / 'group.json'
    if group_path.exists():
        existing = json.loads(group_path.read_text())
        comparable = {k: existing.get(k) for k in group}
        if comparable != group:
            raise ValueError(f'self-reward stream group identity mismatch: {key}')
    else:
        atomic(group_path, group)
    if candidate_snapshot:
        candidate = json.loads(Path(candidate_snapshot).read_text())
    else:
        candidate = {
            'session_id': str(session_id), 'workspace': '', 'run_dir': '',
            'initial_files': [], 'final_message': '',
        }
    candidate.update({
        'candidate_id': candidate_id,
        'trajectory_index': sample_index,
        'session_id': str(session_id),
        'valid': bool(valid),
        'status': str(status),
        'task_dir': str(task_dir),
        'published_at': time.time(),
    })
    candidate_path = dest / 'candidates' / f'{candidate_id}.json'
    if candidate_path.exists():
        existing = json.loads(candidate_path.read_text())
        # A completed trajectory has one immutable terminal publication.  A
        # retry with a different session must use a new snapshot_root/attempt.
        if existing.get('session_id') != candidate['session_id']:
            raise ValueError(
                f'self-reward candidate slot already belongs to another session: {candidate_path}'
            )
    else:
        atomic(candidate_path, candidate)
    candidate_paths = [dest / 'candidates' / f'{index}.json' for index in range(group_size)]
    closed_path = dest / 'closed.json'
    if all(path.exists() for path in candidate_paths) and not closed_path.exists():
        # This is a per-group terminal marker, written by whichever rollout
        # finishes last.  It is intentionally independent of the outer
        # rollout batch, so the service can finish and persist the response
        # before unrelated groups complete.
        atomic(closed_path, {
            'protocol': STREAM_PROTOCOL,
            'group_id': key,
            'closed_at': time.time(),
            'candidate_ids': list(range(group_size)),
            'rollout_step': rollout_step,
        })
    return {
        'protocol': STREAM_PROTOCOL,
        'group_id': key,
        'candidate_id': candidate_id,
        'candidate_path': str(candidate_path),
        'rollout_step': rollout_step,
        'group_index': group_index,
    }

def _stream_info(metadata):
    if not isinstance(metadata, dict):
        return None
    clawgym = metadata.get('clawgym_nanobot')
    if not isinstance(clawgym, dict):
        return None
    value = clawgym.get('self_reward_stream')
    return value if isinstance(value, dict) else None

def configure_payload(payload):
    cfg=configuration()
    if cfg is None:return
    count = configured_group_size(cfg)
    if type(payload.get('num_samples')) is not int or payload['num_samples'] != count:
        raise ValueError(f'self-reward mode requires {count} rollout slots')
    old=payload['evaluator']
    payload['evaluator']={'strategy':'clawgym_nanobot.self_reward:CaptureEvaluator','refresh_runtime':False,
        'config':{'task_dir':old['config']['task_dir'],'snapshot_root':cfg['snapshot_root'],
                  'dataset':cfg['profile']['dataset']}}

class CaptureEvaluator(BaseTrajectoryEvaluator):
    require_completed_session=True
    def __init__(self,*,task_dir,snapshot_root,dataset='researchclaw',**unused):
        self.task_dir=Path(task_dir);self.root=Path(snapshot_root);self.dataset=dataset
    async def evaluate(self,trajectory,**context):
        return await asyncio.to_thread(self.capture,trajectory,context)
    def capture(self,trajectory,context):
        from .self_reward_snapshot import collect
        sid=str(context['session_id'])
        if not sid or Path(sid).name!=sid:raise ValueError('Unsafe session identity')
        # A fresh snapshot is required even when a restarted solver reuses session IDs.
        dest=self.root/(sid+'-'+uuid.uuid4().hex);marker=dest/'candidate.json'
        if marker.exists():candidate=json.loads(marker.read_text())
        else:
            workspace=Path(context['session_dir'])/'workspace'
            target=dest/'workspace'
            if self.dataset == 'wsclaw':
                excluded={'sessions','.nanobot','.git','reward','_judge','judge_workspace',
                          'reference_answer','__pycache__'}
                def ignore(_folder,names):
                    return {name for name in names if name in excluded or name.startswith('.nanobot-no-web-bin')}
                shutil.copytree(workspace,target,symlinks=True,ignore=ignore)
                audit={'scope':'complete_task_facing_workspace_mount_view_v1',
                       'input_mounts_preserved':True}
            else:
                mapping,audit=collect(workspace,[])
                target.mkdir(parents=True,exist_ok=True)
                for relative,item in mapping.items():
                    out=target/relative;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(item['path'],out)
            session=workspace/'sessions'/f'{sid}.jsonl'
            if not session.is_file():raise ValueError('Completed rollout is missing its original session transcript')
            out=target/'sessions'/session.name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(session,out)
            messages=[json.loads(x) for x in session.read_text().splitlines() if x.strip()]
            final=next((x.get('content') or '' for x in reversed(messages) if x.get('role')=='assistant'),'')
            candidate={'session_id':sid,'workspace':str(target),'run_dir':str(dest),'initial_files':[],
                'final_message':final,'valid':True,'status':'completed','task_dir':str(self.task_dir)}
            atomic(dest/'snapshot-audit.json',audit);atomic(marker,candidate)
        return EvalResult(outcome_reward=None,metadata={'self_reward_pending':True,'candidate_snapshot':str(marker)})

def _group_summary(packet):
    metrics=packet.get('metrics') if isinstance(packet,dict) else None
    return {'group_id':packet.get('group_id'),'metrics':metrics if isinstance(metrics,dict) else {}}

def _result_update(result,score,valid,trainable,reason,row,group):
    if not isinstance(score,(int,float)) or isinstance(score,bool) or not math.isfinite(score) or not 0<=score<=1:
        raise ValueError('Nonfinite or out-of-range training reward')
    result.metadata={**result.metadata,
        'self_reward_finalized':True,'self_reward_valid':valid,
        'self_reward_trainable':trainable,'self_reward_reason':reason,
        'raw_reward':float(score),
        'raw_gt_reward':row.get('raw_gt_reward'),
        'raw_self_reward':row.get('raw_self_reward'),
        'self_reward_repeat_count':row.get('accepted_repeats'),
        'self_reward_group':group}
    traces=[]
    for trace in result.trajectory.traces:
        updates={'reward':float(score)}
        if not valid or not trainable:updates['loss_mask']=[0]*len(trace.loss_mask)
        traces.append(trace.model_copy(update=updates))
    result.trajectory=result.trajectory.model_copy(update={'traces':traces,'metadata':{
        **result.trajectory.metadata,'evaluation':{**result.trajectory.metadata.get('evaluation',{}),'outcome_reward':float(score),'self_reward_pending':False,'reward_source':reason,'raw_reward':float(score),'raw_gt_reward':row.get('raw_gt_reward'),'raw_self_reward':row.get('raw_self_reward'),'self_reward_valid':valid,'self_reward_trainable':trainable,'self_reward_repeat_count':row.get('accepted_repeats')},'self_reward':{**row,'score':score,'valid':valid,'trainable':trainable,'reason':reason},'self_reward_group':group}})

def apply_rewards(task_result,packet):
    rows=packet['candidates']
    group=_group_summary(packet)
    if set(rows)!=set(str(i) for i in range(len(task_result.results))):raise ValueError('Reward candidate IDs mismatch')
    for i,result in enumerate(task_result.results):
        row=rows[str(i)]
        if row['session_id']!=result.session_id:raise ValueError('Reward session identity mismatch')
        _result_update(result,row['score'],row['valid'],row.get('trainable',row['valid']),row['reason'],row,group)
    cfg=configuration()
    if cfg and rows['0']['valid'] and any(row.get('trainable',row['valid']) for key,row in rows.items() if key!='0'):
        atomic(Path(os.environ['CLAWGYM_SELF_REWARD_CONFIG']).parent/'reward-ready.json',{'at':time.time(),'group_id':packet.get('group_id'),'anchor_and_self_rewards':True})
    return task_result

def _failed_rollout_packet(candidates, *, group_id=None):
    """Mask a group that has no usable rollout artifact before reward work."""
    return {
        'group_id': group_id,
        'metrics': {},
        'candidates': {
            str(i): {
                'session_id': candidate['session_id'],
                'score': 0.0, 'valid': False, 'trainable': False,
                'reason': 'rollout_failed_or_timeout',
                'raw_reward': 0.0, 'raw_gt_reward': None, 'raw_self_reward': None,
                'accepted_repeats': 0,
            }
            for i, candidate in enumerate(candidates)
        },
    }

async def finalize_group(task_result):
    cfg=configuration()
    if cfg is None:return task_result
    count = configured_group_size(cfg)
    if len(task_result.results)!=count:raise ValueError(f'Reward group must contain {count} logical rollout slots')
    candidates=[]
    for i,result in enumerate(task_result.results):
        meta=result.trajectory.metadata.get('evaluation',{})
        marker=meta.get('candidate_snapshot')
        if marker:
            record=json.loads(Path(marker).read_text());record['candidate_id']=i
        else:record={'candidate_id':i,'session_id':result.session_id,'valid':False,'status':str(result.status),'workspace':'','final_message':''}
        candidates.append(record)
    if not any(candidate.get('task_dir') for candidate in candidates):
        return apply_rewards(task_result, _failed_rollout_packet(candidates, group_id=task_result.task_id))
    key=hashlib.sha256(json.dumps([cfg['profile']['model'], cfg['snapshot_root'], candidates],sort_keys=True).encode()).hexdigest()
    dest=Path(cfg['queue_root'])/key
    scheduler=_scheduler_fields(getattr(task_result,'metadata',None),*(getattr(r,'metadata',None) for r in task_result.results),*(getattr(r.trajectory,'metadata',None) for r in task_result.results))
    request={'group_id':key,'task_id':task_result.task_id,'candidate_count':count,'candidates':candidates,'profile':cfg['profile'],
        'scheduler':scheduler,'rollout_step':scheduler.get('rollout_step'),
        'task_dir':next((x['task_dir'] for x in candidates if x.get('task_dir')),None),'model':cfg['profile']['model']}
    response=dest/'response.json'
    if not (dest/'request.json').exists():atomic(dest/'request.json',request)
    deadline=time.monotonic()+float(cfg.get('group_timeout',7500))
    while not response.exists():
        if time.monotonic()>deadline:
            packet={'candidates':{str(i):{'session_id':r.session_id,'score':0.0,'valid':False,'trainable':False,'reason':'reward_service_timeout'} for i,r in enumerate(task_result.results)}}
            atomic(dest/'client-timeout.json',packet)
            return apply_rewards(task_result,packet)
        await asyncio.sleep(1)
    return apply_rewards(task_result,json.loads(response.read_text()))

def finalize_legacy_rewards(args,samples):
    """Finalize complete legacy batches; no blocking barrier inside generator slots."""
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor
    from .reward_post_process import grpo_normalize_by_group_index
    cfg=configuration()
    if cfg is None:return grpo_normalize_by_group_index(args,samples)
    count = configured_group_size(cfg)
    groups=defaultdict(lambda:defaultdict(list))
    for s in samples:
        # index is the stable sampled-trajectory identity, shared by its trace pieces.
        groups[s.group_index][s.index].append(s)
    def one(item):
        group_id,trajectories=item
        ordered=sorted(trajectories.items())
        if len(ordered)!=count:raise ValueError(f'Legacy group {group_id} has {len(ordered)} trajectories; expected {count}')
        cs=[]
        stream_rows=[]
        for i,(idx,pieces) in enumerate(ordered):
            m=(pieces[0].metadata or {}).get('clawgym_nanobot',{});marker=m.get('candidate_snapshot')
            stream=_stream_info(pieces[0].metadata or {})
            valid=any(not getattr(s,'remove_sample',False) and any(s.loss_mask or []) for s in pieces)
            if marker and valid:c=json.loads(Path(marker).read_text())
            else:c={'session_id':m.get('session_id') or f'{group_id}-{idx}','valid':False,'status':'failed','workspace':'','final_message':''}
            candidate_id=int(stream['candidate_id']) if stream is not None else i
            c.update(candidate_id=candidate_id, valid=bool(valid), trajectory_index=int(idx))
            cs.append(c);stream_rows.append(stream)
        streaming=(
            all(row is not None and row.get('protocol')==STREAM_PROTOCOL for row in stream_rows)
            and len({row['group_id'] for row in stream_rows})==1
            and sorted(row['candidate_id'] for row in stream_rows)==list(range(count))
        )
        if any(row is not None for row in stream_rows) and not streaming:
            raise ValueError('Incomplete or inconsistent self-reward stream identities')
        if streaming:
            key=str(stream_rows[0]['group_id']);dest=Path(cfg['queue_root'])/key
            by_id={int(c['candidate_id']):c for c in cs}
            for candidate_id,candidate in by_id.items():
                path=dest/'candidates'/f'{candidate_id}.json'
                if not path.exists():atomic(path,candidate)
            # The client-side group timeout starts only after every rollout is
            # terminal.  Slow rollout generation therefore never consumes the
            # judge/scoring deadline.
            if not (dest/'closed.json').exists():
                atomic(dest/'closed.json',{
                    'protocol':STREAM_PROTOCOL,'group_id':key,'closed_at':time.time(),
                    'candidate_ids':list(range(count)),'rollout_step':next((row.get('rollout_step') for row in stream_rows if row),None),
                })
        else:
            key=hashlib.sha256(json.dumps([cfg['profile']['model'], cfg['snapshot_root'], cs],sort_keys=True).encode()).hexdigest();dest=Path(cfg['queue_root'])/key
        first_meta=(ordered[0][1][0].metadata or {}) if ordered and ordered[0][1] else {}
        scheduler=_scheduler_fields(first_meta,first_meta.get('clawgym_nanobot'),first_meta.get('polar'))
        task_dir=next((c['task_dir'] for c in cs if c.get('task_dir')),None)
        if task_dir is None:
            packet=_failed_rollout_packet(cs, group_id=key)
        else:
            if not streaming and not (dest/'request.json').exists():atomic(dest/'request.json',{'group_id':key,'task_id':str(group_id),'candidate_count':count,'candidates':cs,'profile':cfg['profile'],'model':cfg['profile']['model'],'scheduler':scheduler,'rollout_step':scheduler.get('rollout_step'),'task_dir':task_dir})
            deadline=time.monotonic()+cfg.get('group_timeout',7500)
            while not (dest/'response.json').exists() and time.monotonic()<deadline:time.sleep(1)
            if (dest/'response.json').exists():packet=json.loads((dest/'response.json').read_text())
            else:packet={'group_id':key,'metrics':{},'candidates':{str(i):{'session_id':c['session_id'],'score':0.,'valid':False,'trainable':False,'reason':'reward_service_timeout'} for i,c in enumerate(cs)}}
        rows=packet['candidates'];group=_group_summary(packet)
        if set(rows) != {str(i) for i in range(count)}:
            raise ValueError('Legacy reward candidate IDs mismatch')
        cs_by_id={int(c['candidate_id']):c for c in cs}
        for i,(_,pieces) in enumerate(ordered):
            candidate_id=int(stream_rows[i]['candidate_id']) if streaming else i
            row=rows[str(candidate_id)]
            if row['session_id']!=cs_by_id[candidate_id]['session_id']:raise ValueError('Legacy reward identity mismatch')
            score=row['score']
            if isinstance(score,bool) or not isinstance(score,(float,int)) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid self reward')
            for s in pieces:
                s.reward={'score':float(score)}
                s.metadata['self_reward']={**row,'group_id':key}
                cm=s.metadata.setdefault('clawgym_nanobot',{})
                cm.update(reward=float(score),self_reward_pending=False,reward_source=row['reason'],
                    raw_reward=float(score),raw_gt_reward=row.get('raw_gt_reward'),
                    raw_self_reward=row.get('raw_self_reward'),self_reward_valid=row['valid'],
                    self_reward_trainable=row.get('trainable',row['valid']),
                    self_reward_repeat_count=row.get('accepted_repeats'),
                    self_reward_group=group)
                cm.pop('code_verifier_score',None)
                trainable=row.get('trainable',row['valid'])
                if not row['valid'] or not trainable:
                    s.loss_mask=[0]*len(s.loss_mask)
                    # Invalid trajectories are removed.  The valid anchor in
                    # train5/train7 mode stays in audit output but has no policy loss.
                    if not row['valid']:s.remove_sample=True
                    # A rejected trajectory cannot remain an eligible linear
                    # credit chain: Slime correctly rejects zero-action chains.
                    # Keep the audit metadata, but exclude every sibling piece.
                    cross=cm.get('cross_trajectory')
                    if isinstance(cross,dict):
                        cm['cross_trajectory']={**cross,'eligible':False,
                            'self_reward_mask_reason':row['reason']}
    with ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(one,groups.items()))
    if any((s.metadata or {}).get('self_reward',{}).get('trainable') for s in samples):
        atomic(Path(os.environ['CLAWGYM_SELF_REWARD_CONFIG']).parent/'reward-ready.json',{'at':time.time(),'anchor_and_self_rewards':True})
    return grpo_normalize_by_group_index(args,samples)
