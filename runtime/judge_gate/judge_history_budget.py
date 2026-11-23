"""Enforce the existing judge history budget without restoring oversized history."""
def install(runner_module):
    cls=runner_module.AgentRunner
    if getattr(cls,'_researchclaw_bounded_history',False):return
    original=cls._snip_history
    def bounded(self,spec,messages):
        selected=original(self,spec,messages)
        if not messages or not spec.context_window_tokens:return selected
        maximum=getattr(getattr(self.provider,'generation',None),'max_tokens',4096)
        output=spec.max_tokens if isinstance(spec.max_tokens,int)else maximum if isinstance(maximum,int)else 4096
        budget=spec.context_block_limit or(spec.context_window_tokens-output-runner_module._SNIP_SAFETY_BUFFER)
        if budget<=0:return selected
        def size(rows):return runner_module.estimate_prompt_tokens_chain(self.provider,spec.model,rows,spec.tools.get_definitions())[0]
        if size(selected)<=budget:return selected
        system=[dict(m)for m in messages if m.get('role')=='system']
        other=[dict(m)for m in messages if m.get('role')!='system']
        first=next((i for i,m in enumerate(other)if m.get('role')=='user'),None)
        if first is None:raise RuntimeError('judge bounded history requires original user prompt')
        anchor=other[first];base=system+[anchor]
        if size(base)>budget:raise RuntimeError('judge fixed prompts exceed configured history budget')
        for start in range(first+1,len(other)+1):
            tail=other[start:];tail=tail[runner_module.find_legal_message_start(tail):]
            candidate=base+tail
            if size(candidate)<=budget:return candidate
        return base
    cls._snip_history=bounded;cls._researchclaw_bounded_history=True
