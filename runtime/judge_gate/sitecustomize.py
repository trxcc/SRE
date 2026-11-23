"""Preserve existing startup hooks; install judge-only global request cap64."""
import os,sys,runpy
from pathlib import Path
here=Path(__file__).resolve().parent
for entry in sys.path:
    try:p=Path(entry or '.').resolve()/'sitecustomize.py'
    except (OSError,RuntimeError):continue
    if p.is_file()and p.resolve()!=Path(__file__).resolve():
        runpy.run_path(str(p),run_name='_researchclaw_previous_sitecustomize');break
if os.environ.get('RESEARCHCLAW_AGENTIC_JUDGE')=='1':
    try:
        from judge_request_gate32 import install
        from nanobot.providers.openai_compat_provider import OpenAICompatProvider
        install(OpenAICompatProvider)
        from judge_history_budget import install as install_history
        import nanobot.agent.runner as runner_module
        install_history(runner_module)
    except BaseException:
        os.write(2,b'judge64 request gate initialization failed\n');os._exit(86)
