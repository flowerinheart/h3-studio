import json
import threading
import time
from pathlib import Path

WORKFLOW_MODES = ('face', 'character', 'flex_character', 'quality_character', 'draft', 'refine', 'qwen', 'long')


class JobStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {p.stem: json.loads(p.read_text()) for p in self.root.glob('*/*.json')}
        for path in self.root.glob('*.json'):
            job = json.loads(path.read_text())
            if job['id'] not in self.jobs:
                self.save(job)
            path.unlink()

    def path(self, job):
        mode = job.get('mode', 'face')
        if mode not in WORKFLOW_MODES or Path(job['id']).name != job['id']:
            raise ValueError('任务工作流或编号不合法')
        return self.root / mode / (job['id'] + '.json')

    def list(self, mode=None):
        with self.lock:
            return sorted((job for job in self.jobs.values() if mode is None or job.get('mode', 'face') == mode),
                          key=lambda job: job.get('created_at', 0), reverse=True)

    def delete(self, job_id, mode):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job.get('mode', 'face') != mode:
                raise KeyError(job_id)
            if job['status'] not in ('completed', 'error', 'interrupted'):
                raise ValueError('执行中的任务不能删除历史，请先取消此任务')
            self.path(job).unlink(missing_ok=True)
            del self.jobs[job_id]

    def save(self, job):
        with self.lock:
            path = self.path(job)
            profile = self.root.parent / 'profiles' / job.get('mode', 'face') / (job['id'] + '.json')
            job['profile_path'] = str(profile.resolve())
            job['profile_version'] = 1
            if 'asset_paths' not in job:
                input_root = Path(__file__).resolve().parents[2] / 'input'
                names = [job.get(key) for key in ('video_input', 'image_input', 'last_image_input')]
                for files in job.get('references', {}).values():
                    names.extend(files)
                job['asset_paths'] = {name: str((input_root / name).resolve()) for name in names if name}
            if 'parameters' not in job:
                job['parameters'] = {key: value for key, value in job.items() if key in (
                    'mode','seed','steps','resolution','width','height','seconds','frame_count',
                    'variant','aspect','fit','target_width','target_height','blend_frames','face_region','face_strength','skin_cleanup','skin_strength','matte_prompt','extra_loras','extra_lora_precision','extra_lora_triggers','attention_backend','attention_tau','attention_precision','project','chain','continuation_frames')}
                job['parameters']['model_nodes'] = {key: node['inputs'] for key, node in job.get('graph', {}).items()
                    if any(field in node.get('inputs', {}) for field in ('unet_name','clip_name','vae_name','lora_name','precision'))}
            path.parent.mkdir(parents=True, exist_ok=True)
            self.jobs[job['id']] = job
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(job, ensure_ascii=False))
            temporary.replace(path)
            profile.parent.mkdir(parents=True, exist_ok=True)
            temporary = profile.with_suffix('.tmp')
            temporary.write_text(json.dumps(job, ensure_ascii=False))
            temporary.replace(profile)

    def load_imports(self):
        with self.lock:
            for path in list(self.root.glob('*/*.json')) + list(self.root.glob('*.json')):
                if path.stem not in self.jobs:
                    self.save(json.loads(path.read_text()))
                if path.parent == self.root:
                    self.save(self.jobs[path.stem])
                    path.unlink()

    def event(self, event, data, sid=None):
        if not isinstance(data, dict):
            return
        with self.lock:
            job = self.jobs.get(data.get('prompt_id'))
            if job is None and sid and not data.get('prompt_id'):
                job = next((j for j in self.jobs.values() if j.get('client_id') == sid
                            and j['status'] in ('queued', 'running')), None)
            if not job or job['status'] in ('completed', 'error', 'interrupted'):
                return
            now = time.time()
            if event == 'execution_start':
                job.update(status='running', started_at=now)
            elif event == 'executing' and data.get('node') is not None:
                previous = job.get('node')
                if previous is not None:
                    job['timings'].append({'node': previous, 'seconds': now - job['stage_started_at']})
                job.update(node=str(data['node']), stage_started_at=now, step=0, total=0, layer=0, layers=0)
            elif event == 'progress':
                job.update(step=data.get('value', 0), total=data.get('max', 0))
            elif event == 'h3_webui_layer':
                job.update(layer=data['layer'], layers=data['layers'], layer_updated_at=now)
            elif event in ('execution_error', 'execution_interrupted'):
                job.update(status='error' if event == 'execution_error' else 'interrupted',
                           error=data.get('exception_message', '任务已中止'), finished_at=now)
            elif event == 'execution_success':
                if job.get('node') is not None:
                    job['timings'].append({'node': job['node'], 'seconds': now - job['stage_started_at']})
                job['status'] = 'finalizing'
            else:
                return
            self.save(job)
