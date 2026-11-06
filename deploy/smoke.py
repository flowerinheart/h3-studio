"""Submit one existing H3 draft and wait until the WebUI records a playable MP4."""
import json
import os
import subprocess
import sys
import time
import urllib.request

BASE = os.environ.get('H3_URL', 'http://127.0.0.1:8188')
SECONDS = os.environ.get('H3_SMOKE_SECONDS', '1')


def post_job():
    boundary = 'h3smokeboundary'
    fields = {
        'mode': 'draft',
        'prompt': 'A ceramic cup of tea on a wooden table. Soft daylight, the camera drifts a little, no cut.',
        'seed': '7',
        'variant': 'baseline',
        'input_mode': 'text',
        'seconds': SECONDS,
        'resolution': '384',
        'aspect': '16:9',
        'skin_cleanup': 'false',
        'matte_prompt': 'false',
        'extra_loras': '[]',
        'attention_backend': 'native',
    }
    body = b''
    for key, value in fields.items():
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
        ).encode()
    body += f'--{boundary}--\r\n'.encode()
    request = urllib.request.Request(
        BASE + '/simple-swap/api/jobs', data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def get_json(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as response:
        return json.load(response)


def main():
    options = get_json('/simple-swap/api/options')
    if 'draft' not in options.get('workflows', []):
        sys.exit('WebUI 没有现有的 draft 模式: ' + json.dumps(options.get('workflows')))
    job = post_job()
    job_id = job['id']
    print('submitted', job_id, flush=True)
    deadline = time.time() + float(os.environ.get('H3_SMOKE_TIMEOUT', '14400'))
    while time.time() < deadline:
        current = get_json(f'/simple-swap/api/jobs/{job_id}')
        status = current.get('status')
        print(status, current.get('stage'), flush=True)
        if status == 'completed':
            video = current.get('video_path')
            if not video:
                sys.exit('任务完成但没有 video_path')
            probe = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video],
                check=False, capture_output=True, text=True)
            if probe.returncode != 0 or 'video' not in probe.stdout:
                sys.exit('ffprobe 失败: ' + probe.stderr[-400:])
            history = get_json('/history/' + job_id)
            if job_id not in history:
                sys.exit('ComfyUI 历史里没有这条 prompt id')
            print('VIDEO_OK', video)
            print('PROMPT_OK', job_id)
            return
        if status in ('error', 'interrupted'):
            sys.exit(current.get('error') or status)
        time.sleep(20)
    sys.exit('超时')


if __name__ == '__main__':
    main()
