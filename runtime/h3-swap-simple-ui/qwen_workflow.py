import math


MODELS = {
    'diffusion_models': 'qwen_image_2.1_bf16.safetensors',
    'text_encoders': 'qwen3vl_8b_bf16.safetensors',
    'vae': 'qwen_image_2.1_vae_bf16.safetensors',
}


def build_qwen(prompt, seed, prefix, fields, references, source_size=None):
    width, height = int(fields.get('width', 1024)), int(fields.get('height', 1024))
    width, height = max(16, round(width / 16) * 16), max(16, round(height / 16) * 16)
    steps, batch = int(fields.get('steps', 40)), int(fields.get('batch', 1))
    cfg, denoise = float(fields.get('cfg', 1)), float(fields.get('denoise', 1))
    resolution = int(fields.get('ref_res', 1024))
    if max(width, height) > 8192 or not 1 <= steps <= 200 or not 1 <= batch <= 8:
        raise ValueError('尺寸最大 8192，步数为 1–200，张数为 1–8')
    if not math.isfinite(cfg) or not 0 <= cfg <= 100 or not math.isfinite(denoise) or not 0 <= denoise <= 1:
        raise ValueError('CFG 必须为 0–100，降噪强度必须为 0–1')
    if not 0 <= resolution <= 4096 or resolution % 32:
        raise ValueError('参考图分辨率应为 0–4096，按 32 像素对齐')
    follow = fields.get('follow_ref') == 'true' and bool(references)
    sampler, scheduler = fields.get('sampler', 'euler'), fields.get('scheduler', 'simple')
    negative = fields.get('negative', '')
    graph = {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': MODELS['diffusion_models'], 'weight_dtype': 'default'}},
        '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': MODELS['text_encoders'], 'type': 'qwen_image', 'device': 'default'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': MODELS['vae']}},
        '4': {'class_type': 'TextEncodeQwenImage21', 'inputs': {'clip': ['2', 0], 'vae': ['3', 0],
            'prompt': prompt, 'negative_prompt': negative, 'resolution': resolution}},
        '9': {'class_type': 'QwenImage21Cache', 'inputs': {'model': ['1', 0], 'device': 'auto', 'dtype': 'default'}},
    }
    for index, name in enumerate(references, 1):
        node = str(10 + index)
        graph[node] = {'class_type': 'LoadImage', 'inputs': {'image': name}}
        graph['4']['inputs'][f'images.image_{index}'] = [node, 0]
    if follow:
        if resolution:
            ratio = source_size[0] / source_size[1]
            width = max(32, round(math.sqrt(resolution * resolution * ratio) / 32) * 32)
            height = max(32, round(math.sqrt(resolution * resolution / ratio) / 32) * 32)
        else:
            width, height = (max(32, round(value / 32) * 32) for value in source_size)
        latent = ['4', 2]
        if batch > 1:
            graph['5'] = {'class_type': 'RepeatLatentBatch', 'inputs': {'samples': latent, 'amount': batch}}
            latent = ['5', 0]
    else:
        graph['5'] = {'class_type': 'EmptyFlux2LatentImage', 'inputs': {'width': width, 'height': height, 'batch_size': batch}}
        latent = ['5', 0]
    graph['6'] = {'class_type': 'KSampler', 'inputs': {'model': ['9', 0], 'positive': ['4', 0], 'negative': ['4', 1],
        'latent_image': latent, 'seed': seed, 'steps': steps, 'cfg': cfg, 'sampler_name': sampler,
        'scheduler': scheduler, 'denoise': denoise}}
    graph['7'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['6', 0], 'vae': ['3', 0]}}
    graph['8'] = {'class_type': 'SaveImage', 'inputs': {'images': ['7', 0], 'filename_prefix': prefix}}
    metadata = dict(width=width, height=height, steps=steps, cfg=cfg, batch=batch, denoise=denoise,
        sampler=sampler, scheduler=scheduler, ref_res=resolution, follow_ref=follow, negative=negative)
    return {'prompt': graph, 'final_node': '8'}, metadata
