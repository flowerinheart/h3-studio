"""Shared default skin preparation and additional H3 LoRA composition."""
import json
import math
import struct
from pathlib import Path

PEOPLE_MODES = ('face', 'character', 'flex_character', 'quality_character', 'draft', 'long')


def rocm_gfx1151():
    """True only on the validated AMD gfx1151 box. CUDA and other GPUs use PyTorch attention."""
    try:
        import torch
    except Exception:
        return False
    if getattr(torch.version, 'hip', None) is None or not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), 'gcnArchName', '') or ''
    return arch.split(':')[0] == 'gfx1151'
MATTE = 'Natural soft matte skin texture, fine pores, restrained skin specular highlights and zero oily shine. Preserve natural skin detail, original facial proportions, lip color, expression and lighting direction; avoid plastic skin and beauty smoothing.'
TRIGGERS = {'h3-realism-people-t2v-i2v-r2v.safetensors': 'r34l1sm'}
EXCLUDED = ('pika_', 'turbo_', 'regenerate', 'sharp_curve', 'SS_FaceSwap_')


def lora_catalog(folder_paths):
    items = []
    presets = json.loads(Path(__file__).with_name('lora_presets.json').read_text())
    for name in folder_paths.get_filename_list('loras'):
        if not name.endswith('.safetensors') or any(s.lower() in name.lower() for s in EXCLUDED):
            continue
        try:
            with open(folder_paths.get_full_path('loras', name), 'rb') as f:
                size = struct.unpack('<Q', f.read(8))[0]
                if size > 16_000_000:
                    continue
                header = json.loads(f.read(size))
            tensors = {k: v for k, v in header.items() if k != '__metadata__'}
            if not any(('blocks.' in k or 'blocks_' in k) and ('attn.qkv_proj' in k or 'attn_qkv_proj' in k) for k in tensors):
                continue
            modes = list(PEOPLE_MODES) + ['refine']
            if 'ref2' in name.lower():
                modes.remove('refine')
                modes.remove('draft')  # Dedicated motion adapters need a reference-video workflow.
            preset = presets.get(name, {})
            items.append({'name': name, 'label': 'Realism People · 写实人物' if name in TRIGGERS else name,
                          'modes': modes, 'default_strength': 0.5, 'trigger': TRIGGERS.get(name, ''),
                          'weight_precision': sorted({v['dtype'] for v in tensors.values()}), **preset})
        except (OSError, ValueError, KeyError, struct.error):
            continue
    return items


def settings(fields, mode, catalog):
    enabled = mode in PEOPLE_MODES
    flags = {}
    for key in ('skin_cleanup', 'matte_prompt'):
        value = fields.get(key, 'true' if enabled else 'false')
        if value not in ('true', 'false'):
            raise ValueError('无效的皮肤处理选项')
        flags[key] = value == 'true'
    if not enabled and any(flags.values()):
        raise ValueError('该流程不支持自动皮肤处理')
    try:
        strength = float(fields.get('skin_strength', '0.9'))
        selected = json.loads(fields['extra_loras']) if 'extra_loras' in fields else [{'name': x['name'], 'strength': x['default_strength']} for x in catalog if mode in x['modes'] and x['name'] in ('h3-realism-people-t2v-i2v-r2v.safetensors', 'h3_camera_motion_v1_3000_pruned.safetensors')]
    except (ValueError, TypeError):
        raise ValueError('无效的皮肤强度或 LoRA 列表')
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('去油强度须在0至1之间')
    if not isinstance(selected, list) or len(selected) > 8:
        raise ValueError('最多选择8个额外 LoRA')
    available = {x['name']: x for x in catalog}
    seen = set()
    for item in selected:
        if not isinstance(item, dict) or set(item) != {'name', 'strength'}:
            raise ValueError('LoRA 必须包含文件名与强度')
        name = item['name']
        if not isinstance(name, str) or name not in available or mode not in available[name]['modes']:
            raise ValueError('LoRA 未安装或不兼容该流程：' + str(name))
        if name in seen:
            raise ValueError('不能重复选择同一个 LoRA')
        if isinstance(item['strength'], bool) or not isinstance(item['strength'], (int, float)) or not math.isfinite(item['strength']) or not -2 <= item['strength'] <= 2:
            raise ValueError('LoRA 强度须在-2至2之间')
        seen.add(name)
    if rocm_gfx1151():
        backend = fields.get('attention_backend', 'ck' if mode == 'qwen' else 'sol-hip')
        if backend not in ('ck', 'sol-hip') or (mode == 'qwen' and backend != 'ck'):
            raise ValueError('该流程不支持所选注意力算子')
    else:
        # Thor CUDA and any non-gfx1151 host have no HIP/CK binary. Keep the same graph and run native attention.
        backend = 'native'
    try:
        tau = float(fields.get('attention_tau', '0.5'))
    except (TypeError, ValueError):
        raise ValueError('Sol 稀疏强度必须是数字')
    if not math.isfinite(tau) or not 0 <= tau <= 6:
        raise ValueError('Sol 稀疏强度须在0至6之间')
    if mode == 'qwen':
        backend = 'native'
    precision = {'native': 'PyTorch native', 'sol-hip': 'BF16 IO / INT8 sparse internals', 'ck': 'dense BF16 CK'}[backend]
    if mode == 'qwen':
        precision = 'Qwen native (unchanged)'
    return {**flags, 'attention_backend': backend, 'attention_tau': tau if mode != 'qwen' else None,
            'attention_precision': precision, 'skin_strength': strength, 'extra_loras': selected,
            'extra_lora_precision': {x['name']: available[x['name']]['weight_precision'] for x in selected},
            'extra_lora_triggers': {x['name']: available[x['name']].get('trigger', '') for x in selected}}


def effective_prompt(prompt, config):
    triggers = config.get('extra_lora_triggers', TRIGGERS)
    prefixes = list(dict.fromkeys(triggers[x['name']] for x in config['extra_loras'] if x['strength'] != 0 and triggers.get(x['name']) and triggers[x['name']] not in prompt))
    if config['matte_prompt'] and MATTE not in prompt:
        prompt = prompt.rstrip() + '\n\n' + MATTE
    return (', '.join(prefixes) + ', ' if prefixes else '') + prompt


def apply_enhancements(workflow, config, prefix):
    graph = workflow['prompt']
    if config.get('attention_backend') == 'native':
        for node in graph.values():
            if node.get('class_type') in ('H3CKAttention', 'H3SolHIPAttention'):
                node['class_type'] = 'H3NativeAttention'
                node['inputs'] = {'model': node['inputs']['model']}
    elif config.get('attention_backend', 'ck') == 'sol-hip':
        attention_nodes = [node for node in graph.values() if node['class_type'] == 'H3CKAttention']
        if not attention_nodes:
            raise ValueError('执行图没有可替换的H3注意力节点')
        for node in attention_nodes:
            node['class_type'] = 'H3SolHIPAttention'
            node['inputs'] = {'model': node['inputs']['model'], 'tau': config['attention_tau']}
            if '_meta' in node:
                node['_meta']['title'] = 'Sol-Attn HIP Attention'
    if config['skin_cleanup']:
        # Process all source frames, not only frame zero: avoids a one-frame color jump.
        loaders = [(key, 0) for key, node in list(graph.items()) if node['class_type'] in ('LoadImage', 'VHS_LoadVideo', 'GetVideoComponents')]
        for index, (source, output) in enumerate(loaders):
            key = 'skin_prepare_' + str(index)
            for node in graph.values():
                for field, value in node['inputs'].items():
                    if value == [source, output]:
                        node['inputs'][field] = [key, 0]
            graph[key] = {'class_type': 'H3WebUISkinMatte', 'inputs': {
                'images': [source, output], 'strength': config['skin_strength'],
                'filename_prefix': prefix + '_' + str(index)}}
    if config['extra_loras']:
        # Shared base model projections; keep existing task/refiner/acceleration adapters downstream.
        bases = [key for key, node in graph.items() if node['class_type'] == 'UNETLoader']
        if len(bases) != 1:
            raise ValueError('该执行图没有唯一的H3模型加载入口')
        base = bases[0]
        model = [base, 0]
        chain = {}
        for index, item in enumerate(config['extra_loras']):
            key = 'extra_lora_' + str(index)
            chain[key] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {
                'model': model, 'lora_name': item['name'], 'strength_model': item['strength']}}
            model = [key, 0]
        for node in graph.values():
            for field, value in node['inputs'].items():
                if value == [base, 0]:
                    node['inputs'][field] = model
        graph.update(chain)
    return workflow
