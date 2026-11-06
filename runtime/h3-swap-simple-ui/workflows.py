"""Split the locally verified CK 768p -> 2K graph at its saved latent."""
import copy
import json
from pathlib import Path

TURBO = 'pika_h3_flex_turbo_4step_fl2va.safetensors'
REF_MODEL = 'minimax_h3_ref2va_pruned_bf16.safetensors'
FACE_REFINER = 'minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors'
TEMPLATE = Path(__file__).resolve().parent / '768p-to-2k.api.json'
ACCEL_STEPS = {'baseline':20, 'pika4':4, 'pika6':6, 'turbo8':8}

def acceleration_file(variant, ref=False):
    if variant not in ACCEL_STEPS:
        raise ValueError('未知 H3 加速方案')
    trunk = 'ref2va' if ref else 'fl2va'
    return {'baseline':None, 'pika4':f'pika_h3_flex_turbo_4step_{trunk}.safetensors',
            'pika6':f'pika_h3_flex_{trunk}_comfy_bf16.safetensors',
            'turbo8':f'minimax_h3_{"ref2v" if ref else "fl2v"}_turbo_8step_v1.0_768p_comfyui_bf16.safetensors'}[variant]

def configure_sampling(graph, variant, sampler, scheduler, shift=12):
    graph[sampler]['inputs']['sampler_name']='lcm' if variant=='pika6' else 'euler'
    if variant=='pika6':
        knots=[1,.75,.5,.25,.1,.04,0]
        graph[scheduler]={'class_type':'ManualSigmas','inputs':{
            'sigmas':', '.join(str(shift*q/(1+(shift-1)*q)) for q in knots)}}
    else:
        graph[scheduler]['inputs']['steps']=ACCEL_STEPS[variant]

def apply_swap_acceleration(graph, variant):
    filename=acceleration_file(variant,True)
    graph['11']['inputs']['steps']=ACCEL_STEPS[variant]
    if filename:
        graph['16']={'class_type':'LoraLoaderModelOnly','inputs':{
            'model':['15',0],'lora_name':filename,'strength_model':1}}
        graph['17']['inputs']['model']=['16',0]
    else:
        graph.pop('16',None)
        graph['17']['inputs']['model']=['15',0]
    configure_sampling(graph,variant,'22','59')


def build_flex(prompt, seed, prefix, video, image, quality=False, width=None, height=None, fit='pad', blend_frames=0, acceleration=None):
    filename = 'ref2va-quality.api.json' if quality else 'ref2va-flex.api.json'
    workflow = json.loads(TEMPLATE.with_name(filename).read_text())
    graph = workflow['prompt']
    graph['5']['inputs']['video'] = video
    graph['2']['inputs']['image'] = image
    graph['11']['inputs']['seed'] = seed
    graph['flex_prompt']['inputs']['string_b'] = prompt
    graph['67']['inputs']['filename'] = prefix
    # A new request must not resume segments rendered with a different prompt.
    graph['model_run_name']['inputs']['string_b'] = '_webui_' + prefix[-32:]
    apply_swap_acceleration(graph,acceleration or ('baseline' if quality else 'pika4'))
    if blend_frames:
        if not 0 <= blend_frames <= 39:
            raise ValueError('衔接融合帧数须在0至39之间')
        graph['26']['inputs']['video_blend_frames'] = blend_frames
        graph['67']['inputs']['blend_video_vae'] = ['20',0]
    if width is not None:
        if height is None or width % 32 or height % 32 or fit not in ('pad', 'crop'):
            raise ValueError('视频画布须按32像素对齐，适配方式须为补边或裁剪')
        graph['11']['inputs']['short_edge'] = min(width, height)
        for node in graph.values():
            for key, value in node['inputs'].items():
                if value == ['5', 0]:
                    node['inputs'][key] = ['quality_fit', 0]
        graph['quality_fit'] = {'class_type':'H3WebUIFitImage', 'inputs':{
            'image':['5',0], 'width':width, 'height':height, 'fit':fit}}
    return workflow


def ancestors(graph, targets):
    keep = set()
    def visit(key):
        if key in keep:
            return
        keep.add(key)
        for value in graph[key]['inputs'].values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                visit(str(value[0]))
    for key in targets:
        visit(key)
    return {key: value for key, value in graph.items() if key in keep}


def build(mode, prompt, seed, prefix, variant='baseline', latent=None, video=None,
          input_mode='text', image=None, last_image=None, references=None,
          width=1344, height=768, frame_count=121, internal_frames=124,
          fit='pad', target_width=0, target_height=0, face_region=False, face_strength=0.55):
    if face_region and mode != 'refine':
        raise ValueError('FaceRegion 仅支持视频精修')
    if not 0.01 <= face_strength <= 1:
        raise ValueError('脸部重绘强度须在0.01至1之间')
    graph = copy.deepcopy(json.loads(TEMPLATE.read_text()))
    if 'prompt' in graph:
        graph = graph['prompt']
    graph['5']['inputs']['prompt'] = prompt
    graph['6']['inputs']['noise_seed'] = seed
    for key in ('15', '16', '27', '31'):
        graph[key]['inputs']['filename_prefix'] = prefix + '_' + key
    graph['16']['inputs']['audio'] = ['13', 1]
    graph['5']['inputs'].update(width=width, height=height, length=internal_frames)
    for key in ('13', '16'):
        graph[key]['inputs']['frame_count'] = frame_count
    graph['13']['inputs']['crop'] = 'none'
    if mode == 'draft':
        if input_mode not in ('text', 'first', 'first_last', 'reference'):
            raise ValueError('Unknown draft input mode')
        if input_mode in ('first', 'first_last'):
            if not image or (input_mode == 'first_last' and not last_image):
                raise ValueError('首帧／首尾帧模式缺少图片')
            graph['50'] = {'class_type': 'LoadImage', 'inputs': {'image': image}}
            graph['52'] = {'class_type': 'H3WebUIFitImage', 'inputs': {
                'image': ['50', 0], 'width': width, 'height': height, 'fit': fit}}
            graph['5']['inputs']['first_frame'] = ['52', 0]
            if input_mode == 'first_last':
                graph['51'] = {'class_type': 'LoadImage', 'inputs': {'image': last_image}}
                graph['53'] = {'class_type': 'H3WebUIFitImage', 'inputs': {
                    'image': ['51', 0], 'width': width, 'height': height, 'fit': fit}}
                graph['5']['inputs']['last_frame'] = ['53', 0]
        if input_mode == 'reference':
            references = references or {}
            if not (references.get('ref_images') or references.get('ref_videos')):
                raise ValueError('多模态参考至少需要一张图片或一段视频')
            graph['1']['inputs']['unet_name'] = REF_MODEL
            graph['8']['inputs']['steps'] = 20
            graph['5']['class_type'] = 'MiniMaxH3ReferenceToVideo'
            graph['5']['inputs'].update(audio_vae=['4', 0], ref_image_size='match')
            index = 50
            for name, files in references.items():
                for ordinal, filename in enumerate(files):
                    key = str(index)
                    index += 1
                    if name == 'ref_images':
                        graph[key] = {'class_type': 'LoadImage', 'inputs': {'image': filename}}
                        graph['5']['inputs'][f'ref_images.ref_image_{ordinal}'] = [key, 0]
                    elif name == 'ref_audios':
                        graph[key] = {'class_type': 'LoadAudio', 'inputs': {'audio': filename}}
                        graph['5']['inputs'][f'ref_audios.ref_audio_{ordinal}'] = [key, 0]
                    elif name == 'ref_videos':
                        graph[key] = {'class_type': 'LoadVideo', 'inputs': {'file': filename}}
                        components = str(index)
                        prepared = str(index + 1)
                        index += 2
                        graph[components] = {'class_type': 'GetVideoComponents', 'inputs': {'video': [key, 0]}}
                        graph[prepared] = {'class_type': 'OpenH3RegenPrepareDraft', 'inputs': {
                            'images': [components, 0], 'size': '768p (1344x768)', 'max_frames': 121,
                            'source_fps': [components, 2]}}
                        graph['5']['inputs'][f'ref_videos.ref_video_{ordinal}'] = [prepared, 0]
                        graph['5']['inputs'][f'ref_video_audios.ref_video_audio_{ordinal}'] = [components, 1]
        if variant != 'baseline':
            graph['40'] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {
                'model': ['32', 0], 'lora_name': acceleration_file(variant,input_mode=='reference'), 'strength_model': 1.0}}
            graph['41'] = {'class_type': 'MiniMaxH3SigmaShift', 'inputs': {
                'model': ['40', 0], 'shift_video': 12.0 if input_mode=='reference' or variant=='pika6' else 6.0, 'shift_audio': 3.0}}
            for key in ('8', '9'):
                graph[key]['inputs']['model'] = ['41', 0]
            configure_sampling(graph,variant,'7','8',graph['41']['inputs']['shift_video'])
        draft_model = ['41', 0] if variant != 'baseline' else ['32', 0]
        graph['45'] = {'class_type': 'H3WebUIProgress', 'inputs': {'model': draft_model}}
        for key in ('8', '9'):
            if graph[key]['class_type']!='ManualSigmas':
                graph[key]['inputs']['model'] = ['45', 0]
        return {'prompt': ancestors(graph, ['15', '16']), 'final_node': '15', 'latent_node': '16'}


CONTINUATION_FRAMES = ('5', '22', '39', '56', '73', '90', '107', '124', '141', '158', '175', '192')
CHAIN_MODES = {
    'auto': 'Auto (首段全新 / 后续自动接力)',
    'force': 'Force Initial (强制新建首段，无上下文)',
}


def build_long(prompt, seed, prefix, project, continuation_frames='39', chain='auto', **draft):
    """One PrefixStream segment on the existing H3 draft graph."""
    if str(continuation_frames) not in CONTINUATION_FRAMES:
        raise ValueError('重叠帧必须是 5、22、39 直到 192 这一档')
    if chain not in CHAIN_MODES:
        raise ValueError('未知接力方式')
    if project != ''.join(c for c in project if c.isalnum() or c in '_- ').strip() or not project:
        raise ValueError('项目名只能包含字母、数字、空格、下划线和连字符')
    if int(continuation_frames) >= draft['internal_frames']:
        raise ValueError('重叠保护帧必须短于本段生成帧数，请加长本段或减小重叠')
    workflow = build('draft', prompt, seed, prefix, **draft)
    graph = workflow['prompt']
    model = graph['9']['inputs']['model']
    graph['ps_config'] = {'class_type': 'MiniMaxPrefixCacheConfig', 'inputs': {
        'cache_mode': 'Native Masked AV (Recommended)', 'continuation_frames': str(continuation_frames),
        'audio_tail_carryover': 'Match Video Handover', 'audio_feather_ticks': 0,
        'video_context_denoise': 0.0, 'reference_noise': False}}
    graph['ps_pick'] = {'class_type': 'MiniMaxClipBinPicker', 'inputs': {
        'project_name': project, 'mode': CHAIN_MODES[chain], 'filter_rating': 'All (1-5 ⭐)',
        'clip_selection': 'latest', 'custom_clip_path': '', 'view_mode': 'Deck (卡片流)'}}
    graph['ps_apply'] = {'class_type': 'MiniMaxPrefixCacheApplier', 'inputs': {
        'model': model, 'conditioning': ['5', 0], 'cache_config': ['ps_config', 0],
        'context_latent': ['ps_pick', 0], 'target_latent': ['5', 1]}}
    for key in ('8', '9'):
        graph[key]['inputs']['model'] = ['ps_apply', 0]
    graph['9']['inputs']['conditioning'] = ['ps_apply', 1]
    graph['10']['inputs']['latent_image'] = ['ps_apply', 3]
    graph['ps_trim'] = {'class_type': 'MiniMaxTrimPrefixLatent', 'inputs': {
        'trim_frames': 0, 'images': ['11', 0], 'audio': ['12', 0], 'session': ['ps_apply', 2],
        'cache_config': ['ps_config', 0], 'fps': 24.0, 'match_tail': True}}
    graph['14']['inputs']['images'] = ['ps_trim', 0]
    graph['14']['inputs']['audio'] = ['ps_trim', 1]
    graph['ps_save'] = {'class_type': 'MiniMaxClipBinSaver', 'inputs': {
        'latent': ['10', 0], 'project_name': project, 'shot_tag': 'Auto (自动编号)', 'rating': 4,
        'images': ['ps_trim', 0], 'audio': ['ps_trim', 1], 'prompt': prompt,
        'parent_clip_id': ['ps_pick', 4], 'save_video': True}}
    workflow['prompt'] = ancestors(graph, ['15', '16', 'ps_save'])
    return workflow
    if mode != 'refine':
        raise ValueError('Unknown generation mode')
    graph['24']['inputs']['noise_seed'] = seed
    if latent:
        graph['40'] = {'class_type': 'OpenH3LoadRefinedLatent', 'inputs': {'latent_file': latent}}
        source, audio, frames = ['40', 0], ['40', 2], ['40', 1]
    elif video:
        graph['40'] = {'class_type': 'LoadVideo', 'inputs': {'file': video}}
        graph['41'] = {'class_type': 'GetVideoComponents', 'inputs': {'video': ['40', 0]}}
        graph['42'] = {'class_type': 'OpenH3RegenPrepareDraft', 'inputs': {
            'images': ['41', 0], 'size': 'keep (snap to 32)', 'max_frames': 0,
            'source_fps': ['41', 2]}}
        graph['44'] = {'class_type': 'H3WebUIFitImage', 'inputs': {
            'image': ['41', 0], 'width': width, 'height': height, 'fit': 'pad'}}
        graph['42']['inputs']['images'] = ['44', 0]
        graph['43'] = {'class_type': 'VAEEncodeTiled', 'inputs': {'pixels': ['42', 0], 'vae': ['3', 0],
            'tile_size': 512, 'overlap': 64, 'temporal_size': 64, 'temporal_overlap': 8}}
        source, audio, frames = ['43', 0], ['41', 1], ['42', 1]
    else:
        raise ValueError('Refine requires a saved draft or uploaded video')
    graph['20']['inputs']['latent'] = source
    graph['20']['inputs'].update(target_width=target_width, target_height=target_height)
    graph['29']['inputs']['crop'] = 'none'
    graph['45'] = {'class_type': 'H3WebUIProgress', 'inputs': {'model': ['22', 0]}}
    graph['25']['inputs']['model'] = ['45', 0]
    if face_region:
        graph['22'] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': ['32', 0], 'lora_name': FACE_REFINER, 'strength_model': 1.0}}
        graph['21']['inputs'].update(audio_mode='draft audio (AUDIO input)', audio=audio)
        if latent:
            graph['face_frames'] = {'class_type': 'VAEDecodeTiled', 'inputs': {
                'samples': source, 'vae': ['3', 0], 'tile_size': 512, 'overlap': 64,
                'temporal_size': 64, 'temporal_overlap': 8}}
            draft_images = ['face_frames', 0]
        else:
            draft_images = ['42', 0]
        graph['face_map'] = {'class_type': 'OpenH3FaceRegionMap', 'inputs': {
            'images': draft_images, 'latent': ['21', 0], 'face_strength': face_strength,
            'pad': 0.35, 'feather': 1.5, 'threshold': 0.6, 'extra_boxes': '',
            'filename_prefix': prefix + '_faces'}}
        graph['24'] = {'class_type': 'OpenH3RegionNoise', 'inputs': {
            'noise_seed': seed, 'mask_file': ['face_map', 0]}}
        graph['face_model'] = {'class_type': 'OpenH3RegionModel', 'inputs': {
            'model': ['22', 0], 'mask_file': ['face_map', 0]}}
        graph['45']['inputs']['model'] = ['face_model', 0]
    for key in ('27', '29'):
        graph[key]['inputs']['audio'] = audio
        graph[key]['inputs']['frame_count'] = frames
    # The upscaler receives the video latent directly, then creates native AV
    # conditioning in node21; audio is preserved separately in the final MP4.
    return {'prompt': ancestors(graph, ['31', '27']), 'final_node': '31', 'latent_node': '27'}
