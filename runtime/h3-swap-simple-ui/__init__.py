import json
import os
import secrets
import asyncio
import shutil
import time
import uuid
import base64
import io
from pathlib import Path
from fractions import Fraction
from PIL import Image, ImageOps

from aiohttp import web, ClientSession

import folder_paths
from server import PromptServer
from comfy.ldm.minimax.model import MiniMaxH3Model
from .job_store import JobStore, WORKFLOW_MODES
from .qwen_workflow import build_qwen, MODELS as QWEN_MODELS
from .workflows import build, build_flex, build_long, TURBO, REF_MODEL, FACE_REFINER, ACCEL_STEPS, acceleration_file, configure_sampling
from .layer_progress import callback
from .geometry import canvas, refined_size, placement, frames, RESOLUTIONS
from .image_fit import H3WebUIFitImage
from .skin_matte import H3WebUISkinMatte
from .enhancements import lora_catalog, settings, effective_prompt, apply_enhancements, rocm_gfx1151
from .image_upload import IMAGE_SUFFIXES, image_suffix


ROOT = Path(__file__).resolve().parent
WORKFLOWS = Path(folder_paths.get_user_directory()) / "default/workflows/H3_BF16_Face_and_Character_Swap"
MODES = {
    "face": WORKFLOWS / "H3_FaceSwap_BF16_Relay.json",
    "character": WORKFLOWS / "H3_CharacterSwap_BF16_Relay.json",
}
ALLOWED_VIDEO = {".mp4", ".mov", ".mkv", ".webm"}
ALLOWED_IMAGE = IMAGE_SUFFIXES
ALLOWED_AUDIO = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
OUTPUTS = Path(os.environ['H3_WEBUI_OUTPUT']) if os.environ.get('H3_WEBUI_OUTPUT') else ROOT.parents[2] / 'outputs' / 'h3-webui'
STORE = JobStore(OUTPUTS / 'jobs')
FINALIZING = set()


async def save_part(part, suffixes, limit):
    suffix = Path(part.filename or "").suffix.lower()
    if suffix not in suffixes:
        raise web.HTTPBadRequest(text="不支持的文件格式")
    name = f"simple-swap-{secrets.token_hex(12)}{suffix}"
    path = Path(folder_paths.get_input_directory()) / name
    size = 0
    try:
        with path.open("xb") as output:
            while chunk := await part.read_chunk(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=size)
                output.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if suffixes == ALLOWED_IMAGE:
        try:
            actual_suffix = image_suffix(path)
            if suffix != actual_suffix:
                target = path.with_suffix(actual_suffix)
                path.rename(target)
                name = target.name
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            path.unlink(missing_ok=True)
            raise web.HTTPBadRequest(text='无法读取参考图片：' + str(error))
        return name
    with path.open("rb") as uploaded:
        header = uploaded.read(16)
    valid = (
        suffix in {".mp4", ".mov"} and header[4:8] == b"ftyp"
        or suffix in {".mkv", ".webm"} and header.startswith(b"\x1aE\xdf\xa3")
        or suffix == '.wav' and header.startswith(b'RIFF') and header[8:12] == b'WAVE'
        or suffix == '.flac' and header.startswith(b'fLaC')
        or suffix == '.ogg' and header.startswith(b'OggS')
        or suffix == '.mp3' and (header.startswith(b'ID3') or header[:1] == b'\xff')
        or suffix == '.m4a' and header[4:8] == b'ftyp'
    )
    if not valid:
        path.unlink(missing_ok=True)
        raise web.HTTPBadRequest(text="文件内容与格式不匹配")
    return name


@PromptServer.instance.routes.get("/simple-swap/")
async def index(_request):
    return web.FileResponse(ROOT / "index.html", headers={'Cache-Control': 'no-store'})


@PromptServer.instance.routes.get('/simple-swap/app.js')
async def javascript(_request):
    return web.FileResponse(ROOT / 'app.js', headers={'Cache-Control': 'no-store'})


@PromptServer.instance.routes.get('/simple-swap/api/options')
async def options(_request):
    backends = ['ck', 'sol-hip'] if rocm_gfx1151() else ['native']
    return web.json_response({'resolutions': list(RESOLUTIONS), 'workflows': list(WORKFLOW_MODES), 'flex_geometry': True, 'extra_loras': lora_catalog(folder_paths), 'skin_preparation': True, 'attention_backends': backends}, headers={'Cache-Control': 'no-store'})


def input_image(name):
    path = Path(folder_paths.get_annotated_filepath(name)).resolve()
    if not path.is_relative_to(Path(folder_paths.get_input_directory()).resolve()):
        raise ValueError('输入图片路径不允许')
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


async def video_info(path):
    process = await asyncio.create_subprocess_exec('ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,avg_frame_rate,duration:format=duration', '-of', 'json', str(path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    output, error = await process.communicate()
    if process.returncode:
        raise ValueError('无法读取视频尺寸和时长：' + error.decode()[-500:])
    info = json.loads(output)
    if not info.get('streams'):
        raise ValueError('文件没有可解码的视频轨')
    stream = info['streams'][0]
    rate = Fraction(stream.get('avg_frame_rate', '0/1'))
    if str(stream.get('nb_frames', '')).isdigit() and rate > 0:
        seconds = int(stream['nb_frames']) / float(rate)
    else:
        seconds = float(stream.get('duration') or info['format']['duration'])
    return (stream['width'], stream['height']), seconds


@PromptServer.instance.routes.post('/simple-swap/api/preview')
async def preview(request):
    fields, names, uploaded = {}, {}, []
    try:
        reader = await request.multipart()
        while part := await reader.next():
            if part.name in ('image', 'last_image', 'source_frame'):
                name = await save_part(part, ALLOWED_IMAGE, 25 * 1024**2)
                names[part.name] = name
                uploaded.append(name)
            else:
                fields[part.name] = await part.text()
        restored = STORE.jobs.get(fields.get('restore_job', ''), {})
        images = {}
        for name, key in (('image', 'image_input'), ('last_image', 'last_image_input')):
            if name == 'last_image' and fields.get('input_mode') != 'first_last':
                continue
            filename = names.get(name) or (restored.get(key) if fields.get('input_mode') in ('first', 'first_last') else None)
            if filename:
                images[name] = input_image(filename)
        source = images['image'].size if 'image' in images else None
        width, height = canvas(fields.get('aspect', 'auto'), source, fields.get('resolution'))
        count, internal = frames(fields.get('seconds', '5'))
        fit = fields.get('fit', 'pad')
        if fit not in ('crop', 'pad'):
            raise ValueError('请选择裁剪或补边')
        target_width, target_height = 0, 0
        if fields.get('mode') in ('quality_character', 'flex_character'):
            if names.get('source_frame'):
                images['image'] = input_image(names['source_frame'])
                source = images['image'].size
            elif restored.get('video_input'):
                source, seconds = await video_info(Path(folder_paths.get_annotated_filepath(restored['video_input'])))
                fields['seconds'] = str(seconds)
            elif fields.get('source_width') and fields.get('source_height'):
                source = int(fields['source_width']), int(fields['source_height'])
            if source is None:
                return web.json_response({'pending_source': True, 'previews': {}})
            if min(source) < 1 or max(source) > 16384:
                raise ValueError('无法识别视频画幅')
            width, height = canvas(fields.get('aspect','auto'), source, fields.get('resolution','480'))
            count = max(5, round(float(fields.get('seconds','5'))*24))
            count -= (count-5)%17
            internal = count
        if fields.get('mode') == 'refine':
            source_job = STORE.jobs.get(fields.get('source_job', ''))
            if source_job and source_job.get('video_path'):
                path = (Path(folder_paths.get_annotated_filepath(source_job['video_input']))
                    if source_job.get('mode') == 'refine' and source_job.get('video_input')
                    else Path(source_job['video_path']))
                source, seconds = await video_info(path)
                count, internal = frames(seconds)
            elif fields.get('source_width') and fields.get('source_height'):
                source = int(fields['source_width']), int(fields['source_height'])
                if min(source) < 1 or max(source) > 16384:
                    raise ValueError('无法识别视频画幅')
            else:
                return web.json_response({'pending_source': True, 'previews': {}})
            width, height = source if source_job else canvas('auto', source)
            target_width, target_height = refined_size(*source)
        previews = {}
        for name, image in images.items():
            w, h, x, y = placement(*image.size, width, height, fit)
            rendered = Image.new('RGB', (width, height), 'black')
            rendered.paste(image.resize((w, h), Image.Resampling.BILINEAR), (x, y))
            rendered.thumbnail((480, 480))
            buffer = io.BytesIO()
            rendered.save(buffer, format='PNG')
            previews[name] = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        return web.json_response({'width': width, 'height': height, 'frame_count': count,
            'internal_frames': internal, 'seconds': count / 24, 'previews': previews,
            'target_width': target_width, 'target_height': target_height})
    except (ValueError, OSError) as error:
        raise web.HTTPBadRequest(text=str(error))
    finally:
        for name in uploaded:
            (Path(folder_paths.get_input_directory()) / name).unlink(missing_ok=True)


@PromptServer.instance.routes.post("/simple-swap/api/prepare")
async def prepare(request):
    reader = await request.multipart()
    mode = None
    user_prompt = None
    video = None
    image = None
    last_image = None
    references = {'ref_images': [], 'ref_videos': [], 'ref_audios': []}
    fields = {}
    uploads = []
    try:
        while part := await reader.next():
            if part.name == "mode":
                mode = await part.text()
            elif part.name == "prompt":
                user_prompt = (await part.text()).strip()
            elif part.name == "video":
                video = await save_part(part, ALLOWED_VIDEO, 2 * 1024**3)
                uploads.append(video)
            elif part.name == "image":
                image = await save_part(part, ALLOWED_IMAGE, 25 * 1024**2)
                uploads.append(image)
            elif part.name == 'last_image':
                last_image = await save_part(part, ALLOWED_IMAGE, 25 * 1024**2)
                uploads.append(last_image)
            elif part.name in references:
                suffixes, limit = {'ref_images': (ALLOWED_IMAGE, 25 * 1024**2),
                    'ref_videos': (ALLOWED_VIDEO, 2 * 1024**3),
                    'ref_audios': (ALLOWED_AUDIO, 100 * 1024**2)}[part.name]
                name = await save_part(part, suffixes, limit)
                references[part.name].append(name)
                uploads.append(name)
            else:
                fields[part.name] = await part.text()
        restored = STORE.jobs.get(fields.get('restore_job', ''))
        if restored and restored.get('mode', 'face') != mode:
            raise web.HTTPBadRequest(text='不能恢复其他工作流的任务素材')
        if restored:
            video = video or restored.get('video_input')
            image = image or restored.get('image_input')
            last_image = last_image or restored.get('last_image_input')
            for name in references:
                references[name] = references[name] or restored.get('references', {}).get(name, [])
        if mode not in WORKFLOW_MODES or not user_prompt:
            raise web.HTTPBadRequest(text='请选择模式并填写提示词')
        if mode in (*MODES, 'flex_character', 'quality_character') and (not video or not image):
            raise web.HTTPBadRequest(text="请选择模式、视频、参考图片并填写提示词")
        if len(user_prompt) > 12000:
            raise web.HTTPBadRequest(text="提示词不能超过 12000 个字符")
        if mode in MODES and "<Picture 1>" not in user_prompt:
            raise web.HTTPBadRequest(text="提示词必须保留 <Picture 1>，否则模型无法识别参考图片")
        seed_text = fields.get('seed', '0').strip()
        try:
            seed = int(seed_text) if seed_text else secrets.randbelow(2**32)
        except ValueError:
            raise web.HTTPBadRequest(text='种子必须是整数')
        if not 0 <= seed < 2**32:
            raise web.HTTPBadRequest(text='种子必须在 0 到 4294967295 之间')
        variant = fields.get('variant', 'baseline' if mode in ('refine', 'qwen') else 'pika4')
        input_mode = fields.get('input_mode', 'text')
        if mode in ('draft', 'long') and input_mode not in ('text', 'first', 'first_last', 'reference'):
            raise web.HTTPBadRequest(text='未知底稿输入模式')
        if mode not in ('draft', 'long') and input_mode != 'text':
            raise web.HTTPBadRequest(text='该工作流不支持切换底稿输入模式')
        if variant not in ACCEL_STEPS:
            raise web.HTTPBadRequest(text='未知底稿版本')
        metadata = {'mode': mode, 'prompt_text': user_prompt, 'seed': seed,
                    'variant': variant, 'video_input': video, 'image_input': image,
                    'input_mode': input_mode, 'last_image_input': last_image, 'references': references,
                    'source_job': fields.get('source_job', '')}
        try:
            enhancement_config = settings(fields, mode, lora_catalog(folder_paths))
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error))
        if enhancement_config['attention_backend'] == 'sol-hip':
            import torch
            import nodes
            binary = ROOT.parents[2] / 'research/sol-attn-hip/comfy-kitchen/comfy_kitchen/backends/hip/_C.abi3.so'
            if 'H3SolHIPAttention' not in nodes.NODE_CLASS_MAPPINGS or not binary.is_file():
                raise web.HTTPBadRequest(text='本机缺少Sol-Attn HIP节点或原生算子')
            if torch.version.hip is None or not torch.cuda.is_available() or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx1151':
                raise web.HTTPBadRequest(text='当前Sol-Attn HIP构建仅适用于本机ROCm gfx1151')
        metadata.update(enhancement_config, original_prompt=user_prompt)
        user_prompt = effective_prompt(user_prompt, enhancement_config)
        metadata['effective_prompt'] = user_prompt
        def finish(workflow):
            apply_enhancements(workflow, enhancement_config, 'skin_' + uuid.uuid4().hex)
            metadata['skin_artifacts'] = [str(OUTPUTS / 'skin' / (node['inputs']['filename_prefix'] + suffix))
                for node in workflow['prompt'].values() if node['class_type'] == 'H3WebUISkinMatte'
                for suffix in ('_source.png', '_cleaned.png', '_mask.png', '.json')]
            workflow['_job'] = metadata
            return web.json_response(workflow)
        if mode=='refine' and variant!='baseline':
            raise web.HTTPBadRequest(text='2K Regenerate 只支持专用精修 LoRA，不兼容生成加速 LoRA')
        face_region = fields.get('face_region', 'false')
        if face_region not in ('true', 'false'):
            raise web.HTTPBadRequest(text='无效的 FaceRegion 选项')
        face_region = face_region == 'true'
        try:
            face_strength = float(fields.get('face_strength', '0.55'))
        except ValueError:
            raise web.HTTPBadRequest(text='脸部重绘强度必须是数字')
        if not 0.01 <= face_strength <= 1:
            raise web.HTTPBadRequest(text='脸部重绘强度须在0.01至1之间')
        if face_region:
            if mode != 'refine':
                raise web.HTTPBadRequest(text='FaceRegion 仅支持视频精修')
            if FACE_REFINER not in folder_paths.get_filename_list('loras'):
                raise web.HTTPBadRequest(text='缺少 FaceRegion BF16 refiner：' + FACE_REFINER)
        if mode == 'refine':
            metadata.update(face_region=face_region, face_strength=face_strength)
        if mode!='qwen' and mode!='refine':
            filename=acceleration_file(variant,mode!='draft' or input_mode=='reference')
            if filename and filename not in folder_paths.get_filename_list('loras'):
                raise web.HTTPBadRequest(text='缺少所选加速权重：'+filename)
            metadata['steps']=8 if mode=='draft' and variant=='baseline' and input_mode!='reference' else ACCEL_STEPS[variant]
        if mode in ('flex_character', 'quality_character'):
            try:
                quality = mode == 'quality_character'
                size, seconds = await video_info(Path(folder_paths.get_annotated_filepath(video)))
                input_image(image)
                short_edge, alignment = 480, 32
                scale = short_edge / min(size)
                width, height = (max(alignment, round(n * scale / alignment) * alignment) for n in size)
                aspect, fit = fields.get('aspect','auto'), fields.get('fit','pad')
                if mode in ('quality_character', 'flex_character'):
                    resolution = fields.get('resolution','480')
                    width, height = canvas(aspect, size, resolution)
                    if fit not in ('pad','crop'):
                        raise ValueError('请选择等比例裁剪或补边')
                count = max(5, round(seconds * 24))
                count -= (count - 5) % 17
                metadata.update(variant=variant, width=width, height=height,
                    seconds=count / 24, frame_count=count, internal_frames=count,
                    resolution=resolution, steps=ACCEL_STEPS[variant])
                if mode in ('quality_character', 'flex_character'):
                    metadata.update(aspect=aspect,fit=fit)
                blend_frames = int(fields.get('blend_frames','0'))
                if not 0 <= blend_frames <= 39:
                    raise ValueError('衔接融合帧数须在0至39之间')
                metadata['blend_frames'] = blend_frames
                workflow = build_flex(user_prompt, seed, 'h3_webui_' + mode + '_' + uuid.uuid4().hex,
                                      video, image, quality=quality,
                                      width=width,
                                      height=height,fit=fit,blend_frames=blend_frames,acceleration=variant)
                graph=workflow['prompt']
                extra_images=[name for name in references['ref_images'] if name!=image]
                if len(extra_images)>8:
                    raise ValueError('最多支持九张参考图')
                for ordinal,name in enumerate(extra_images,1):
                    key='extra_ref_'+str(ordinal)
                    graph[key]={'class_type':'LoadImage','inputs':{'image':name}}
                    graph['30']['inputs'][f'ref_images.ref_image_{ordinal}']=[key,0]
                if extra_images and restored and restored.get('parameters',{}).get('roles'):
                    graph['11']['inputs']['picture_2']=['extra_ref_1',0]
                    graph['11']['inputs']['roles']=restored['parameters']['roles']
                    for key in ('39','40'):graph[key]['inputs']['max_people']=0
            except (ValueError, OSError) as error:
                raise web.HTTPBadRequest(text=str(error))
            return finish(workflow)
        if mode == 'qwen':
            if references['ref_videos'] or references['ref_audios'] or video or image or last_image:
                raise web.HTTPBadRequest(text='Qwen 工作流只接受参考图片')
            for kind, name in QWEN_MODELS.items():
                if name not in folder_paths.get_filename_list(kind):
                    raise web.HTTPBadRequest(text='缺少 Qwen Image 2.1 BF16 文件：' + name)
            try:
                source_size = None
                for index, name in enumerate(references['ref_images']):
                    size = input_image(name).size
                    if index == 0:
                        source_size = size
                workflow, params = build_qwen(user_prompt, seed, 'qwen_webui_' + uuid.uuid4().hex,
                    fields, references['ref_images'], source_size)
            except (ValueError, OSError) as error:
                raise web.HTTPBadRequest(text=str(error))
            metadata.update(params)
            return finish(workflow)
        if mode in ('draft', 'long', 'refine'):
            aspect = fields.get('aspect', 'auto')
            resolution = fields.get('resolution')
            fit = fields.get('fit', 'pad')
            seconds = fields.get('seconds', '5')
            if fit not in ('pad', 'crop'):
                raise web.HTTPBadRequest(text='请选择裁剪或补边')
            latent = None
            source = None
            if metadata['source_job']:
                source = STORE.jobs.get(metadata['source_job'])
                if not source or source.get('mode') != 'draft' or source['status'] != 'completed' or not source.get('latent_file'):
                    raise web.HTTPBadRequest(text='请选择已完成并保存 latent 的底稿任务')
                latent = source['latent_file']
            if mode == 'refine' and not latent and not video:
                raise web.HTTPBadRequest(text='请选择已完成底稿或上传 768p 视频')
            if mode in ('draft', 'long') and variant == 'pika4' and TURBO not in folder_paths.get_filename_list('loras'):
                raise web.HTTPBadRequest(text='Pika Turbo LoRA 尚未安装完成')
            if mode in ('draft', 'long') and input_mode == 'reference' and REF_MODEL not in folder_paths.get_filename_list('diffusion_models'):
                raise web.HTTPBadRequest(text='缺少 H3 REF2VA BF16 模型，不使用量化模型替代')
            try:
                if mode in ('draft', 'long'):
                    image_size = input_image(image).size if image and input_mode in ('first', 'first_last') else None
                    width, height = canvas(aspect, image_size, resolution)
                    frame_count, internal_frames = frames(seconds)
                    target_width, target_height = 0, 0
                else:
                    path = Path(source['video_path']) if latent else Path(folder_paths.get_annotated_filepath(video))
                    size, seconds = await video_info(path)
                    width, height = (size if latent else canvas('auto', size))
                    target_width, target_height = refined_size(*size)
                    frame_count, internal_frames = frames(seconds)
                    aspect, fit = 'auto', 'pad'
                metadata.update(aspect=aspect, resolution=resolution if mode in ('draft', 'long') else None,
                    fit=fit, seconds=frame_count / 24, width=width, height=height,
                    frame_count=frame_count, internal_frames=internal_frames,
                    target_width=target_width, target_height=target_height)
                common = dict(variant=variant, input_mode=input_mode, image=image, last_image=last_image,
                              references=references, width=width, height=height, frame_count=frame_count,
                              internal_frames=internal_frames, fit=fit)
                if mode == 'long':
                    project = fields.get('project', 'webui-long').strip() or 'webui-long'
                    chain = fields.get('chain', 'auto')
                    continuation = fields.get('continuation_frames', '39')
                    metadata.update(project=project, chain=chain, continuation_frames=continuation)
                    workflow = build_long(user_prompt, seed, 'h3_webui_long_' + uuid.uuid4().hex,
                                           project, continuation, chain, **common)
                else:
                    workflow = build(mode, user_prompt, seed, 'h3_webui_' + uuid.uuid4().hex,
                                     latent=latent, video=video, target_width=target_width,
                                     target_height=target_height, face_region=face_region,
                                     face_strength=face_strength, **common)
            except ValueError as error:
                raise web.HTTPBadRequest(text=str(error))
            return finish(workflow)
        adapter = ROOT.parents[2] / "openjutsu/local-webui.mjs"
        process = await asyncio.create_subprocess_exec("node", "--experimental-strip-types", str(adapter),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        payload = {"mode": mode, "video": video, "image": image, "prompt": user_prompt, "seed": seed}
        output, error = await process.communicate(json.dumps(payload).encode())
        if process.returncode:
            raise web.HTTPInternalServerError(text="Openjutsu 工作流准备失败：" + error.decode()[-1500:])
        workflow = json.loads(output)
        graph=workflow['prompt']
        backend=next(k for k,n in graph.items() if n.get('_meta',{}).get('title')=='Comfy Kitchen Attention')
        model=graph[backend]['inputs']['model']
        filename=acceleration_file(variant,True)
        if filename:
            graph['webui_speed']={'class_type':'LoraLoaderModelOnly','inputs':{'model':model,'lora_name':filename,'strength_model':1}}
            model=['webui_speed',0]
        graph[backend]={'class_type':'H3CKAttention','inputs':{'model':model}}
        sampler=next(k for k,n in graph.items() if n['class_type']=='KSamplerSelect')
        scheduler=next(k for k,n in graph.items() if n['class_type']=='BasicScheduler')
        configure_sampling(graph,variant,sampler,scheduler)
        for node in graph.values():
            if node['class_type'].startswith('H3') and 'steps' in node['inputs']:
                node['inputs']['steps']=ACCEL_STEPS[variant]
        return finish(workflow)
    except Exception:
        for name in uploads:
            (Path(folder_paths.get_input_directory()) / name).unlink(missing_ok=True)
        raise


@PromptServer.instance.routes.post('/simple-swap/api/jobs')
async def submit(request):
    response = await prepare(request)
    workflow = json.loads(response.body)
    job = workflow.pop('_job')
    job_id = str(uuid.uuid4())
    job.update(id=job_id, client_id=str(uuid.uuid4()), created_at=time.time(),
               status='queued', timings=[], graph=workflow['prompt'],
               final_node=workflow['final_node'], latent_node=workflow.get('latent_node'))
    STORE.save(job)
    async with ClientSession() as session:
        async with session.post(f'http://127.0.0.1:{PromptServer.instance.port}/prompt', json={
            'prompt_id': job_id, 'client_id': job['client_id'], 'prompt': job['graph']}) as queued:
            data = await queued.json()
            if queued.status != 200:
                job.update(status='error', error=json.dumps(data, ensure_ascii=False), finished_at=time.time())
                STORE.save(job)
                raise web.HTTPBadRequest(text='工作流验证失败：' + job['error'])
    return web.json_response(public_job(job))


def public_job(job):
    result = {key: value for key, value in job.items() if key not in ('graph', 'client_id')}
    node_key = str(job.get('node'))
    graph = job.get('graph', {})
    node = graph.get(node_key) or graph.get(node_key.split('.')[-1], {})
    result['stage'] = node.get('_meta', {}).get('title') or node.get('class_type') or (
        '节点 ' + node_key if job.get('node') else '等待执行')
    result['elapsed'] = max(0, (job.get('finished_at') or time.time()) - (job.get('started_at') or job['created_at']))
    result['timings'] = [{**timing, 'stage': graph.get(str(timing['node']), {}).get('class_type', '')}
                         for timing in job.get('timings', [])]
    if job.get('step', 0) >= 2 and job.get('total', 0) > job['step'] and job.get('stage_started_at'):
        result['stage_eta'] = (time.time() - job['stage_started_at']) / job['step'] * (job['total'] - job['step'])
    return result


def find_video(value):
    if isinstance(value, dict):
        if str(value.get('filename', '')).lower().endswith('.mp4'):
            return value
        values = value.values()
    elif isinstance(value, list):
        values = value
    else:
        return None
    for item in values:
        found = find_video(item)
        if found:
            return found


async def finalize(job, history):
    if job['id'] in FINALIZING:
        return
    FINALIZING.add(job['id'])
    try:
        if job['mode'] == 'qwen':
            images = history.get('outputs', {}).get(job['final_node'], {}).get('images', [])
            if not images:
                raise ValueError('任务执行结束，但没有输出图片')
            source_root = Path(folder_paths.get_output_directory()).resolve()
            target_root = OUTPUTS / 'images' / job['id']
            target_root.mkdir(parents=True, exist_ok=True)
            results = []
            for index, image in enumerate(images):
                source = (source_root / image.get('subfolder', '') / image['filename']).resolve()
                if not source.is_relative_to(source_root):
                    raise ValueError('图片输出路径不合法')
                with Image.open(source) as decoded:
                    decoded.verify()
                target = target_root / f'{index + 1}.png'
                await asyncio.to_thread(shutil.copy2, source, target)
                results.append({'path': str(target), 'url': f'/simple-swap/api/jobs/{job["id"]}/images/{index}'})
            job.update(status='completed', finished_at=time.time(), images=results)
            STORE.save(job)
            return
        file = find_video(history.get('outputs', {}).get(job['final_node']))
        if not file:
            raise ValueError('任务执行结束，但最终视频节点没有输出')
        source_root = Path(folder_paths.get_output_directory()).resolve()
        source = source_root / file.get('subfolder', '') / file['filename']
        if not source.resolve().is_relative_to(source_root) and not source.resolve().is_relative_to(OUTPUTS.resolve()):
            raise ValueError('视频输出路径不合法')
        target = OUTPUTS / 'videos' / (job['id'] + '.mp4')
        target.parent.mkdir(parents=True, exist_ok=True)
        paseo = Path.home() / '.paseo' / 'streaming-video.json'
        if paseo.is_file():
            allowed = json.loads(paseo.read_text())['allowedRoots']
            if not any(target.resolve().is_relative_to(Path(root).resolve()) for root in allowed):
                raise ValueError('成片目录不在播放器允许目录内')
        if not target.exists() and job['mode'] == 'flex_character':
            # Match the verified720p export while preserving the source aspect.
            process = await asyncio.create_subprocess_exec('ffmpeg', '-v', 'error', '-n', '-i', str(source),
                '-vf', f'scale={job["width"]}:{job["height"]}:force_original_aspect_ratio=increase,'
                f'crop={job["width"]}:{job["height"]},setsar=1', '-c:v', 'libx264', '-crf', '16',
                '-preset', 'slow', '-c:a', 'copy', '-movflags', '+faststart', str(target),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            _, error = await process.communicate()
            if process.returncode:
                raise ValueError('720p 导出失败：' + error.decode()[-500:])
        elif not target.exists():
            await asyncio.to_thread(shutil.copy2, source, target)
        process = await asyncio.create_subprocess_exec('ffmpeg', '-v', 'error', '-i', str(target),
            '-f', 'null', '-', stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        _, error = await process.communicate()
        if process.returncode:
            raise ValueError('成片解码校验失败：' + error.decode()[-500:])
        if source.resolve() != target.resolve():
            source.unlink(missing_ok=True)
            source.symlink_to(target)
        job.update(status='completed', finished_at=time.time(), video_path=str(target),
                   video_url=f'/simple-swap/api/jobs/{job["id"]}/video')
        if job.get('latent_node'):
            texts = history.get('outputs', {}).get(job['latent_node'], {}).get('text', [])
            if texts:
                name = Path(texts[-1]).name
                if name.endswith('.safetensors') and (source_root / 'open_h3_latents' / name).is_file():
                    job['latent_file'] = name
        STORE.save(job)
    except (ValueError, OSError) as error:
        job.update(status='error', error=str(error), finished_at=time.time())
        STORE.save(job)
    finally:
        FINALIZING.discard(job['id'])


async def reconcile(job):
    if job['status'] in ('completed', 'error', 'interrupted'):
        return
    server = PromptServer.instance
    history = server.prompt_queue.get_history(job['id']).get(job['id'])
    if history:
        if history.get('status', {}).get('status_str') == 'error':
            job.update(status='error', error=json.dumps(history['status'].get('messages'), ensure_ascii=False), finished_at=time.time())
            STORE.save(job)
        elif history.get('status', {}).get('completed'):
            job['status'] = 'finalizing'
            asyncio.create_task(finalize(job, history))
        return
    running, pending = server.prompt_queue.get_current_queue()
    ids = [item[1] for item in running + pending]
    if job['id'] not in ids:
        job.update(status='interrupted', error='服务已重启或任务已移出队列；可恢复参数后重新制作', finished_at=time.time())
        STORE.save(job)
    elif job['id'] in [item[1] for item in pending]:
        job['queue_ahead'] = ids.index(job['id'])


@PromptServer.instance.routes.get('/simple-swap/api/jobs')
async def list_jobs(_request):
    STORE.load_imports()
    mode = _request.query.get('mode')
    if mode is not None and mode not in WORKFLOW_MODES:
        raise web.HTTPBadRequest(text='未知工作流')
    jobs = STORE.list(mode)
    for job in jobs:
        await reconcile(job)
    return web.json_response([public_job(job) for job in jobs[:100]], headers={'Cache-Control': 'no-store'})


@PromptServer.instance.routes.delete('/simple-swap/api/jobs/{job_id}')
async def delete_job(request):
    mode = request.query.get('mode')
    if mode not in WORKFLOW_MODES:
        raise web.HTTPBadRequest(text='请选择工作流')
    try:
        STORE.delete(request.match_info['job_id'], mode)
    except KeyError:
        raise web.HTTPNotFound(text='此工作流中没有该任务')
    except ValueError as error:
        raise web.HTTPConflict(text=str(error))
    return web.json_response({'deleted': True})


@PromptServer.instance.routes.get('/simple-swap/api/jobs/{job_id}')
async def get_job(request):
    job = STORE.jobs.get(request.match_info['job_id'])
    if not job:
        raise web.HTTPNotFound(text='任务不存在')
    await reconcile(job)
    return web.json_response(public_job(job), headers={'Cache-Control': 'no-store'})


@PromptServer.instance.routes.get('/simple-swap/api/jobs/{job_id}/video')
async def video_result(request):
    job = STORE.jobs.get(request.match_info['job_id'])
    if not job or job['status'] != 'completed':
        raise web.HTTPNotFound(text='视频未完成')
    return web.FileResponse(job['video_path'])


@PromptServer.instance.routes.get('/simple-swap/api/jobs/{job_id}/images/{index}')
async def image_result(request):
    job = STORE.jobs.get(request.match_info['job_id'])
    if not job or job.get('mode') != 'qwen' or job['status'] != 'completed':
        raise web.HTTPNotFound(text='图片未完成')
    try:
        index = int(request.match_info['index'])
        if index < 0:
            raise IndexError
        image = job['images'][index]
    except (ValueError, IndexError, KeyError):
        raise web.HTTPNotFound(text='图片不存在')
    return web.FileResponse(image['path'])


original_send = PromptServer.instance.send_sync
async def finish_when_recorded(job):
    # execution_success precedes the queue's history insertion.
    while job['status'] == 'finalizing':
        await asyncio.sleep(1)
        await reconcile(job)


def track_event(event, data, sid=None):
    STORE.event(event, data, sid)
    original_send(event, data, sid)
    if event == 'execution_success' and isinstance(data, dict):
        job = STORE.jobs.get(data.get('prompt_id'))
        if job:
            PromptServer.instance.loop.call_soon_threadsafe(lambda: asyncio.create_task(finish_when_recorded(job)))
PromptServer.instance.send_sync = track_event


class H3WebUIProgress:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',)}}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'patch'
    CATEGORY = 'MiniMax H3/progress'

    def patch(self, model):
        dm = model.get_model_object('diffusion_model')
        if not isinstance(dm, MiniMaxH3Model):
            raise ValueError('H3WebUIProgress requires MiniMax H3')
        patched = model.clone()
        replacements = model.model_options.get('transformer_options', {}).get('patches_replace', {}).get('dit', {})
        total = len(dm.blocks)
        def report(layer):
            running, _ = PromptServer.instance.prompt_queue.get_current_queue()
            if running and running[0][1] in STORE.jobs:
                PromptServer.instance.send_sync('h3_webui_layer', {
                    'prompt_id': running[0][1], 'layer': layer, 'layers': total})
        for index in range(total):
            patched.set_model_patch_replace(callback(index, report, replacements.get(('double_block', index))),
                                            'dit', 'double_block', index)
        return (patched,)


NODE_CLASS_MAPPINGS = {'H3WebUIProgress': H3WebUIProgress, 'H3WebUIFitImage': H3WebUIFitImage, 'H3WebUISkinMatte': H3WebUISkinMatte}
NODE_DISPLAY_NAME_MAPPINGS = {'H3WebUIProgress': 'H3 WebUI · Model Layer Dispatch Progress'}
WEB_DIRECTORY = "."
