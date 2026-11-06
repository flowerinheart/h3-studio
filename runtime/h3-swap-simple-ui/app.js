const $ = selector => document.querySelector(selector);
const BASE = (location.pathname.includes('/simple-swap/') ? location.pathname.split('/simple-swap/')[0] : '') + '/simple-swap';
const COMFY_BASE = BASE.slice(0,-'/simple-swap'.length);
const prompts = {
face:'Faceswap. Replace only the selected person face with the face in <Picture 1>. Preserve the source hairstyle, body, clothing, scene, camera, performance, lighting and original audio timing. Keep the reference identity consistent in every frame.',
character:'Replace the selected person with the character in <Picture 1>. Preserve the reference character face, hair, clothing and proportions while following the source performance. Preserve the scene, camera, lighting and original audio timing. Keep the reference identity consistent in every frame.',
draft:'A red fox trots through fresh snow ...',long:'同一人物、服装和场景连续向前。动作从上一段自然接上，镜头不切。',refine:'A red fox trots through fresh snow ...',qwen:'一张真实摄影风格的图片，柔和自然光，清晰细节。'};
prompts.flex_character='将原视频中的唯一表演者完整替换为参考图 Picture 1 中的同一人物，不只是换脸。参考图只提供身份、五官、发型、服装、头饰和身体设计，不复制参考图背景或静止姿态。逐帧跟随原视频的动作、表情、嘴型、视线、手指遮挡和头肩运动，头发自然跟随头部与原视频气流摆动，不重新编排动作。保留原视频的镜头、构图、背景、光照、时序和原声音。不新增字幕或水印，避免手指融合、脸部变形、服装漂移和背景闪烁。';
prompts.quality_character=prompts.flex_character;
const names={face:'只换脸',character:'换整个人',flex_character:'视频换人 · REF2VA Flex',quality_character:'视频换人 · BF16 质量档',draft:'生成底稿',long:'长视频接力',refine:'2K 精修',qwen:'Qwen Image 2.1'};
const states={queued:'排队中',running:'执行中',finalizing:'输出校验中',completed:'已完成',error:'失败',interrupted:'已中止'};
const experimentNames={turbo8:'LightX2V Turbo 8步',pika6:'Pika 官方 6步'};
function accelerationHint(){
 if(['qwen','refine'].includes(mode))return;
 const label={baseline:'无生成加速 LoRA',pika4:'Pika 社区融合 4步',pika6:'Pika 官方 6步 · LCM 专用 sigma',turbo8:'LightX2V Turbo 8步'}[$('#variant').value];
 if(label)$('#fixedMode').textContent='所选方案：'+label+' · 本地 H3 BF16 · '+($('#attentionBackend').value==='sol-hip'?'Sol-Attn HIP':'CK')+(mode==='long'?'。同一项目名接上一段 latent，成片裁掉开头重叠。视频 VAE FP16，音频 VAE FP32。':'。自动按输入模式匹配 FL2VA／REF2VA；只换脸保留专用换脸 LoRA。');
}
const stages={H3WebUISkinMatte:'处理输入皮肤高光',TextEncodeQwenImage21:'编码 Qwen 提示词与参考图',QwenImage21Cache:'准备 Qwen 缓存',KSampler:'Qwen 采样',SaveImage:'保存图片',UNETLoader:'加载 H3 BF16',CLIPLoader:'加载文字编码器',MiniMaxH3ImageToVideo:'编码提示词／准备底稿',MiniMaxH3ReferenceToVideo:'编码参考并准备底稿',LoraLoaderModelOnly:'加载 H3 LoRA',H3SolHIPAttention:'配置Sol稀疏注意力',H3CKAttention:'配置密集 CK',SamplerCustomAdvanced:'H3 采样',VAEDecode:'原生 VAE 视频解码',VAEDecodeTiled:'原生 VAE 分块解码',VAEDecodeAudio:'音频解码',VAEEncodeTiled:'输入视频 VAE 编码',OpenH3FaceRegionMap:'检测人脸／构建保护区域',OpenH3RegionNoise:'配置脸部区域噪声',OpenH3RegionModel:'配置脸部重绘强度',OpenH3RegenUpscale:'BF16 latent 升级到 2K',OpenH3RegenLora:'加载 Regenerate Turbo＋Sharp',OpenH3SaveRefinedLatent:'保存 latent',SaveVideo:'编码 MP4',CreateVideo:'合成音视频',MiniMaxClipBinPicker:'读取上一段镜头',MiniMaxPrefixCacheApplier:'保护上一段音视频前缀',MiniMaxTrimPrefixLatent:'裁掉重叠开头',MiniMaxClipBinSaver:'归档到镜头素材箱'};
let mode='face',current=null,restoreId='',jobs=[],initialized=false,uploading=false;
let previewTimer,previewController,sourceBlob='';
let materialBlobs=[],stopRequestedId='';
let draftJobs=[];
const formIds=['attentionBackend','attentionTau','skinCleanup','skinStrength','mattePrompt','extraLoras','faceRegion','faceStrength','inputMode','source','resolution','aspect','fit','seconds','variant','prompt','seed','followRef','qwenWidth','qwenHeight','refRes','qwenSteps','qwenBatch','qwenCfg','negative','qwenSampler','qwenScheduler','qwenDenoise','project','chain','continuation'];
const fileIds=['video','image','lastImage','refImages','refVideos','refAudios','qwenRefs'];
const formDefaults=Object.fromEntries(formIds.map(id=>[id,$('#'+id).value]));
const workflowForms=new Map();
function switchMode(value){
 if(uploading||value===mode)return;
 workflowForms.set(mode,{values:Object.fromEntries(formIds.map(id=>[id,$('#'+id).value])),files:Object.fromEntries(fileIds.map(id=>[id,[...$('#'+id).files]])),restoreId,current});
 const saved=workflowForms.get(value);
 for(const id of formIds)$('#'+id).value=saved?saved.values[id]:formDefaults[id];
 for(const id of fileIds){const transfer=new DataTransfer();for(const file of saved?.files[id]||[])transfer.items.add(file);$('#'+id).files=transfer.files}
 current=saved?.current||null;restoreId=saved?.restoreId||'';jobs=[];initialized=!!saved;stopRequestedId='';$('#stopHint').textContent='';
 if(sourceBlob)URL.revokeObjectURL(sourceBlob);sourceBlob='';$('#sourcePreview').hidden=true;$('#sourcePreview').removeAttribute('src');
 $('#restored').hidden=!restoreId;$('#task').hidden=!current;$('#make').disabled=false;
 setMode(value,!saved);localStorage.setItem('h3-workflow',value);$('#history').replaceChildren(new Option('正在读取当前工作流历史…',''));
 if(current){display(current);showMaterials(current)}refresh();
}

let extraLoraCatalog=[],enhancementsReady=false,attentionReady=false;
let resolutionReady=false,resolutionCheckAt=0,flexGeometryReady=false;
const duration=seconds=>{seconds=Math.max(0,Math.round(seconds||0));return Math.floor(seconds/60)+' 分 '+seconds%60+' 秒'};
function setMode(value,reset=true){
 mode=value;const gen=mode==='draft'||mode==='long';document.querySelectorAll('.mode').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));
 const people=['face','character','flex_character','quality_character'].includes(mode);
 $('#peopleModes').hidden=!people;
 document.querySelectorAll('[data-category]').forEach(b=>b.classList.toggle('active',b.dataset.category===(people?'people':mode)));
 const inputMode=$('#inputMode').value;
 $('#qwenFields').hidden=mode!=='qwen';$('#qwenWidth').disabled=$('#qwenHeight').disabled=$('#followRef').value==='true';$('#historyLabel').textContent=names[mode]+' · 独立历史任务';$('#deleteHistory').hidden=!current||!['completed','error','interrupted'].includes(current.status);
 $('#longFields').hidden=mode!=='long';$('#inputModeField').hidden=!gen;$('#fixedMode').hidden=['draft','qwen'].includes(mode);
 $('#fixedMode').textContent=mode==='refine'?'精修只有一种处理模式；下方选择输入来源。':'此工作流固定使用原视频＋参考人物图，不切换为生成模式。';
 $('#videoField').hidden=['draft','long','qwen'].includes(mode)||(mode==='refine'&&!!$('#source').value);
 $('#imageField').hidden=!(['face','character','flex_character','quality_character'].includes(mode)||(gen&&['first','first_last'].includes(inputMode)));$('#sourceField').hidden=mode!=='refine';$('#variantField').hidden=mode==='qwen';
 $('#skinStrength').disabled=$('#skinCleanup').value!=='true';
 $('#skinFields').hidden=!['face','character','flex_character','quality_character','draft','long'].includes(mode);$('#extraLoraField').hidden=mode==='qwen';renderExtraLoras();
 $('#attentionField').hidden=mode==='qwen';$('#attentionBackend').disabled=!attentionReady;$('#attentionTauField').hidden=$('#attentionBackend').value!=='sol-hip';
 $('#faceRegionField').hidden=mode!=='refine';$('#faceStrengthField').hidden=$('#faceRegion').value!=='true';
 $('#variant').disabled=mode==='refine';
 for(const option of $('#variant').options)option.disabled=mode==='refine'&&option.value!=='baseline';
 $('#variant').options[0].textContent=mode==='refine'?'Regenerate 专用精修配置（不兼容生成加速 LoRA）':'无生成加速 LoRA · BF16 基线';
 if(mode==='refine')$('#variant').value='baseline';
 if(reset&&mode!=='refine')$('#variant').value=formDefaults.variant;
 $('#imageLabel').textContent=gen?'首帧图片':'参考人物图片';$('#lastImageField').hidden=!gen||inputMode!=='first_last';$('#referenceField').hidden=!gen||inputMode!=='reference';
 $('#promptHint').textContent=['face','character'].includes(mode)?'必须保留 <Picture 1>，代表参考图片。':'底稿与精修使用同一场景描述。精修沿用已授权的 Turbo＋Sharp，底稿 LoRA 不会叠加到精修模型。';
 if(mode==='qwen')$('#promptHint').textContent='本地 Qwen Image 2.1 BF16。用 image_1、image_2 或图像1、图像2说明每张参考图的用途。批量图片共用一个任务和种子，作为整体排队。';
 if(mode==='flex_character'){
  $('#fixedMode').textContent='REF2VA BF16＋Pika Flex LoRA（4步、强度1）＋CK。分辨率选择正在部署，当前任务仍按提交时的参数执行。';
  $('#promptHint').textContent='Picture 1 为参考人物图。保留原视频声音，按 24fps 和 H3 帧网格处理全片，末尾可能少约 0.7 秒。单人工作流，长视频耗时显著增加；8.7 秒样例约 42 分钟。';
 }
 if(mode==='quality_character'){
  $('#fixedMode').textContent='REF2VA BF16 原模型＋CK，基础配置为20步；可在下方选择额外增强LoRA。默认480p，可选择分辨率与画幅；更改画幅时等比例补边或裁剪，不拉伸。';
  $('#promptHint').textContent='Picture 1 为参考人物图。保留原视频动作和原声音；H3 帧网格可能截去不足 0.7 秒的片尾。单人工作流，8.7 秒样例耗时约 58 分钟。质量改善不代表能消除所有手部和身份瑕疵。';
 }
 if(gen)$('#promptHint').textContent=(mode==='long'?'这一段的动作和声音。接力时开头重叠会被裁掉，提示词写新出现的内容。':'')+(inputMode==='reference'?'明确每份参考素材控制身份、风格、动作或声音中的哪一项；参考图不是首帧。':inputMode==='text'?'用文字描述场景、动作、镜头和声音。':'首帧是视频的开场画面，尾帧是结束画面；提示词描述中间的动作和声音。');
 if(mode==='flex_character'&&flexGeometryReady)$('#fixedMode').textContent='REF2VA BF16＋Pika 4步 LoRA＋CK；默认480p，可选择分辨率、画幅和等比例裁剪／补边。';
 if(reset){$('#prompt').value=prompts[mode];restoreId='';$('#restored').hidden=true;if(['quality_character','flex_character'].includes(mode))$('#resolution').value='480'}
 $('#geometryFields').hidden=!['draft','long','quality_character','flex_character'].includes(mode);$('#fitField').hidden=!['quality_character','flex_character'].includes(mode)&&(!gen||!['first','first_last'].includes(inputMode));
 $('#seconds').hidden=['quality_character','flex_character'].includes(mode);$('#seconds').previousElementSibling.hidden=['quality_character','flex_character'].includes(mode);$('#seconds').nextElementSibling.hidden=['quality_character','flex_character'].includes(mode);
 $('#aspect').options[0].textContent=['quality_character','flex_character'].includes(mode)?'跟随原视频比例':'跟随首帧原图（无首帧时默认横屏）';
 $('#resolutionField').hidden=!['draft','long','quality_character','flex_character'].includes(mode);
 $('#resolutionField').querySelector('p:last-child').textContent=['quality_character','flex_character'].includes(mode)?'分辨率决定短边，比例决定画面形状；实际宽高按32像素对齐，以预览为准。高分辨率显著增加耗时和内存需求，尚未逐档实测画质。':'比例决定画面形状，分辨率决定像素尺寸。二者共同计算宽高，按32像素对齐，以预览为准。高清直出尚未验收，Pika LoRA画质、耗时和内存需求可能变化。';
 if(mode==='flex_character'&&!flexGeometryReady){$('#geometryFields').hidden=true;$('#resolutionField').hidden=true;$('#fitField').hidden=true}
 updateGeometry();
 showMaterials(restoreId?jobs.find(job=>job.id===restoreId):null);
 accelerationHint();
}
function showMaterials(job){
 materialBlobs.forEach(url=>URL.revokeObjectURL(url));materialBlobs=[];
 const container=$('#materialImages');container.replaceChildren();
 const inputMode=$('#inputMode').value;
 const fields=[];
 const gen=mode==='draft'||mode==='long';
 if(['face','character','flex_character','quality_character'].includes(mode)||(gen&&['first','first_last'].includes(inputMode)))fields.push(['image',[job?.image_input].filter(Boolean),gen?'首帧':'参考人物图']);
 if(gen&&inputMode==='first_last')fields.push(['lastImage',[job?.last_image_input].filter(Boolean),'尾帧']);
 if(mode==='qwen')fields.push(['qwenRefs',job?.references?.ref_images||[],'参考图片']);
 if((mode==='draft'||mode==='long')&&inputMode==='reference')fields.push(['refImages',job?.references?.ref_images||[],'参考图片']);
 if(['face','character','flex_character','quality_character','refine'].includes(mode)&&job?.references?.ref_images?.length)fields.push(['refImages',job.references.ref_images.filter(name=>name!==job.image_input),'额外参考图片']);
 for(const [id,names,label] of fields){
  const files=[...$('#'+id).files];
  for(const [index,item] of (files.length?files:names).entries()){
   const figure=document.createElement('figure');figure.style.cssText='margin:0;max-width:100%;width:220px';
   const image=document.createElement('img');image.alt=label+' '+(index+1);image.style.cssText='width:100%;height:auto;border-radius:10px';
   let filename;
   if(item instanceof File){filename=item.name;image.src=URL.createObjectURL(item);materialBlobs.push(image.src)}
   else{filename=item;image.src=COMFY_BASE+'/view?'+new URLSearchParams({filename:item,type:'input'})}
   const caption=document.createElement('figcaption');caption.className='hint';caption.style.overflowWrap='anywhere';caption.textContent=label+' '+(index+1)+' · '+filename;
   image.onerror=()=>{caption.textContent=label+'文件不可用，请重新上传：'+filename;image.hidden=true};
   figure.append(image,caption);container.append(figure);
  }
 }
 $('#savedMaterials').hidden=!container.children.length;
}
for(const id of ['image','lastImage','refImages','qwenRefs'])$('#'+id).addEventListener('change',()=>showMaterials(restoreId?jobs.find(job=>job.id===restoreId):null));
function updateGeometry(){
 clearTimeout(previewTimer);if(previewController)previewController.abort();
 $('#geometryPreview').hidden=!['draft','long','refine','quality_character','flex_character'].includes(mode);
 if(mode==='flex_character'&&!flexGeometryReady)$('#geometryPreview').hidden=true;
 if($('#geometryPreview').hidden)return;
 if(['quality_character','flex_character'].includes(mode)){
  const file=$('#video').files[0],job=jobs.find(j=>j.id===restoreId);
  if(file&&!sourceBlob)sourceBlob=URL.createObjectURL(file);
  const url=file?sourceBlob:job?.video_input?COMFY_BASE+'/view?'+new URLSearchParams({filename:job.video_input,type:'input'}):'';
  $('#sourcePreview').hidden=!url;if(url&&$('#sourcePreview').getAttribute('src')!==url)$('#sourcePreview').src=url;
 }
 previewTimer=setTimeout(async()=>{
  previewController=new AbortController();const signal=previewController.signal;
  const form=new FormData();form.set('mode',mode);form.set('input_mode',$('#inputMode').value);form.set('aspect',['draft','long','quality_character','flex_character'].includes(mode)?$('#aspect').value:'auto');form.set('fit',$('#fit').value);form.set('seconds',mode==='draft'||mode==='long'?$('#seconds').value:($('#sourcePreview').duration||5));
  if(['draft','long','quality_character','flex_character'].includes(mode)&&resolutionReady)form.set('resolution',$('#resolution').value);
  if(['quality_character','flex_character'].includes(mode)){
   form.set('restore_job',restoreId);
   const video=$('#sourcePreview');
   if(video.videoWidth){form.set('source_width',video.videoWidth);form.set('source_height',video.videoHeight)}
   if(video.readyState>=2){const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;canvas.getContext('2d').drawImage(video,0,0);const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));if(blob)form.set('source_frame',blob,'source-frame.png')}
  }
  if((mode==='draft'||mode==='long')&&['first','first_last'].includes($('#inputMode').value)){
   form.set('restore_job',restoreId);if($('#image').files[0])form.set('image',$('#image').files[0]);
   if($('#inputMode').value==='first_last'&&$('#lastImage').files[0])form.set('last_image',$('#lastImage').files[0]);
  }
  if(mode==='refine'){
   form.set('source_job',$('#source').value);
   if($('#video').files[0]&&$('#sourcePreview').videoWidth){form.set('source_width',$('#sourcePreview').videoWidth);form.set('source_height',$('#sourcePreview').videoHeight)}
   if(!$('#source').value&&!$('#video').files[0]&&restoreId){const restored=jobs.find(j=>j.id===restoreId);if(restored){form.set('source_job',restored.source_job||restoreId)}}
  }
  try{
   const response=await fetch(BASE+'/api/preview',{method:'POST',body:form,signal});if(!response.ok)throw new Error(await response.text());const data=await response.json();
   if(signal.aborted)return;
   $('#geometryInfo').textContent=data.pending_source?'选择源视频后显示实际画幅和时长。':(mode==='refine'?'输入 '+data.width+'×'+data.height+' → 精修 '+data.target_width+'×'+data.target_height:'实际生成画布 '+data.width+'×'+data.height)+' · '+data.seconds.toFixed(3)+' 秒 · '+data.frame_count+' 帧／24fps'+(['quality_character','flex_character'].includes(mode)?' · '+($('#fit').value==='crop'?'等比例裁剪':'等比例补边')+'，不拉伸。':mode==='draft'||mode==='long'?' · 内部对齐 '+data.internal_frames+' 帧；首帧不拉伸。':' · 保持源比例，按 32 像素对齐。');
   $('#firstPreview').style.maxWidth=(data.previews||{}).last_image?'48%':'100%';
   for(const [id,key] of [['firstPreview','image'],['lastPreview','last_image']]){const src=(data.previews||{})[key];$('#'+id).hidden=!src;if(src)$('#'+id).src=src}
  }catch(error){if(error.name!=='AbortError')$('#geometryInfo').textContent='预览失败：'+error.message}
 },250);
}
for(const id of ['resolution','aspect','fit','seconds','image','lastImage'])$('#'+id).addEventListener('change',updateGeometry);
$('#video').addEventListener('change',()=>{
 if(sourceBlob)URL.revokeObjectURL(sourceBlob);sourceBlob='';
 $('#sourcePreview').hidden=mode!=='refine'||!$('#video').files[0];
 if($('#video').files[0]&&['refine','quality_character','flex_character'].includes(mode)){sourceBlob=URL.createObjectURL($('#video').files[0]);$('#sourcePreview').src=sourceBlob}else $('#sourcePreview').removeAttribute('src');updateGeometry();
});
$('#sourcePreview').addEventListener('loadedmetadata',updateGeometry);
$('#sourcePreview').addEventListener('loadeddata',updateGeometry);
document.querySelectorAll('.mode').forEach(b=>b.onclick=()=>switchMode(b.dataset.mode));$('#source').onchange=()=>setMode(mode,false);
$('#inputMode').onchange=()=>{restoreId='';$('#restored').hidden=true;setMode(mode,false)};
$('#variant').onchange=()=>accelerationHint();
function explain(text){
 if(/out of memory|allocate memory|OOM/i.test(text))return '显存／共享内存不足：检查其他 GPU 任务，稍后重试。';
 if(/not.*found|does not exist|not in list/i.test(text))return '模型或输入文件不可用：检查文件是否仍存在，以及服务是否已刷新模型列表。';
 if(/mask|detect|no.*face/i.test(text))return '检查人物是否清晰、无遮挡，以及参考图是否为清晰正脸。';
 if(/interrupt|重启|中止/i.test(text))return '恢复参数后可重新制作；不会自动重跑已中止任务。';
 return '保留任务编号和错误详情，便于定位失败节点。';
}
function display(job,restore=false){
 if(job.mode!==mode)return;current=job;localStorage.setItem('h3-last-job-'+mode,job.id);$('#task').hidden=false;
 if(restore&&stopRequestedId!==job.id)$('#stopHint').textContent='';
 $('#jobId').textContent='任务 '+job.id;$('#taskTitle').textContent=names[job.mode]+' · '+states[job.status];
 $('#profileDetails').hidden=!job.profile_path;
 $('#profileData').textContent=JSON.stringify({profile_path:job.profile_path,parameters:job.parameters||{mode:job.mode,seed:job.seed,steps:job.steps,resolution:job.resolution,width:job.width,height:job.height,seconds:job.seconds,variant:job.variant},asset_paths:job.asset_paths,references:job.references,prompt:job.prompt_text,effective_prompt:job.effective_prompt,skin_artifacts:job.skin_artifacts},null,2);
 $('#make').disabled=uploading;$('#deleteHistory').hidden=!['completed','error','interrupted'].includes(job.status);
 const stoppable=['queued','running'].includes(job.status);
 $('#stop').hidden=!stoppable;$('#stop').disabled=stopRequestedId===job.id;
 $('#stop').textContent=stopRequestedId===job.id?'已请求停止，等待中断…':job.status==='queued'?'取消当前排队任务':'停止当前任务';
 if(stopRequestedId===job.id&&!stoppable){stopRequestedId='';$('#stopHint').textContent=job.status==='interrupted'?'已停止。提示词、图片和参数已保留；重新制作会从头开始。':'任务已结束。'}
 const label=job.mode==='qwen'&&job.stage==='UNETLoader'?'加载 Qwen Image 2.1 BF16':job.mode==='qwen'&&job.stage==='VAEDecode'?'Qwen VAE 图片解码':stages[job.stage]||job.stage;
 if(job.status==='queued')$('#status').textContent='排队中，前面还有 '+(job.queue_ahead||0)+' 个任务。';
 else if(job.status==='running')$('#status').textContent=label+(job.total?'：'+job.step+' / '+job.total+(job.step===job.total?'（已到末步，等待本阶段收尾）':'（当前阶段 '+Math.floor(job.step/job.total*100)+'%）'):'：处理中，等待阶段进度。');
 else $('#status').textContent=job.status==='finalizing'?(job.mode==='qwen'?'正在校验和保存图片。':'正在完整解码校验成片；通过后才标记完成。'):states[job.status];
 if(job.status==='completed')$('#progress').value=100;else if(job.status==='running'&&job.total)$('#progress').value=job.step/job.total*100;else $('#progress').removeAttribute('value');
 $('#layers').hidden=job.status!=='running'||!job.layers;
 if(job.layers){$('#layerLabel').textContent='正在提交模型层 '+job.layer+' / '+job.layers+'（不是 GPU 完成率）';$('#layerProgress').max=job.layers;$('#layerProgress').value=job.layer;}
 $('#elapsed').textContent='已用时 '+duration(job.elapsed)+(job.stage_eta?' · 当前阶段预计剩余 '+duration(job.stage_eta)+'（估算，不含后续阶段）':'');
 $('#jobGeometry').textContent=job.mode==='qwen'?job.width+'×'+job.height+' · '+(job.batch||1)+' 张 · '+(job.steps||40)+' 步':job.width?(job.target_width?'成片 '+job.target_width+'×'+job.target_height:'成片 '+job.width+'×'+job.height)+' · '+job.seconds+' 秒 · '+job.frame_count+' 帧':'';
 $('#timings').textContent=(job.timings||[]).filter(t=>t.seconds>=1).map(t=>(stages[t.stage]||t.stage||'节点 '+t.node)+'：'+duration(t.seconds)).join(' · ');
 $('#errorBox').hidden=!job.error;$('#error').textContent=job.error?explain(job.error)+'\n\n'+job.error:'';$('#result').hidden=!job.video_url&&!job.images?.length;$('#output').hidden=!job.video_url;$('#download').hidden=!job.video_url;
 if(job.video_url){const url=BASE+job.video_url.slice('/simple-swap'.length);if($('#output').getAttribute('src')!==url)$('#output').src=url;$('#download').href=url;$('#outputPath').textContent=job.video_path;}
 const gallery=$('#imageResults');
 const imageKey=JSON.stringify(job.images||[]);
 if(gallery.dataset.images!==imageKey){gallery.dataset.images=imageKey;gallery.replaceChildren();for(const [index,item] of (job.images||[]).entries()){
 const url=BASE+item.url.slice('/simple-swap'.length),image=document.createElement('img'),link=document.createElement('a');image.src=url;image.alt='生成图片 '+(index+1);image.style.cssText='width:100%;height:auto;border-radius:12px';link.href=url;link.download='qwen-'+job.id+'-'+(index+1)+'.png';link.className='download';link.textContent='下载图片 '+(index+1);gallery.append(image,link)}}
 if(job.mode==='qwen')$('#outputPath').textContent=(job.images||[]).map(item=>item.path).join('\n');
 $('#toRefine').hidden=job.mode!=='draft'||!job.latent_file||job.status!=='completed';
 if(restore){setMode(job.mode,false);if(job.project)$('#project').value=job.project;if(job.chain)$('#chain').value=job.chain;if(job.continuation_frames)$('#continuation').value=job.continuation_frames;$('#inputMode').value=job.input_mode||'text';$('#aspect').value=job.aspect||'auto';$('#fit').value=job.fit||'pad';$('#seconds').value=job.seconds||5;$('#prompt').value=job.prompt_text;$('#seed').value=job.seed;$('#variant').value=job.variant||'baseline';$('#source').value=job.source_job||'';$('#faceRegion').value=String(job.face_region===true);$('#faceStrength').value=job.face_strength??0.55;$('#attentionBackend').value=['sol-hip','sol-attn-hip'].includes(job.attention_backend)?'sol-hip':'ck';$('#attentionTau').value=job.attention_tau??0.5;$('#skinCleanup').value=String(job.skin_cleanup===true);$('#skinStrength').value=job.skin_strength??0.9;$('#mattePrompt').value=String(job.matte_prompt===true);$('#extraLoras').value=JSON.stringify(job.extra_loras||[]);restoreId=job.id;for(const id of fileIds)$('#'+id).value='';if(sourceBlob)URL.revokeObjectURL(sourceBlob);sourceBlob='';$('#sourcePreview').hidden=true;$('#sourcePreview').removeAttribute('src');$('#restored').hidden=false;$('#restored').textContent='已恢复参数'+(job.video_input?' · 原视频已保留':'')+(job.image_input?' · 首帧／参考图已保留':'')+(job.last_image_input?' · 尾帧已保留':'')+(Object.values(job.references||{}).some(files=>files.length)?' · 多模态参考素材已保留':'')+'。无需重复上传；选择新文件可替换。';setMode(mode,false);showMaterials(job);}
 if(restore){if(job.variant==='ref2va_flex4')$('#variant').value='pika4';$('#resolution').value=job.resolution||'768';if(job.mode==='qwen')for(const [id,key] of [['followRef','follow_ref'],['qwenWidth','width'],['qwenHeight','height'],['refRes','ref_res'],['qwenSteps','steps'],['qwenBatch','batch'],['qwenCfg','cfg'],['negative','negative'],['qwenSampler','sampler'],['qwenScheduler','scheduler'],['qwenDenoise','denoise']])$('#'+id).value=job[key]===undefined?formDefaults[id]:String(job[key]);$('#qwenWidth').disabled=$('#qwenHeight').disabled=$('#followRef').value==='true';updateGeometry()}
 if(experimentNames[job.variant])$('#fixedMode').textContent='当前任务：'+experimentNames[job.variant]+' · CK。恢复会保留所选加速方案；重新制作会重新准备输入，不自动读取实验缓存。';
}
$('#stop').onclick=async()=>{
 const job=current;if(!job||!['queued','running'].includes(job.status))return;
 $('#stop').disabled=true;
 try{
  const response=await fetch(COMFY_BASE+'/queue',{cache:'no-store'});if(!response.ok)throw new Error(await response.text());const queue=await response.json();
  const running=queue.queue_running.some(item=>item[1]===job.id),pending=queue.queue_pending.some(item=>item[1]===job.id);
  if(running||pending){
   const stopped=await fetch(COMFY_BASE+(running?'/interrupt':'/queue'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(running?{prompt_id:job.id}:{delete:[job.id]})});
   if(!stopped.ok)throw new Error(await stopped.text());
   stopRequestedId=job.id;
   if(current?.id===job.id){$('#stop').textContent='已请求停止，等待中断…';$('#stopHint').textContent='仅停止此任务。正在运行的 GPU 操作可能需要等待安全中断点；停止后不能从原采样步续跑。'}
  }else if(current?.id===job.id)$('#stopHint').textContent='任务已不在执行队列，无需中断。';
  await refresh();
 }catch(error){if(current?.id===job.id){$('#stop').disabled=false;$('#stopHint').textContent='无法确认停止结果：'+error.message+'。请刷新查看任务状态。'}}
};
async function refresh(){
 try{
  if((!resolutionReady||!enhancementsReady||!attentionReady)&&Date.now()-resolutionCheckAt>15000){
   resolutionCheckAt=Date.now();const options=await fetch(BASE+'/api/options',{cache:'no-store'});
   if(options.ok){const data=await options.json();attentionReady=data.attention_backends?.includes('sol-hip')===true;extraLoraCatalog=data.extra_loras||[];enhancementsReady=data.skin_preparation===true;renderExtraLoras();resolutionReady=data.resolutions?.includes('768');flexGeometryReady=data.flex_geometry===true;$('#resolution').disabled=!resolutionReady;if(resolutionReady){$('#resolutionHint').textContent='已生效。选择档位后，下面会显示对应的实际宽高。';setMode(mode,false)}}
  }
  const requestedMode=mode;
  const responses=await Promise.all([fetch(BASE+'/api/jobs',{cache:'no-store'}),fetch(BASE+'/api/jobs?mode=draft',{cache:'no-store'}),fetch(COMFY_BASE+'/queue',{cache:'no-store'})]);
  for(const response of responses)if(!response.ok)throw new Error(await response.text());
  const [items,drafts,queue]=await Promise.all(responses.map(response=>response.json()));if(requestedMode!==mode)return;
  const runningIds=new Set(queue.queue_running.map(item=>item[1]));
  for(const job of items)if(job.status==='queued'&&runningIds.has(job.id))job.status='running';
  jobs=items.filter(job=>job.mode===mode);draftJobs=drafts.filter(job=>job.mode==='draft');
  $('#queueSummary').textContent='共享 GPU 队列：执行中 '+queue.queue_running.length+' 个 · 等待 '+queue.queue_pending.length+' 个';
  let activeJobs=$('#activeJobs');
  if(!activeJobs){activeJobs=document.createElement('div');activeJobs.id='activeJobs';$('#queueSummary').after(activeJobs)}
  activeJobs.replaceChildren();
  for(const job of items.filter(j=>['queued','running','finalizing'].includes(j.status))){
   const button=document.createElement('button');button.type='button';button.className='mode';
   button.textContent=states[job.status]+' · '+(experimentNames[job.variant]||names[job.mode])+' · '+job.id.slice(0,8)+' · 点击查看';
   button.onclick=()=>{if(uploading)return;switchMode(job.mode);display(job,true);$('#task').scrollIntoView({block:'start'})};
   activeJobs.append(button);
  }
  const history=$('#history'),source=$('#source'),oldSource=source.value;
  history.replaceChildren(new Option('选择 '+names[mode]+' 历史任务',''));source.replaceChildren(new Option('上传底稿视频',''));
  jobs.forEach(job=>{history.add(new Option(states[job.status]+' · '+(experimentNames[job.variant]||names[job.mode])+' · '+new Date(job.created_at*1000).toLocaleString(),job.id))});
  draftJobs.forEach(job=>{if(job.status==='completed'&&job.latent_file)source.add(new Option(new Date(job.created_at*1000).toLocaleString(),job.id))});source.value=oldSource;
  if(!initialized){const selected=jobs.find(j=>j.id===localStorage.getItem('h3-last-job-'+mode))||jobs.find(j=>['queued','running','finalizing'].includes(j.status))||jobs[0];if(selected)display(selected,true);initialized=true;}
  else if(current){const updated=jobs.find(j=>j.id===current.id);if(updated)display(updated);}
  if(current)history.value=current.id;
 }catch(error){$('#task').hidden=false;$('#status').textContent='连接暂时中断，正在重连。任务不会重新提交。'+error.message.slice(0,120);}
}
$('#history').onchange=()=>{const job=jobs.find(j=>j.id===$('#history').value);if(job)display(job,true)};
$('#toRefine').onclick=()=>{
 const draft=current;switchMode('refine');current=null;restoreId='';initialized=true;$('#task').hidden=true;$('#video').value='';$('#source').value=draft.id;$('#prompt').value=draft.prompt_text;$('#seed').value=draft.seed;setMode('refine',false);$('#make').disabled=false;
 $('#restored').hidden=false;$('#restored').textContent='已选择这份底稿，提示词和种子已带入。确认后点击制作开始 2K 精修。';
 $('#sourceField').scrollIntoView({block:'start'});$('#source').focus({preventScroll:true});
};
$('#newTask').onclick=()=>{
 if(uploading)return;
 for(const id of formIds)$('#'+id).value=formDefaults[id];
 for(const id of fileIds)$('#'+id).value='';
 current=null;restoreId='';workflowForms.delete(mode);$('#task').hidden=true;$('#restored').hidden=true;$('#deleteHistory').hidden=true;$('#history').value='';
 setMode(mode,true);showMaterials(null);
};
$('#make').onclick=async()=>{
 if(uploading)return;const form=new FormData();form.append('mode',mode);form.append('prompt',$('#prompt').value);form.append('seed',$('#seed').value);form.append('variant',$('#variant').value);form.append('restore_job',restoreId);
 const inputMode=mode==='draft'||mode==='long'?$('#inputMode').value:'text';form.set('input_mode',inputMode);
 if(mode!=='qwen'){if(!enhancementsReady){$('#status').textContent='后端尚未加载新工作流，请稍后刷新。';return}const skin=!$('#skinFields').hidden;form.set('skin_cleanup',skin?$('#skinCleanup').value:'false');form.set('skin_strength',$('#skinStrength').value);form.set('matte_prompt',skin?$('#mattePrompt').value:'false');form.set('extra_loras',$('#extraLoras').value);if($('#attentionBackend').value==='sol-hip'&&!attentionReady){$('#status').textContent='注意力选项尚未加载，请刷新后重试。';return}form.set('attention_backend',$('#attentionBackend').value);form.set('attention_tau',$('#attentionTau').value)}
 if(mode==='draft'||mode==='long'){form.set('aspect',$('#aspect').value);form.set('fit',$('#fit').value);form.set('seconds',$('#seconds').value)}
 if(mode==='long'){form.set('project',$('#project').value);form.set('chain',$('#chain').value);form.set('continuation_frames',$('#continuation').value)}
 if(['draft','long','quality_character','flex_character'].includes(mode)&&resolutionReady)form.set('resolution',$('#resolution').value);
 if(['quality_character','flex_character'].includes(mode)){form.set('aspect',$('#aspect').value);form.set('fit',$('#fit').value)}
 if(mode==='refine'){form.append('source_job',$('#source').value);form.set('face_region',$('#faceRegion').value);form.set('face_strength',$('#faceStrength').value)}
 if(!['draft','long','qwen'].includes(mode)&&$('#video').files[0])form.append('video',$('#video').files[0]);if(!$('#imageField').hidden&&$('#image').files[0])form.append('image',$('#image').files[0]);
 if((mode==='draft'||mode==='long')&&inputMode==='first_last'&&$('#lastImage').files[0])form.append('last_image',$('#lastImage').files[0]);
 if((mode==='draft'||mode==='long')&&inputMode==='reference')for(const [id,name] of [['refImages','ref_images'],['refVideos','ref_videos'],['refAudios','ref_audios']])for(const file of $('#'+id).files)form.append(name,file);
 if(mode==='qwen'){
 for(const [id,key] of [['followRef','follow_ref'],['qwenWidth','width'],['qwenHeight','height'],['refRes','ref_res'],['qwenSteps','steps'],['qwenBatch','batch'],['qwenCfg','cfg'],['negative','negative'],['qwenSampler','sampler'],['qwenScheduler','scheduler'],['qwenDenoise','denoise']])form.set(key,$('#'+id).value);
 for(const file of $('#qwenRefs').files)form.append('ref_images',file);
 }
 uploading=true;$('#make').disabled=true;$('#task').hidden=false;$('#result').hidden=true;
 try{
  const job=await new Promise((resolve,reject)=>{const xhr=new XMLHttpRequest();xhr.open('POST',BASE+'/api/jobs');xhr.upload.onprogress=e=>{if(e.lengthComputable){$('#progress').value=e.loaded/e.total*100;$('#status').textContent='上传素材 '+Math.floor(e.loaded/e.total*100)+'%（不是生成进度）'}};xhr.onload=()=>{if(xhr.status>=200&&xhr.status<300)resolve(JSON.parse(xhr.responseText));else reject(new Error(xhr.responseText))};xhr.onerror=()=>reject(new Error('网络中断，请刷新检查历史任务后再重试，避免重复提交。'));xhr.send(form);});
  display(job,true);await refresh();
 }catch(error){$('#status').textContent='提交失败：'+error.message;$('#make').disabled=false;}
 finally{uploading=false;$('#make').disabled=false;}
};
$('#deleteHistory').onclick=async()=>{
 const job=current;if(!job||!confirm('删除当前工作流的这条任务记录？已生成的文件会保留。'))return;
 try{const response=await fetch(BASE+'/api/jobs/'+job.id+'?mode='+mode,{method:'DELETE'});if(!response.ok)throw new Error(await response.text());current=null;restoreId='';localStorage.removeItem('h3-last-job-'+mode);workflowForms.delete(mode);$('#task').hidden=true;$('#deleteHistory').hidden=true;$('#restored').hidden=true;showMaterials(null);await refresh()}catch(error){$('#status').textContent=error.message}
};
$('#followRef').onchange=()=>{$('#qwenWidth').disabled=$('#qwenHeight').disabled=$('#followRef').value==='true'};
setMode(names[localStorage.getItem('h3-workflow')]?localStorage.getItem('h3-workflow'):'flex_character');refresh();setInterval(()=>{if(!uploading)refresh()},3000);

$('#faceRegion').onchange=()=>{$('#faceStrengthField').hidden=$('#faceRegion').value!=='true'};

function renderExtraLoras(){
 const container=$('#extraLoraList');container.replaceChildren();
 let selected=[];try{selected=JSON.parse($('#extraLoras').value)}catch{}
 const available=extraLoraCatalog.filter(item=>item.modes.includes(mode));
 const sync=()=>{selected=[...container.children].filter(row=>row.querySelector('input[type=checkbox]').checked).map(row=>({name:row.dataset.name,strength:Number(row.querySelector('input[type=number]').value)}));$('#extraLoras').value=JSON.stringify(selected);updateHint()};
 for(const item of available){
  const row=document.createElement('div');row.dataset.name=item.name;row.style.cssText='display:flex;align-items:center;gap:8px;margin:8px 0';
  const label=document.createElement('label');label.style.cssText='flex:1;margin:0;font-weight:400';
  const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.style.cssText='width:auto;margin-right:8px';const saved=selected.find(x=>x.name===item.name);checkbox.checked=!!saved;
  label.append(checkbox,document.createTextNode(item.label+' · '+item.weight_precision.map(value=>({F16:'FP16',F32:'FP32'}[value]||value)).join('/')));if(item.hint){const hint=document.createElement('small');hint.className='hint';hint.style.display='block';hint.textContent=item.hint;label.append(hint)}
  const strength=document.createElement('input');strength.type='number';strength.min=-2;strength.max=2;strength.step=.05;strength.value=saved?.strength??item.default_strength;strength.style.width='85px';strength.setAttribute('aria-label',item.label+' 强度');strength.disabled=!checkbox.checked;
  checkbox.onchange=()=>{strength.disabled=!checkbox.checked;sync()};strength.onchange=sync;row.append(label,strength);container.append(row);
 }
 function updateHint(){
  $('#extraLoraHint').textContent=!enhancementsReady?'正在读取增强选项…':available.length?(selected.length?'已选择 '+selected.length+' 项，触发词会自动加入提示词。':'未选择增强 LoRA，使用当前基础配置。'):'当前流程没有适用的增强 LoRA。';
  const missing=selected.filter(x=>!available.some(a=>a.name===x.name));if(missing.length)$('#extraLoraHint').textContent+=' 记录中的 LoRA 已不可用：'+missing.map(x=>x.name).join('、');
 }
 updateHint();
}

$('#skinCleanup').onchange=()=>{$('#skinStrength').disabled=$('#skinCleanup').value!=='true'};

$('#attentionBackend').onchange=()=>{$('#attentionTauField').hidden=$('#attentionBackend').value!=='sol-hip';accelerationHint()};
