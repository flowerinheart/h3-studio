# MiniMax H3 simple WebUI

Image upload update 2026-11-04: shared reference upload accepts JPG/JPEG, PNG,
WebP, BMP, TIFF and GIF. Pillow verifies the actual image encoding and decoding;
incorrect image extensions are corrected without changing original file bytes.
PNG RGB/RGBA/palette/16-bit variants are covered. Invalid/truncated images and
uploads over the existing 25 MiB limit are rejected and temporary files removed.
All image pickers advertise the supported extensions. No generation model changes.
Validation: scripts/test_h3_image_upload.py (4 tests, 9 encoding/mode cases),
live Qwen prepare endpoint (9 cases, no inference); evidence in
outputs/tests/h3-image-formats/live-prepare.json.

Quality geometry update2026-11-03: quality_character now offers the existing
384/480/576/768/1080/1440short-edge presets plus auto/9:16/16:9/1:1/4:3/3:4
aspect selection. Default480/auto retains the original verified graph path.
Other sizes/explicit aspects connect source video frames through the existing
H3WebUIFitImage crop/pad node before every frame consumer, and set Swap Plan's
short_edge to the aligned canvas minimum. Original source audio is untouched.
No aspect stretch, no diffusion/LoRA/steps changes, no user duration override.
Mobile previews use the actual source video frame and shared placement helper.
Metadata stores resolution/aspect/fit; history restore and refresh restore them.
Tests:14unit tests including72geometry variants;4mobile button submissions
intercepted to prepare; native60-node validation; public mobile preview;
mocked historical geometry reload; original quality/Flex mobile regression.
No new GPU render or high-resolution quality/performance claim.
Evidence: outputs/h3-quality-geometry-20261103.

Quality-mode deployment2026-11-03: `quality_character` is a separate
"视频换人 · BF16 质量档" entry based on the verified20-step480p graph in
`ref2va-quality.api.json`. No LoRA, local REF2VA BF16, dense CK, Euler/simple/CFG1,
shifts12/3, motion512 and loose mask unchanged. Source aspect follows the native
480-short-edge32px grid (1080x1920 source→480x864); original audio retained.
Only materials/prompt/seed/unique run identity are patched. Existing Flex mode
remains720p/4step and has separate history. Verified20step result imported with
its reference image, original video, full prompt and seed. Same progress, scoped
stop, reload and final allowed-root decode verification are reused.
Tests:13unit tests, local/public mobile restore/playback/reload and intercepted
make-button preparation, native59-node validation; no new inference.
Evidence: outputs/h3-webui-ref2va-quality-20261103.

REF2VA Flex addition2026-11-02: separate `flex_character` mode uses the frozen,
verified720p graph in `ref2va-flex.api.json`, not the old20-step character route.
H3 REF2VA BF16 + Pika REF2VA Flex4 LoRA strength1, CK dense attention,
Euler/simple/CFG1, shifts12/3, motion512, loose SAM mask, clip9.5s, overlap39.
Only video/image/prompt/seed and unique output/run identity are patched.
Full source is loaded at24fps; H3's17k+5 grid can drop up to16trailing frames.
Native short edge736, deterministic aspect-preserving export to short edge720
(H.264 requires even dimensions). No new model or precision substitution.
H3/TE BF16; native video/audio VAE files FP16/FP32 (runtimeBF16), SAM fileFP16.
The verified720p example is imported into its independent history. Generic
default prompt is reference-agnostic; restoring the example restores its exact
character-specific prompt and retained source/reference files.
All old modes remain. New jobs use the existing progress/stop/history/finalizer;
every final MP4 is decoded and stored in the allowed outputs root.
Checks:12unit tests, real mobile history/image/video reload and intercepted
make-button prepare, native60-node validation. No new diffusion render.
Evidence: outputs/h3-webui-ref2va-flex-20261102.

Resolution update2026-11-02: draft is named "生成底稿", not locked to768p.
Separate linked resolution(short-edge target) and aspect selectors compute the
actual32px-aligned canvas. Presets384/480/576/768/1080/1440; last two experimental.
For example768+16:9→1376x768,768+square→768x768,1440+9:16→1440x2560.
Metadata persists resolution and history restores it. Existing API callers that
omit resolution retain their prior canvas behavior; no running graph is changed.
Frontend checks /simple-swap/api/options before enabling/sending resolution.
Staged backend loads only when queue and finalization are idle, through
scripts/h3_webui_deploy_resolution.py. It never interrupts a busy render.
Nine unit tests and36isolated prepare/native-validation combinations passed.
Public UI rename/gating passed; linked-preview test used mocked capability/preview
while the live backend was busy. No new render or high-resolution quality claim.
Evidence: outputs/tests/h3-webui-resolution-2026-11-02.

Local `/simple-swap/`; existing authenticated public prefix `/h3/simple-swap/`.
The existing secret-cookie entry also remains supported. No access-control
changes are needed. The JavaScript adapts API/video paths to the public prefix.

Seven workflows: REF2VA20-step quality, verified REF2VA Flex, Qwen Image 2.1 and original Openjutsu face/character swaps; aspect/duration-configurable draft
(8-step BF16 baseline or authorized community Pika Flex Turbo4); saved-latent
or uploaded-video 2K refine (existing Regenerate Turbo+Sharp,3Euler,dense CK).
Uploaded refinement resamples to24fps and preserves the full clip rather than
truncating to121frames. Short5–15second drafts remain recommended.

Geometry fix2026-11-02: first-frame default follows the image aspect, on a32px
aligned canvas. Explicit aspect uses user-selected center crop or black padding,
never direct stretch. CPU preview shares placement with render preprocessing;
both first/last frames pass through H3WebUIFitImage before native conditioning.
Duration writes both internal17k+5 grid length and delivered24fps frame count.
5/10/15seconds produce120/240/360frames (internal124/243/362).
Historical restore preserves aspect,fit,duration; old tasks keep their existing
videos. Refinement computes an aspect-aware2K-budget target, retaining an exact
ratio where32px alignment permits (e.g.768square→1920square,
864x1536→1440x2560,1344x768→2464x1408); no forced16:9 intermediate.
For other ratios, nearest32px alignment may introduce a small ratio difference.
The Open-H3 upscale node accepts optional target_width/target_height; defaults
remain unchanged for old workflows. Uploads are normalized without cropping,
with padding for alignment. No new model/kernel/precision was introduced.
Evidence: outputs/tests/h3-webui-geometry-2026-11-02; browser prepare-only
submissions, native graph validation, CPU fit/upscale/frame-plan tests passed.
No new diffusion render was performed for this fix.

Draft input modes: text, first-frame, first/last-frame (FL2VA BF16 baseline or
authorized Pika4), and multimodal reference (official REF2VA BF16,20steps,
no FL2VA Pika LoRA). Reference uploads support multiple images/videos/audio;
video references are normalized to24fps and at most121frames. Prompt labels
are1-based per modality. History preserves input mode, both keyframe names
and reference lists; restored files are reused unless replacements are uploaded.
Face/character workflows remain fixed video edits; refinement has one processing
mode with two input sources. Audio-only generation is not added to these workflows.
Reference-video soundtracks receive Audio labels before standalone audio clips.
Seven prepared variants passed ComfyUI's actual validate_prompt without enqueue;
public mobile switching and mocked history refresh passed. New reference/keyframe
paths have graph validation, not completed-render or Pika quality acceptance.
Evidence: outputs/tests/h3-webui-modes-2026-11-02.

Draft LoRA is isolated from the refine model. Pika4 uses the full FL2VA LoRA,
strength1,CFG1,Euler/simple,video shift6/audio3. Native H3 encoders/VAEs only.
Diffusion/text/upscaler BF16; native video VAE fileFP16/audio VAE fileFP32,
runtimeBF16; LoRA alpha and softmax/accumulation includeFP32.

Task records and API graphs are persisted atomically under outputs/h3-webui/jobs/<workflow>. Old flat records are migrated without changing task IDs.
Progress is real per-stage/per-sampler progress, NOT a guessed whole-job
percentage. Current-stage ETA is an estimate; downstream work is excluded.
An independent50-layer bar reports model-layer submission, not GPU completion;
sampler maximum still waits for stage cleanup and verified MP4 finalization.
Closing the page does not cancel or resubmit a job. On service restart, queued
or running tasks missing from the queue/history are marked interrupted.
Completed history restores prompt,seed,variant,input references and video.

Restore/stop update2026-11-02: restored original first/last/person/reference
images are displayed beside the inputs via the existing authenticated ComfyUI
view route; file inputs stay empty because retained server files are reused.
New uploads replace the displayed materials. Full prompt restoration is retained;
progress polling does not overwrite ongoing edits.
Stop targets only the selected prompt_id through existing /interrupt, and queued
cancel deletes only that id through /queue. UI waits for actual interrupted state;
no resumable sampling checkpoint/pause is claimed. Finalizing jobs are not stopped.
Frontend-only deployment did not restart the backend or stop the live10s job.
Public mobile tests covered private-link/basic-auth original image+prompt restore,
first/last and multireference displays, editing persistence and scoped stop/cancel.
Stop requests were intercepted in tests to protect live inference.
Evidence: outputs/tests/h3-webui-restore-stop-2026-11-02/browser.json.

Final MP4s are fully decoded before completion, moved to outputs/h3-webui/videos
inside the Paseo allowlist, retaining source symlinks for application links.
Latents remain in ComfyUI/output/open_h3_latents for the existing loader.

Tests: `python3 scripts/test_h3_webui.py`, mobile/public browser checks and
serial full-render controller in `scripts/h3_webui_benchmark.py`.
Evidence: outputs/tests/h3-webui-2026-11-02.

Measured one5-second sample each:8-step baseline1195.63s, Pika4 637.27s
(46.7% less total time). Pika4 remains experimental because schedules and style
differ and a single sample does not establish motion/audio quality equivalence.

Qwen integration2026-11-02: the same /simple-swap/ workspace now includes
Qwen Image2.1 BF16 text generation, multi-image reference/edit, custom or
first-reference canvas, steps/CFG/negative prompt, sampler/scheduler/denoise,
and1–8images per task. Diffusion, text encoder and VAE use the existing BF16
files; no quantization or LoRA is added. Qwen results are verified PNGs under
outputs/h3-webui/images/<task>.

Each workflow has its own history, selected task, input files and in-page
form state. GET api/jobs?mode=<workflow> filters history; deleting a terminal
record requires the matching mode and leaves media intact. Restore cannot
reuse another workflow's materials; refine explicitly accepts completed draft
latents. All submissions use the same ComfyUI /prompt queue, including batches.
Cancellation targets only the selected prompt ID. The standalone Qwen service
is no longer needed for this workspace and remains stopped. Its51existing
history records were imported;11missing media files were noted.

Checks: scripts/test_h3_webui.py; scripts/h3_qwen_integration_test.py against
a CPU-only server on18188; scripts/h3_qwen_validate.py. Tests cover native
graph validation, mixed FIFO queue/cancel, mobile history/form isolation,
PNG serving, draft-to-refine handoff, and intercepted submissions/interrupts.
Evidence: outputs/tests/h3-qwen-integration. No GPU inference was performed.

FaceRegion optional refine (2026-11-05): video refinement exposes an off-by-default
face protection selector and face_strength (default0.55). Enabled graphs use
OpenH3FaceRegionMap + RegionNoise + RegionModel together, draft audio conditioning,
and Kijai rank384 BF16 refiner only (no Turbo/Sharp). Uploaded video uses prepared
frames; saved-latent input decodes the source latent for detection. The existing
3-step schedule, dense CK, native VAE, geometry and original final audio remain.
Selection, strength and exact graph persist in task profiles and restore from history.
Diffusion/text/refiner/latent-upscaler weights: BF16; supporting video/audio VAE files:
FP16/FP32 respectively (server video-VAE runtime BF16 flag retained).
Source: https://github.com/BennyDaBall930/ComfyUI-Open-H3-FaceRegion
Pinned clone:620d956d2fe4aaf5aa7f67ddac0ab792e61287e9. Included YuNet detector runs on CPU.
Refiner source:https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/loras
SHA256:4bf30291343eef5e51ee867ba37dd31bce3133d1da1ace1a245a7130b88818c2
Checks:scripts/h3_webui_face_region_test.py --cpu; no generation submitted.
