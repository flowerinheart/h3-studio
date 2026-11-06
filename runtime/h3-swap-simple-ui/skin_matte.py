"""CPU face-mesh highlight attenuation; no image generation or lip recoloring."""
import json
import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'skin_models/face_landmarker.task'


def skin_mask(shape, landmarks, connections):
    h, w = shape[:2]
    points = np.array([(p.x * w, p.y * h) for p in landmarks], np.int32)
    mask = np.zeros((h, w), np.uint8)
    def polygon(edges):
        indices = sorted({e.start for e in edges} | {e.end for e in edges})
        return cv2.convexHull(points[indices])
    cv2.fillConvexPoly(mask, polygon(connections.FACE_LANDMARKS_FACE_OVAL), 255)
    exclude = np.zeros_like(mask)
    for name in ('LIPS', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EYEBROW', 'RIGHT_EYEBROW'):
        cv2.fillConvexPoly(exclude, polygon(getattr(connections, 'FACE_LANDMARKS_' + name)), 255)
    # Nostril landmarks, protected independently of specular intensity.
    for index in (98, 327):
        cv2.circle(exclude, tuple(points[index]), max(2, int(w * .006)), 255, -1)
    radius = max(1, int(np.ptp(points[:, 0]) * .018))
    exclude = cv2.dilate(exclude, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2))
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2))
    mask[exclude > 0] = 0
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), max(1., radius / 2))
    soft[exclude > 0] = 0
    soft[mask == 0] = 0
    return soft



def detect_faces(frame, detector, mp):
    """Find small faces in crops; reuse the already-installed YuNet detector."""
    rgb = np.ascontiguousarray((np.clip(frame, 0, 1) * 255).astype(np.uint8))
    h, w = rgb.shape[:2]
    scale = min(1, 1024 / max(h, w))
    reduced = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
    found = list(detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(reduced))).face_landmarks)
    yunet = ROOT.parent / 'ComfyUI-Open-H3-FaceRegion/models/face_detection_yunet_2023mar.onnx'
    if not yunet.is_file():
        return found
    boxes = cv2.FaceDetectorYN.create(str(yunet), '', (w, h), .5, .3, 5000).detect(np.ascontiguousarray(rgb[:, :, ::-1]))[1]
    for box in boxes if boxes is not None else []:
        x, y, bw, bh = box[:4]
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        if any(abs(np.mean([p.x for p in face]) - cx) < bw / w and abs(np.mean([p.y for p in face]) - cy) < bh / h for face in found):
            continue
        x0, y0 = max(0, int(x - bw * .3)), max(0, int(y - bh * .3))
        x1, y1 = min(w, int(x + bw * 1.3)), min(h, int(y + bh * 1.3))
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        if not crop.size:
            continue
        detected = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)).face_landmarks
        if not detected:
            continue
        transformed = []
        for point in detected[0]:
            p = copy.copy(point)
            p.x, p.y = (x0 + point.x * (x1 - x0)) / w, (y0 + point.y * (y1 - y0)) / h
            transformed.append(p)
        found.append(transformed)
    return found


def attenuate(rgb, mask, strength):
    """Subtract only low-frequency, bright near-neutral excess; retain high-frequency detail."""
    if not np.any(mask) or strength == 0:
        return rgb.copy(), np.zeros(mask.shape, np.float32)
    face_area = np.count_nonzero(mask > .2)
    radius = max(3., np.sqrt(face_area) * .09)
    smooth = cv2.GaussianBlur(rgb, (0, 0), radius)
    local = cv2.GaussianBlur(rgb, (0, 0), radius * 2)
    luma = smooth @ np.array([.2126, .7152, .0722], np.float32)
    surrounding = local @ np.array([.2126, .7152, .0722], np.float32)
    excess = np.maximum(luma - surrounding - .015, 0)
    saturation = (smooth.max(-1) - smooth.min(-1)) / np.maximum(smooth.max(-1), .001)
    weight = np.clip((luma - .45) / .3, 0, 1) * np.clip((.55 - saturation) / .4, 0, 1)
    amount = np.minimum(excess * weight * strength, .16) * mask
    out = np.clip(rgb - amount[..., None], 0, 1)
    out[mask == 0] = rgb[mask == 0]
    return out, amount


class H3WebUISkinMatte:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'strength': ('FLOAT', {'default': .9, 'min': 0., 'max': 1.}),
                             'filename_prefix': ('STRING', {'default': 'skin_matte'})}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'clean'
    CATEGORY = 'MiniMax H3/WebUI'

    def clean(self, images, strength, filename_prefix):
        if not MODEL.is_file():
            raise RuntimeError('Missing CPU face-landmarker asset: ' + str(MODEL))
        vendor = str(ROOT / 'skin_vendor')
        sys.path.insert(0, vendor)
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
        finally:
            sys.path.remove(vendor)
        options = vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
                    num_faces=8, min_face_detection_confidence=.5, min_face_presence_confidence=.5)
        output = images.clone()
        detected = []
        preview = None
        with vision.FaceLandmarker.create_from_options(options) as detector:
            for index in range(images.shape[0]):
                frame = images[index, :, :, :3].detach().cpu().numpy().astype(np.float32)
                h, w = frame.shape[:2]
                faces = detect_faces(frame, detector, mp)
                mask = np.zeros((h, w), np.float32)
                for landmarks in faces:
                    mask = np.maximum(mask, skin_mask(frame.shape, landmarks, vision.FaceLandmarksConnections))
                cleaned, amount = attenuate(frame, mask, strength)
                output[index, :, :, :3] = torch.from_numpy(cleaned).to(images.device, images.dtype)
                detected.append(len(faces))
                if index == 0:
                    preview = (frame, cleaned, mask)
        import folder_paths
        directory = Path(folder_paths.get_output_directory()).parent.parent / 'outputs/h3-webui/skin'
        directory.mkdir(parents=True, exist_ok=True)
        prefix = Path(filename_prefix).name
        for label, frame in zip(('source', 'cleaned', 'mask'), preview):
            pixels = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(directory / (prefix + '_' + label + '.png')), pixels[:, :, ::-1] if pixels.ndim == 3 else pixels)
        (directory / (prefix + '.json')).write_text(json.dumps({'frames': len(detected), 'faces_per_frame': detected,
            'strength': strength, 'model': str(MODEL), 'detector_precision': 'MediaPipe float16 model asset / CPU runtime',
            'changed_pixels_first_frame': int(np.count_nonzero(np.any(preview[0] != preview[1], axis=-1))),
            'excluded': ['eyes', 'eyebrows', 'lips', 'nostrils'], 'method': 'subtract low-frequency bright neutral excess; no generation'}, indent=2))
        return (output,)
