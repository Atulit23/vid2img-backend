from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import cv2
import numpy as np
import os
import tempfile
import io
import uuid
from pathlib import Path

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _resize_dims(w: int, h: int, target_h: int = 480):
    """
    Return (new_w, new_h) keeping aspect ratio and making even sizes.
    If target_h is None or the video is already <= target_h, return original (evened).
    """
    if target_h is None or h <= target_h:
        return (w - w % 2, h - h % 2)
    scale = target_h / float(h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # even dimensions are safer for codecs and some ops
    new_w -= new_w % 2
    new_h -= new_h % 2
    return new_w, new_h


def _resize_frame(frame: np.ndarray, target_h: int = 480) -> np.ndarray:
    if frame is None:
        return None
    h, w = frame.shape[:2]
    new_w, new_h = _resize_dims(w, h, target_h)
    if (new_w, new_h) == (w, h):
        return frame
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def clip_video_to_duration(input_path: str, output_path: str, max_duration: int = 10, target_h: int = 480) -> str:
    """
    Clip video to max_duration seconds AND downscale frames to target_h (SD) while writing.
    Returns the path to the written (clipped, resized) video.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0  # fallback

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read first frame from video")

    first_resized = _resize_frame(first, target_h)
    h, w = first_resized.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    max_frames = int(min(total_frames, fps * max_duration)) if max_duration else total_frames
    written = 0

    # write first frame
    out.write(first_resized)
    written += 1

    while written < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(_resize_frame(frame, target_h))
        written += 1

    cap.release()
    out.release()
    return output_path


def extract_and_composite_frames(video_path: str, blend_mode: str = 'lighten', alpha: float = 0.1,
                                 frame_skip: int = 1, target_h: int = 480) -> np.ndarray:
    """
    Extract frames (downscaled to target_h) and composite them with the chosen blend mode.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Video info: {total_frames} frames, {fps:.2f} FPS")
    print(f"Processing every {frame_skip} frame(s) with {blend_mode} blend mode at ~{target_h}p")

    ret, base_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read first frame from video")

    base_frame = _resize_frame(base_frame, target_h)
    composite = base_frame.astype(np.float64)

    frame_count = 1
    processed_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        processed_count += 1
        frame_float = _resize_frame(frame, target_h).astype(np.float64)

        if blend_mode == 'lighten':
            composite = np.maximum(composite, frame_float)
        elif blend_mode == 'screen':
            composite = 255 - (255 - composite) * (255 - frame_float) / 255
        elif blend_mode == 'overlay':
            composite = (1 - alpha) * composite + alpha * frame_float
        elif blend_mode == 'add':
            composite = np.clip(composite + frame_float * alpha, 0, 255)
        elif blend_mode == 'difference':
            diff = np.abs(frame_float - base_frame.astype(np.float64))
            composite = np.maximum(composite, diff)

    cap.release()
    print(f"Total frames processed: {processed_count}")

    return np.clip(composite, 0, 255).astype(np.uint8)


def create_motion_trails(video_path: str, decay_factor: float = 0.95, target_h: int = 480) -> np.ndarray:
    """
    Create motion trails by decaying the buffer and taking max with the current frame.
    Frames are downscaled to target_h.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read first frame")

    first_frame = _resize_frame(first_frame, target_h)
    trail_buffer = first_frame.astype(np.float64)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame_float = _resize_frame(frame, target_h).astype(np.float64)
        trail_buffer *= decay_factor
        trail_buffer = np.maximum(trail_buffer, frame_float)

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    print(f"Motion trails created from {frame_count} frames")
    return np.clip(trail_buffer, 0, 255).astype(np.uint8)


# ---------- Routes ----------
@app.get("/")
def health():
    return "ok", 200


@app.route('/process', methods=['POST'])
def process_video():
    # Check file
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400

    # Params (form-data)
    blend_mode = request.form.get('mode', 'lighten')
    alpha = float(request.form.get('alpha', 0.1))
    frame_skip = int(request.form.get('frame_skip', 1))
    decay = float(request.form.get('decay', 0.95))
    target_h = int(request.form.get('target_h', 480))  # SD default

    # Clamp
    alpha = max(0.0, min(1.0, alpha))
    frame_skip = max(1, frame_skip)
    decay = max(0.0, min(1.0, decay))

    temp_dir = tempfile.mkdtemp()

    try:
        original_filename = secure_filename(file.filename)
        in_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}_{original_filename}")
        file.save(in_path)

        clipped_path = os.path.join(temp_dir, f"clipped_{uuid.uuid4().hex}.mp4")
        processed_path = clip_video_to_duration(in_path, clipped_path, max_duration=10, target_h=target_h)

        if blend_mode == 'trails':
            composite_image = create_motion_trails(processed_path, decay_factor=decay, target_h=target_h)
        else:
            composite_image = extract_and_composite_frames(
                processed_path, blend_mode=blend_mode, alpha=alpha, frame_skip=frame_skip, target_h=target_h
            )

        ok, img_encoded = cv2.imencode('.jpg', composite_image)
        if not ok:
            raise ValueError("Failed to encode image")

        img_bytes = img_encoded.tobytes()
        return send_file(
            io.BytesIO(img_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'composite_{blend_mode}_{uuid.uuid4().hex[:8]}.jpg'
        )

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    finally:
        try:
            for p in [in_path, clipped_path]:
                if p and os.path.exists(p):
                    os.remove(p)
            os.rmdir(temp_dir)
        except Exception:
            pass


@app.route('/api/process', methods=['POST'])
def api_process_video():
    """API endpoint that returns JSON with base64-encoded JPEG."""
    import base64

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400

    # Params (JSON or form-data)
    params = request.get_json(silent=True) or request.form.to_dict()

    blend_mode = params.get('mode', 'lighten')
    alpha = float(params.get('alpha', 0.1))
    frame_skip = int(params.get('frame_skip', 1))
    decay = float(params.get('decay', 0.95))
    target_h = int(params.get('target_h', 480))  # SD default

    # Clamp
    alpha = max(0.0, min(1.0, alpha))
    frame_skip = max(1, frame_skip)
    decay = max(0.0, min(1.0, decay))

    temp_dir = tempfile.mkdtemp()

    try:
        original_filename = secure_filename(file.filename)
        in_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}_{original_filename}")
        file.save(in_path)

        clipped_path = os.path.join(temp_dir, f"clipped_{uuid.uuid4().hex}.mp4")
        processed_path = clip_video_to_duration(in_path, clipped_path, max_duration=10, target_h=target_h)

        if blend_mode == 'trails':
            composite_image = create_motion_trails(processed_path, decay_factor=decay, target_h=target_h)
        else:
            composite_image = extract_and_composite_frames(
                processed_path, blend_mode=blend_mode, alpha=alpha, frame_skip=frame_skip, target_h=target_h
            )

        ok, img_encoded = cv2.imencode('.jpg', composite_image)
        if not ok:
            raise ValueError("Failed to encode image")

        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        return jsonify({
            'success': True,
            'image': img_base64,
            'format': 'jpeg',
            'blend_mode': blend_mode,
            'parameters': {
                'alpha': alpha,
                'frame_skip': frame_skip,
                'decay': decay,
                'target_h': target_h
            }
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    finally:
        try:
            for p in [in_path, clipped_path]:
                if p and os.path.exists(p):
                    os.remove(p)
            os.rmdir(temp_dir)
        except Exception:
            pass


if __name__ == '__main__':
    # For local testing. On Render, use Gunicorn:
    # gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 300 --graceful-timeout 300 --keep-alive 5 --worker-tmp-dir /dev/shm
    app.run(debug=True, host='0.0.0.0', port=8000)
