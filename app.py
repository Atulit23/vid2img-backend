from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
import tempfile
import io
from pathlib import Path
from werkzeug.utils import secure_filename
import uuid
from flask_cors import CORS
app = Flask(__name__)
CORS(app) 

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clip_video_to_duration(input_path, output_path, max_duration=10):
    """Clip video to maximum duration in seconds"""
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    if duration <= max_duration:
        cap.release()
        return input_path
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    max_frames = int(fps * max_duration)
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    return output_path

def extract_and_composite_frames(video_path, blend_mode='lighten', alpha=0.1, frame_skip=1):
    """Extract and composite frames from video, returning the composite image"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {total_frames} frames, {fps} FPS, {width}x{height}")
    print(f"Processing every {frame_skip} frame(s) with {blend_mode} blend mode")
    
    ret, base_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame from video")
    
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
        frame_float = frame.astype(np.float64)
        
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
    
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    print(f"Total frames processed: {processed_count}")
    
    return composite

def create_motion_trails(video_path, decay_factor=0.95):
    """Create motion trails from video, returning the result image"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    
    height, width = first_frame.shape[:2]
    trail_buffer = np.zeros((height, width, 3), dtype=np.float64)
    trail_buffer[:] = first_frame.astype(np.float64)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_float = frame.astype(np.float64)
        
        trail_buffer *= decay_factor
        trail_buffer = np.maximum(trail_buffer, frame_float)
        
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    
    result = np.clip(trail_buffer, 0, 255).astype(np.uint8)
    print(f"Motion trails created from {frame_count} frames")
    
    return result

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Composite Generator</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: 0 auto; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select { width: 100%; padding: 8px; margin-bottom: 10px; }
            button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            button:hover { background-color: #45a049; }
            .info { background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Video Composite Generator</h1>
            <div class="info">
                <p><strong>Info:</strong> Upload a video file to generate a composite image. Videos longer than 10 seconds will be automatically clipped.</p>
                <p><strong>Supported formats:</strong> MP4, AVI, MOV, MKV, FLV, WMV, WEBM</p>
            </div>
            
            <form action="/process" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="video">Select Video File:</label>
                    <input type="file" name="video" accept="video/*" required>
                </div>
                
                <div class="form-group">
                    <label for="mode">Blend Mode:</label>
                    <select name="mode">
                        <option value="lighten">Lighten (Default)</option>
                        <option value="screen">Screen</option>
                        <option value="overlay">Overlay</option>
                        <option value="add">Add</option>
                        <option value="difference">Difference</option>
                        <option value="trails">Motion Trails</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="alpha">Alpha (0.0-1.0, for overlay/add modes):</label>
                    <input type="number" name="alpha" min="0" max="1" step="0.1" value="0.1">
                </div>
                
                <div class="form-group">
                    <label for="frame_skip">Frame Skip (process every Nth frame):</label>
                    <input type="number" name="frame_skip" min="1" value="1">
                </div>
                
                <div class="form-group">
                    <label for="decay">Decay Factor (for trails mode, 0.0-1.0):</label>
                    <input type="number" name="decay" min="0" max="1" step="0.01" value="0.95">
                </div>
                
                <button type="submit">Generate Composite</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/process', methods=['POST'])
def process_video():
    # Check if video file is present
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    # Get parameters
    blend_mode = request.form.get('mode', 'lighten')
    alpha = float(request.form.get('alpha', 0.1))
    frame_skip = int(request.form.get('frame_skip', 1))
    decay = float(request.form.get('decay', 0.95))
    
    # Validate parameters
    alpha = max(0.0, min(1.0, alpha))
    frame_skip = max(1, frame_skip)
    decay = max(0.0, min(1.0, decay))
    
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded video
        original_filename = secure_filename(file.filename)
        video_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}_{original_filename}")
        file.save(video_path)
        
        # Clip video if longer than 10 seconds
        clipped_video_path = os.path.join(temp_dir, f"clipped_{uuid.uuid4().hex}.mp4")
        processed_video_path = clip_video_to_duration(video_path, clipped_video_path, max_duration=10)
        
        # Process video based on mode
        if blend_mode == 'trails':
            composite_image = create_motion_trails(processed_video_path, decay_factor=decay)
        else:
            composite_image = extract_and_composite_frames(
                processed_video_path, blend_mode=blend_mode, alpha=alpha, frame_skip=frame_skip
            )
        
        # Convert image to bytes for response
        success, img_encoded = cv2.imencode('.jpg', composite_image)
        if not success:
            raise ValueError("Failed to encode image")
        
        img_bytes = img_encoded.tobytes()
        
        # Create response
        return send_file(
            io.BytesIO(img_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'composite_{blend_mode}_{uuid.uuid4().hex[:8]}.jpg'
        )
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
    finally:
        # Clean up temporary files
        try:
            for file_path in [video_path, clipped_video_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            os.rmdir(temp_dir)
        except:
            pass  # Ignore cleanup errors

@app.route('/api/process', methods=['POST'])
def api_process_video():
    """API endpoint that returns JSON response with base64 encoded image"""
    import base64
    
    # Check if video file is present
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    # Get parameters from JSON or form data
    if request.is_json:
        params = request.get_json()
    else:
        params = request.form.to_dict()
    
    blend_mode = params.get('mode', 'lighten')
    alpha = float(params.get('alpha', 0.1))
    frame_skip = int(params.get('frame_skip', 1))
    decay = float(params.get('decay', 0.95))
    
    # Validate parameters
    alpha = max(0.0, min(1.0, alpha))
    frame_skip = max(1, frame_skip)
    decay = max(0.0, min(1.0, decay))
    
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded video
        original_filename = secure_filename(file.filename)
        video_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}_{original_filename}")
        file.save(video_path)
        
        # Clip video if longer than 10 seconds
        clipped_video_path = os.path.join(temp_dir, f"clipped_{uuid.uuid4().hex}.mp4")
        processed_video_path = clip_video_to_duration(video_path, clipped_video_path, max_duration=10)
        
        # Process video based on mode
        if blend_mode == 'trails':
            composite_image = create_motion_trails(processed_video_path, decay_factor=decay)
        else:
            composite_image = extract_and_composite_frames(
                processed_video_path, blend_mode=blend_mode, alpha=alpha, frame_skip=frame_skip
            )
        
        # Convert image to base64
        success, img_encoded = cv2.imencode('.jpg', composite_image)
        if not success:
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
                'decay': decay
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
    finally:
        # Clean up temporary files
        try:
            for file_path in [video_path, clipped_video_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            os.rmdir(temp_dir)
        except:
            pass  # Ignore cleanup errors

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# Requirements:
"""
To run this application, install the required packages:

pip install flask opencv-python numpy

Usage:
1. Save this as app.py
2. Run: python app.py
3. Open http://localhost:5000 in your browser
4. Upload a video file and select processing options
5. Download the generated composite image

API Usage:
POST to /api/process with form-data:
- video: video file
- mode: blend mode (optional, default: lighten)
- alpha: alpha value (optional, default: 0.1)
- frame_skip: frame skip (optional, default: 1)  
- decay: decay factor for trails (optional, default: 0.95)

Returns JSON with base64 encoded image.
"""