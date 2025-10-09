# --- HƯỚNG DẪN CHẠY BACKEND ---
# 1. Cài đặt Flask: pip install Flask
# 2. Lưu mã này thành file app.py
# 3. Chạy server: python app.py
# 4. Sau khi server chạy, mở file index.html trong trình duyệt.

import time
import os
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS # Cần thiết để cho phép frontend (index.html) truy cập API
from werkzeug.utils import secure_filename
import uuid
import mimetypes
from ultralytics import YOLO
import cv2
import numpy as np
import os


app = Flask(__name__)
# Kích hoạt CORS. Điều này cho phép index.html (chạy từ file local) 
# gửi yêu cầu đến server Flask (chạy ở 127.0.0.1:5000).
CORS(app) 


@app.route('/')
def index():
    """Serve the demo/index.html page."""
    # app.root_path is the directory containing this file (demo/)
    return send_from_directory(app.root_path, 'index.html')


@app.route('/<path:filename>')
def serve_file(filename):
    """Serve other files from the demo directory (CSS/JS/assets) if requested."""
    return send_from_directory(app.root_path, filename)

@app.route('/api/greet', methods=['POST'])
def greet_user():
    """
    Xử lý yêu cầu POST tại endpoint /api/greet.
    Nhận tên từ JSON body và trả về lời chào.
    """
    try:
        # Lấy dữ liệu JSON từ yêu cầu
        data = request.get_json()
        
        # Kiểm tra xem có trường 'name' trong dữ liệu không
        if data and 'name' in data:
            user_name = data['name']
            
            # Logic xử lý: Tạo lời chào
            greeting_message = f"Xin chào {user_name}! Bạn đã kết nối thành công với Python Backend (Flask)."
            
            # Trả về phản hồi JSON
            return jsonify({
                "status": "success",
                "message": greeting_message,
                "timestamp": time.time()
            }), 200 # Mã trạng thái HTTP 200 (OK)
        else:
            # Xử lý trường hợp không tìm thấy 'name'
            return jsonify({
                "status": "error",
                "message": "Không tìm thấy trường 'name' trong dữ liệu yêu cầu."
            }), 400 # Mã trạng thái HTTP 400 (Bad Request)

    except Exception as e:
        # Xử lý các lỗi ngoại lệ khác
        print(f"Lỗi xảy ra trong quá trình xử lý yêu cầu: {e}")
        return jsonify({
            "status": "error",
            "message": f"Server gặp lỗi nội bộ: {str(e)}"
        }), 500 # Mã trạng thái HTTP 500 (Internal Server Error)


@app.route('/api/hello', methods=['GET'])
def api_hello():
    """Simple HelloWorld API for the demo page."""
    return jsonify({
        'status': 'ok',
        'message': 'Hello, World! Đây là API demo từ Flask.'
    })


# ----- Media processing endpoint -----
# Load model once (yolo11n.pt expected in repository 'models' folder)
BASE_DIR = os.path.abspath(os.path.join(app.root_path, '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolo11n.pt')
device_str = 'cpu'
try:
    # try to load model on GPU if available
    # ultralytics YOLO will auto-detect cuda when device='cuda' and available
    # we'll attempt to use cuda; fallback to cpu
    use_cuda = False
    try:
        import torch
        use_cuda = torch.cuda.is_available() if device_str == 'cuda' else False
    except Exception:
        use_cuda = False
    device_str = 'cuda' if use_cuda else 'cpu'
    print(f'Loading YOLO model on device: {device_str}')
    yolo_model = YOLO(MODEL_PATH)
    # set the model device if API supports it
    try:
        yolo_model.to(device_str)
    except Exception:
        # some ultralytics versions manage device via predict args
        pass
except Exception as e:
    print(f"Warning: cannot load model at {MODEL_PATH}: {e}")
    yolo_model = None


def ensure_outputs_dir():
    out_dir = os.path.join(app.root_path, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def predict_on_frame(frame, device: str = 'cpu'):
    """Try different ways to run inference on a single frame and return boxes list.
    device: 'cpu' or 'cuda' string preferred
    """
    if yolo_model is None:
        raise RuntimeError('YOLO model not loaded; ensure models/yolo11n.pt exists and ultralytics is installed')
    # prefer to pass device into predict, but fallback to direct call
    try:
        res = yolo_model.predict(source=frame, conf=0.25, device=device, verbose=False)
        if res and len(res) > 0:
            return res[0].boxes
    except Exception:
        pass
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = yolo_model(rgb)
        if res and len(res) > 0:
            return res[0].boxes
    except Exception:
        pass
    try:
        res = yolo_model.predict(source=[frame], conf=0.25, device=device, verbose=False)
        if res and len(res) > 0:
            return res[0].boxes
    except Exception:
        pass
    return []


# active_streams holds cancellation flags for running streams (stream_id -> dict)
active_streams = {}


def gen_webcam_stream(rows=4, cols=6):
    """Generator that captures webcam frames, runs detection and yields multipart JPEG stream."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam cannot be opened")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cell_w = width / cols
    cell_h = height / rows
    heat_w = int(width * 0.5)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                boxes = predict_on_frame(frame, device=device_str)
            except Exception as e:
                print('Webcam prediction error:', e)
                boxes = []

            grid_counts = np.zeros((rows, cols), dtype=int)
            centers = []
            for box in boxes:
                box_coords = box.xyxy[0].cpu().numpy()
                x_center = (box_coords[0] + box_coords[2]) / 2
                y_center = (box_coords[1] + box_coords[3]) / 2
                col = int(x_center // cell_w)
                row = int(y_center // cell_h)
                col = min(max(col, 0), cols - 1)
                row = min(max(row, 0), rows - 1)
                grid_counts[row, col] += 1
                centers.append((x_center, y_center))

            # draw overlay
            disp = frame.copy()
            for r in range(1, rows):
                y = int(r * cell_h)
                cv2.line(disp, (0, y), (width, y), (0, 255, 255), 1, lineType=cv2.LINE_AA)
            for c in range(1, cols):
                x = int(c * cell_w)
                cv2.line(disp, (x, 0), (x, height), (0, 255, 255), 1, lineType=cv2.LINE_AA)
            for (cx, cy) in centers:
                cv2.circle(disp, (int(cx), int(cy)), 4, (255, 0, 0), -1)

            heat_img = make_heatmap_image(grid_counts, heat_w, height)
            disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            combined = np.concatenate([disp_rgb, heat_img], axis=1)
            ret2, jpg = cv2.imencode('.jpg', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            if not ret2:
                continue
            frame_bytes = jpg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    data = request.get_json() or {}
    sid = data.get('stream_id')
    if not sid:
        return jsonify({'status': 'error', 'message': 'stream_id required'}), 400
    entry = active_streams.get(sid)
    if not entry:
        return jsonify({'status': 'error', 'message': 'stream_id not found'}), 404
    entry['stop'] = True
    return jsonify({'status': 'ok', 'stopped': sid})


def gen_video_stream(file_name: str, rows: int = 4, cols: int = 6, stream_id: str = None):
    """Stream a saved video file frame-by-frame with detections and heatmap as MJPEG."""
    out_dir = ensure_outputs_dir()
    video_path = os.path.join(out_dir, file_name)
    if not os.path.exists(video_path):
        print('Requested video for streaming not found:', video_path)
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Failed to open video for streaming:', video_path)
        return

    try:
        while True:
            # allow external stop request
            if stream_id and stream_id in active_streams and active_streams[stream_id].get('stop'):
                print(f'Stream {stream_id} received stop signal')
                break
            ret, frame = cap.read()
            if not ret:
                break
            try:
                boxes = predict_on_frame(frame, device=device_str)
            except Exception as e:
                print('Video prediction error:', e)
                boxes = []

            h, w = frame.shape[:2]
            cell_w = w / cols
            cell_h = h / rows

            grid_counts = np.zeros((rows, cols), dtype=int)
            centers = []
            for box in boxes:
                try:
                    box_coords = box.xyxy[0].cpu().numpy()
                    x_center = (box_coords[0] + box_coords[2]) / 2
                    y_center = (box_coords[1] + box_coords[3]) / 2
                except Exception:
                    # fallback if box format differs
                    vals = np.array(box).flatten()
                    x_center = (vals[0] + vals[2]) / 2
                    y_center = (vals[1] + vals[3]) / 2
                col = int(x_center // cell_w)
                row = int(y_center // cell_h)
                col = min(max(col, 0), cols - 1)
                row = min(max(row, 0), rows - 1)
                grid_counts[row, col] += 1
                centers.append((x_center, y_center))

            # draw overlay
            disp = frame.copy()
            for r in range(1, rows):
                y = int(r * cell_h)
                cv2.line(disp, (0, y), (w, y), (0, 255, 255), 1, lineType=cv2.LINE_AA)
            for c in range(1, cols):
                x = int(c * cell_w)
                cv2.line(disp, (x, 0), (x, h), (0, 255, 255), 1, lineType=cv2.LINE_AA)
            for (cx, cy) in centers:
                cv2.circle(disp, (int(cx), int(cy)), 4, (255, 0, 0), -1)

            heat_w = int(w * 0.5)
            heat_img = make_heatmap_image(grid_counts, heat_w, h)
            disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            combined = np.concatenate([disp_rgb, heat_img], axis=1)
            # encode to jpeg
            ret2, jpg = cv2.imencode('.jpg', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            if not ret2:
                print('Failed to encode frame to jpg for streaming')
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
    finally:
        cap.release()


@app.route('/stream_video')
def stream_video():
    file = request.args.get('file')
    if not file:
        return 'file param required', 400
    try:
        rows = int(request.args.get('rows', 4))
        cols = int(request.args.get('cols', 6))
    except Exception:
        rows, cols = 4, 6
    sid = request.args.get('sid')
    return Response(gen_video_stream(file, rows=rows, cols=cols, stream_id=sid), mimetype='multipart/x-mixed-replace; boundary=frame')


def make_heatmap_image(counts, out_w, out_h, colormap=cv2.COLORMAP_MAGMA):
    """Create a color heatmap image (BGR) from a small integer grid using OpenCV.
    counts: 2D numpy array (rows x cols)
    returns BGR image of size (out_h, out_w, 3)
    """
    rows, cols = counts.shape
    maxv = counts.max() if counts.max() > 0 else 1
    norm = (counts.astype(np.float32) / maxv * 255.0).astype(np.uint8)
    # resize the indexed map to target size
    resized = cv2.resize(norm, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap(resized, colormap)
    # overlay integer text per cell
    cell_w = out_w / cols
    cell_h = out_h / rows
    for r in range(rows):
        for c in range(cols):
            val = int(counts[r, c])
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.55) * cell_h)
            cv2.putText(heat, str(val), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return heat


def process_image(file_path, rows=4, cols=6):
    """Run detection on an image, create grid overlay + heatmap, save combined PNG and return path."""
    img = cv2.imread(file_path)
    if img is None:
        raise RuntimeError('Cannot read image')
    h, w = img.shape[:2]
    cell_w = w / cols
    cell_h = h / rows

    if yolo_model is None:
        raise RuntimeError('YOLO model not loaded; ensure models/yolo11n.pt exists and ultralytics is installed')
    try:
        results = yolo_model.predict(source=file_path, conf=0.25, device='cpu')
        boxes = results[0].boxes if results and len(results) > 0 else []
    except Exception as e:
        print(f"Error during model.predict on image: {e}")
        boxes = []

    grid_counts = np.zeros((rows, cols), dtype=int)
    centers = []
    for box in boxes:
        box_coords = box.xyxy[0].cpu().numpy()
        x_center = (box_coords[0] + box_coords[2]) / 2
        y_center = (box_coords[1] + box_coords[3]) / 2
        col = int(x_center // cell_w)
        row = int(y_center // cell_h)
        col = min(max(col, 0), cols - 1)
        row = min(max(row, 0), rows - 1)
        grid_counts[row, col] += 1
        centers.append((x_center, y_center))

    # Draw overlay with OpenCV
    disp = img.copy()
    # draw grid
    for r in range(1, rows):
        y = int(r * cell_h)
        cv2.line(disp, (0, y), (w, y), (0, 255, 255), 1, lineType=cv2.LINE_AA)
    for c in range(1, cols):
        x = int(c * cell_w)
        cv2.line(disp, (x, 0), (x, h), (0, 255, 255), 1, lineType=cv2.LINE_AA)
    # draw centers
    for idx, (cx, cy) in enumerate(centers):
        cv2.circle(disp, (int(cx), int(cy)), 4, (255, 0, 0), -1)

    # Create heatmap via OpenCV and combine with original image
    heat_w = int(w * 0.5)
    heat_h = h
    heat_img = make_heatmap_image(grid_counts, heat_w, heat_h)
    disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    combined = np.concatenate([disp_rgb, heat_img], axis=1)

    out_dir = ensure_outputs_dir()
    out_name = f"image_result_{uuid.uuid4().hex}.png"
    out_path = os.path.join(out_dir, out_name)
    # save as PNG (RGB -> BGR for cv2)
    cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    return out_name


def process_video(file_path, rows=4, cols=6):
    """Process video frames, generate side-by-side output video with per-frame heatmap; return filename."""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cell_w = width / cols
    cell_h = height / rows

    out_dir = ensure_outputs_dir()
    out_name = f"video_result_{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(out_dir, out_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # determine output frame size by rendering matplotlib figure to image; to simplify, make combined width = width + heatmap_width (~width//3)
    heat_w = int(width * 0.5)
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width + heat_w, height))
    if not out_writer.isOpened():
        raise RuntimeError(f'Cannot open VideoWriter for path {out_path} with size {(width+heat_w, height)}')

    recent = []
    try:
        frame_idx = 0
        pass


        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            try:
                boxes = predict_on_frame(frame)
            except Exception as e:
                import traceback
                print(f"Frame {frame_idx}: unexpected prediction helper error: {e}")
                traceback.print_exc()
                boxes = []

            grid_counts = np.zeros((rows, cols), dtype=int)
            centers = []
            for box in boxes:
                box_coords = box.xyxy[0].cpu().numpy()
                x_center = (box_coords[0] + box_coords[2]) / 2
                y_center = (box_coords[1] + box_coords[3]) / 2
                col = int(x_center // cell_w)
                row = int(y_center // cell_h)
                col = min(max(col, 0), cols - 1)
                row = min(max(row, 0), rows - 1)
                grid_counts[row, col] += 1
                centers.append((x_center, y_center))

            # draw overlay
            disp = frame.copy()
            for r in range(1, rows):
                y = int(r * cell_h)
                cv2.line(disp, (0, y), (width, y), (0, 255, 255), 1, lineType=cv2.LINE_AA)
            for c in range(1, cols):
                x = int(c * cell_w)
                cv2.line(disp, (x, 0), (x, height), (0, 255, 255), 1, lineType=cv2.LINE_AA)
            for (cx, cy) in centers:
                cv2.circle(disp, (int(cx), int(cy)), 4, (255, 0, 0), -1)

            # create heatmap image via OpenCV then combine
            heat_img = make_heatmap_image(grid_counts, heat_w, height)
            disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            combined = np.concatenate([disp_rgb, heat_img], axis=1)
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            try:
                out_writer.write(combined_bgr)
            except Exception as e:
                import traceback
                print(f"Frame {frame_idx}: error writing frame to output: {e}")
                traceback.print_exc()
                # attempt to continue

        out_writer.release()
        cap.release()
        return out_name
    finally:
        if cap.isOpened():
            cap.release()
        if out_writer is not None:
            out_writer.release()



@app.route('/api/process_media', methods=['POST'])
def api_process_media():
    """Accept uploaded image or video, process with YOLO and return URL to result."""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    rows = int(request.form.get('rows', 4))
    cols = int(request.form.get('cols', 6))

    filename = secure_filename(file.filename)
    out_dir = ensure_outputs_dir()
    input_name = f"upload_{uuid.uuid4().hex}_{filename}"
    input_path = os.path.join(out_dir, input_name)
    file.save(input_path)

    # detect type by mimetype or extension
    mime, _ = mimetypes.guess_type(input_path)
    try:
        if mime and mime.startswith('image') or filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            out_name = process_image(input_path, rows=rows, cols=cols)
            return jsonify({'status': 'ok', 'type': 'image', 'result': f'/outputs/{out_name}'})
        else:
            # treat as video: save uploaded file and return uploaded path and a stream endpoint
            sid = uuid.uuid4().hex
            active_streams[sid] = {'stop': False, 'file': input_name}
            stream_url = f"/stream_video?file={input_name}&rows={rows}&cols={cols}&sid={sid}"
            return jsonify({'status': 'ok', 'type': 'video', 'uploaded': f'/outputs/{input_name}', 'stream': stream_url, 'stream_id': sid})
    except Exception as e:
        print('Processing error:', e)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    out_dir = os.path.join(app.root_path, 'outputs')
    return send_from_directory(out_dir, filename)


@app.route('/webcam_stream')
def webcam_stream():
    # query params rows, cols
    try:
        rows = int(request.args.get('rows', 4))
        cols = int(request.args.get('cols', 6))
    except Exception:
        rows, cols = 4, 6
    return Response(gen_webcam_stream(rows=rows, cols=cols), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Chạy ứng dụng Flask trên cổng 5000 (cổng mặc định)
    print("Flask Server đang khởi động...")
    app.run(debug=True)
