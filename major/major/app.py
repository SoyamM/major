from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import face_recognition
import base64
import numpy as np
import cv2
import os
import json
from datetime import datetime, timedelta
import pytz
import time

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'videos'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('known_admins', exist_ok=True)

# Load known admin faces
KNOWN_FACES = []
KNOWN_NAMES = []

def load_admins():
    global KNOWN_FACES, KNOWN_NAMES
    admin_dir = 'known_admins'
    if not os.path.exists(admin_dir):
        os.makedirs(admin_dir)
        return
    KNOWN_FACES.clear()
    KNOWN_NAMES.clear()
    for file in os.listdir(admin_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(admin_dir, file)
            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    KNOWN_FACES.append(encodings[0])
                    KNOWN_NAMES.append(os.path.splitext(file)[0])
            except Exception as e:
                print(f"Error loading {file}: {e}")

# Call once at startup
load_admins()

# Meetings storage
MEETINGS_FILE = 'meetings.json'
if not os.path.exists(MEETINGS_FILE):
    with open(MEETINGS_FILE, 'w') as f:
        json.dump({}, f)

def get_meetings():
    try:
        with open(MEETINGS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_meetings(data):
    with open(MEETINGS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def generate_slots(date_key):
    meetings = get_meetings()
    if date_key not in meetings:
        ist = pytz.timezone('Asia/Kolkata')
        start = ist.localize(datetime.strptime(f"{date_key} 09:00", "%Y-%m-%d %H:%M"))
        slots = []
        for i in range(17):
            slot_time = (start + timedelta(minutes=30*i)).strftime("%I:%M %p")
            slots.append({"time": slot_time, "booked": False, "guest": None})
        meetings[date_key] = slots
        save_meetings(meetings)
    return meetings[date_key]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.json
        img_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"is_admin": False, "name": "Guest"}), 400
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        result = {"is_admin": False, "name": "Guest"}
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(KNOWN_FACES, encoding, tolerance=0.6)
            if True in matches:
                idx = matches.index(True)
                result = {"is_admin": True, "name": KNOWN_NAMES[idx]}
                break
        return jsonify(result)
    except Exception as e:
        print("Recognition error:", e)
        return jsonify({"is_admin": False, "name": "Guest"}), 500

@app.route('/admin')
def admin():
    name = request.args.get('name', 'Admin')
    ist = pytz.timezone('Asia/Kolkata')
    today = datetime.now(ist).strftime("%Y-%m-%d")
    tomorrow = (datetime.now(ist) + timedelta(days=1)).strftime("%Y-%m-%d")
    slots_today = generate_slots(today)
    slots_tomorrow = generate_slots(tomorrow)
    try:
        videos = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith('.mp4')]
    except:
        videos = []
    return render_template('admin.html', 
                         name=name, 
                         slots_today=slots_today, 
                         slots_tomorrow=slots_tomorrow, 
                         videos=sorted(videos), 
                         today_date=today)

@app.route('/guest')
def guest():
    return render_template('guest.html')

@app.route('/schedule', methods=['POST'])
def schedule():
    try:
        data = request.json
        guest_name = data.get('name', 'Guest').strip()
        if not guest_name:
            guest_name = 'Guest'
        ist = pytz.timezone('Asia/Kolkata')
        today = datetime.now(ist).strftime("%Y-%m-%d")
        tomorrow = (datetime.now(ist) + timedelta(days=1)).strftime("%Y-%m-%d")
        meetings = get_meetings()
        slots = generate_slots(today)
        available = next((i for i, s in enumerate(slots) if not s['booked']), None)
        if available is not None:
            slots[available]['booked'] = True
            slots[available]['guest'] = guest_name
            meetings[today] = slots
            save_meetings(meetings)
            return jsonify({"success": True, "time": slots[available]['time'], "date": today})
        else:
            return jsonify({"success": False, "tomorrow": tomorrow})
    except Exception as e:
        print("Schedule error:", e)
        return jsonify({"success": False}), 500

@app.route('/schedule_tomorrow', methods=['POST'])
def schedule_tomorrow():
    try:
        data = request.json
        tomorrow = data['date']
        guest_name = data.get('name', 'Guest').strip() or 'Guest'
        meetings = get_meetings()
        slots = generate_slots(tomorrow)
        available = next((i for i, s in enumerate(slots) if not s['booked']), None)
        if available is not None:
            slots[available]['booked'] = True
            slots[available]['guest'] = guest_name
            meetings[tomorrow] = slots
            save_meetings(meetings)
            return jsonify({"success": True, "time": slots[available]['time'], "date": tomorrow})
        return jsonify({"success": False})
    except Exception as e:
        print("Schedule tomorrow error:", e)
        return jsonify({"success": False}), 500

@app.route('/record_video', methods=['POST'])
def record_video():
    try:
        if 'video' not in request.files:
            return jsonify({"success": False, "error": "No video"}), 400
        file = request.files['video']
        guest_name = request.form.get('name', 'Guest').strip() or 'Guest'
        if file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"}), 400
        
        ist = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(ist).strftime('%Y%m%d_%H%M%S')
        safe_name = "".join(c for c in guest_name if c.isalnum() or c in " _-")[:20]
        filename = f"{safe_name}_{timestamp}.mp4"
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        return jsonify({"success": True, "filename": filename})
    except Exception as e:
        print("Record video error:", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/delete_video', methods=['POST'])
def delete_video():
    try:
        data = request.json
        filename = data.get('filename')
        if not filename:
            return jsonify(success=False), 400
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(path):
            os.remove(path)
            return jsonify(success=True)
        return jsonify(success=False), 404
    except Exception as e:
        print("Delete error:", e)
        return jsonify(success=False), 500

@app.route('/cancel_meeting', methods=['POST'])
def cancel_meeting():
    try:
        data = request.json
        date = data['date']
        index = int(data['index'])
        meetings = get_meetings()
        if date in meetings and 0 <= index < len(meetings[date]):
            meetings[date][index]['booked'] = False
            meetings[date][index]['guest'] = None
            save_meetings(meetings)
            return jsonify(success=True)
        return jsonify(success=False), 400
    except Exception as e:
        print("Cancel error:", e)
        return jsonify(success=False), 500

# FINAL VIDEO SERVING WITH RANGE SUPPORT
@app.route('/videos/<path:filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(video_path):
        return "Video not found", 404

    range_header = request.headers.get('Range', None)
    if not range_header:
        response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'no-cache'
        return response

    size = os.stat(video_path).st_size
    byte1, byte2 = 0, None
    m = range_header.replace('bytes=', '').split('-')
    byte1 = int(m[0])
    byte2 = int(m[1]) if m[1] else size - 1
    if byte2 >= size:
        byte2 = size - 1
    length = byte2 - byte1 + 1

    with open(video_path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data, 206, mimetype='video/mp4', direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte2}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    rv.headers.add('Cache-Control', 'no-cache')
    return rv

# Auto-reload known faces every 10 seconds
last_reload = time.time()
def reload_admins_if_needed():
    global last_reload
    if time.time() - last_reload > 10:
        load_admins()
        last_reload = time.time()

@app.before_request
def before_request():
    """
    Called before each request to reload known admin faces if needed.
    """
    reload_admins_if_needed()

if __name__ == '__main__':
    print("Server starting...")
    print("   Guest: http://localhost:5000/guest")
    print("   Admin: http://localhost:5000/admin")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)