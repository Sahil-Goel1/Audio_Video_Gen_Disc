from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os, threading, uuid
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # explicit wildcard on all routes

UPLOAD_DIR = "/DATA/HACKATHON/Real_Videos"
OUTPUT_DIR = "/DATA/HACKATHON/Fake_Videos"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

jobs = {}  # job_id -> {"status": ..., "output": ...}




def run_pipeline(job_id, input_path, output_path):
    try:
        # ── paste your model.py imports & pipe init here, OR import from model.py ──
        from model import pipe, generate_long_video_resume
        import torch
        pipe_gpu = pipe.to("cuda", torch.float16)
        generate_long_video_resume(
            input_path,
            output=output_path,
            loops=1,
            save_dir=f"chunks_{job_id}"
        )
        jobs[job_id]["status"] = "done"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

@app.route("/generate", methods=["POST"])
def generate():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    f = request.files["file"]
    job_id = uuid.uuid4().hex[:8]
    input_path  = os.path.join(UPLOAD_DIR, f"{job_id}_{f.filename}")
    output_path = os.path.join(OUTPUT_DIR, f"clone_{job_id}_SAVINGNOW.mp4")
    
    print("Input:",input_path)
    print("Output:", output_path)

    f.save(input_path)
    jobs[job_id] = {"status": "processing", "output": output_path}

    # Run pipeline in background thread
    t = threading.Thread(target=run_pipeline, args=(job_id, input_path, output_path))
    t.daemon = True
    t.start()

    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify(job)
    
    
from flask import make_response

@app.route("/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    #return send_file(job["output"], as_attachment=True, download_name=f"clone_{job_id}.mp4")
    response = make_response(send_file(job["output"], as_attachment=True, download_name=f"clone_{job_id}.mp4"))
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response
    
    

import shutil

FRAMES_DIR = "/DATA/HACKATHON/Frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

# Yeh function add karo
from frames_extraction import extract_frames

@app.route("/discriminate", methods=["POST"])
def discriminate():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    f = request.files["file"]
    job_id = uuid.uuid4().hex[:8]
    
    # Video save karo
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{f.filename}")
    f.save(input_path)
    
    # Frames extract karo
    output_folder = os.path.join(FRAMES_DIR, f"job_{job_id}")
    
    try:
        extract_frames(input_path, output_folder, step=4)
        
        # Kitne frames bane, count karo
        frame_count = len([
            name for name in os.listdir(output_folder)
            if name.endswith(".jpg")
        ])
        
        return jsonify({
            "job_id": job_id,
            "status": "done",
            "frames_extracted": frame_count,
            "frames_dir": output_folder
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

from feature_extraction1 import extract_video_features

avg_std_real = 12.989171742172147
avg_std_fake = 42.38298429210211

@app.route("/final_discriminate", methods=["POST"])

def final_discriminate():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    f = request.files["file"]
    job_id = uuid.uuid4().hex[:8]

    # Video save karo
    input_path = os.path.join(OUTPUT_DIR, f"{job_id}_{f.filename}")
    f.save(input_path)

    # Frames extract karo
    frames_folder = os.path.join(FRAMES_DIR, f"job_{job_id}")

    try:
        extract_frames(input_path, frames_folder, step=4)

        # Features extract karo
        features = extract_video_features(frames_folder)
        # features = [sharpness_mean, sharpness_std, grad_mean, grad_std, temporal_var]

        # sharpness_std == features[1] — yahi single value use karein
        stats = features[1]

        # Real vs Fake check
        if np.abs(stats - avg_std_real) < np.abs(stats - avg_std_fake):
            verdict = "real"
        else:
            verdict = "fake"
            
        if (verdict=="real"):
            distance = ((np.abs(stats-avg_std_fake))/(avg_std_fake - avg_std_real))*100
            distance = min(distance, 95)
            
        if (verdict=="fake"):
            distance = ((np.abs(stats-avg_std_real))/(avg_std_fake - avg_std_real))*100
            distance = min(distance, 95)
            

        return jsonify({
            "job_id": job_id,
            "status": "done",
            "verdict": verdict,
            "distance": distance,
            "stats_used": stats,
            "features": {
                "sharpness_mean": features[0],
                "sharpness_std": features[1],
                "gradient_mean": features[2],
                "gradient_std": features[3],
                "temporal_variation": features[4]
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
