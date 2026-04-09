import os
import subprocess
import time
from flask import Flask, request, send_file
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": ["http://10.6.95.43:5173", "https://10.6.95.43:5173", "http://localhost:5000", "https://localhost:5000"]}})
app.config['MAX_CONTENT_LENGTH'] = 600 * 1024 * 1024

def segment_ply(input_path):
    command = [
        "./dist/test",
        "--log", "/home/leith/AdaptConv-master/sem_seg/train",
        "--infile", f"/home/leith/AdaptConv-master/sem_seg/data/input.ply",
        "--outdir", "/home/leith/AdaptConv-master/sem_seg/output",
        "--iters", "0",
        "--outtype", "class"
    ]

    try:
        result = subprocess.run(command, check=False, capture_output=False, text=False)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    except FileNotFoundError:
        print(f"Error: The executable or file was not found. Check your paths.")


@app.route('/process-ply', methods=['POST'])
def process_ply():
    if not request.data:
        return "No data received", 400

    input_tmp = f"/home/leith/AdaptConv-master/sem_seg/data/input.ply"
    output_tmp = f"/home/leith/AdaptConv-master/sem_seg/output/input.ply"

    try:
        with open(input_tmp, 'wb') as f:
            f.write(request.data)
        segment_ply(input_tmp)
        return send_file(output_tmp, mimetype='application/octet-stream')
    except Exception as e:
        return str(e), 500
    finally:
        print('FAILED')
        if os.path.exists(input_tmp): os.remove(input_tmp)
        if os.path.exists(output_tmp): os.remove(output_tmp)

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)