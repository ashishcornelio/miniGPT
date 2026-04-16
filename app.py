from flask import Flask, request, redirect, url_for, render_template_string
import threading
import os
import sys
import time

try:
    import psutil
except ImportError:
    import subprocess
    subprocess.call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except ImportError:
    import subprocess
    subprocess.call([sys.executable, "-m", "pip", "install", "nvidia-ml-py"])
    try:
        import pynvml
        pynvml.nvmlInit()
        HAS_NVML = True
    except:
        HAS_NVML = False
except Exception:
    HAS_NVML = False

from train_complete import train_model
from generate_with_model import load_model_and_tokenizer, generate_code

import logging
class NoLogsFilter(logging.Filter):
    def filter(self, record):
        return 'GET /logs' not in record.getMessage()
logging.getLogger('werkzeug').addFilter(NoLogsFilter())

app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace-with-secure-key'

training_thread = None
training_process = None
training_state = {
    'status': 'idle',
    'message': 'Waiting for user action.',
    'start_time': None,
    'end_time': None,
    'error': None,
}

TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MiniGPT Web UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background-color: #121212; color: #e0e0e0; }
    input, select, textarea { width: 100%; padding: 8px; margin: 4px 0 10px; background-color: #2a2a2a; color: #ffffff; border: 1px solid #444; border-radius: 4px; }
    button { padding: 10px 16px; border-radius: 4px; border: none; background-color: #4CAF50; color: white; cursor: pointer; font-weight: bold; }
    button:hover { filter: brightness(1.1); }
    .section { border: 1px solid #333; padding: 16px; margin-bottom: 20px; border-radius: 8px; background-color: #1e1e1e; }
    .status { font-weight: bold; color: #4CAF50; }
    .error { color: #f44336; }
    .log { background: #000000; color: #00FF00; padding: 10px; border-radius: 4px; height: 180px; overflow: auto; white-space: pre-wrap; font-family: monospace; border: 1px solid #333; }
    .dashboard-card { background-color: #2a2a2a; padding: 15px; border-radius: 8px; flex: 1; border: 1px solid #444; }
    .dashboard-container { display: flex; gap: 20px; margin-bottom: 20px; }
    .stat-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #333; font-size: 14px; }
  </style>
</head>
<body>
  <h1>MiniGPT Web UI</h1>

  <div class="section">
    <h2>Status</h2>
    <p>State: <span class="status">{{ state.status }}</span></p>
    <p>Message: {{ state.message }}</p>
    <p>Started: {{ state.start_time or 'N/A' }}</p>
    <p>Completed: {{ state.end_time or 'N/A' }}</p>
    {% if state.error %}<p class="error">Error: {{ state.error }}</p>{% endif %}
  </div>

  <div class="section">
    <h2>Dataset Preparation</h2>
    <form action="/prepare" method="post">
      <label>Preparation mode</label>
      <select name="mode">
        <option value="0">Skip / Use existing</option>
        <option value="1" selected>Quick expansion (expand_data.py)</option>
        <option value="2">Download from Hugging Face + prepare_dataset</option>
      </select>
      <button type="submit">Run Preparation</button>
    </form>
  </div>

  <div class="section">
    <h2>Train Model</h2>
    <form action="/train" method="post">
      <label>Data file</label>
      <input type="text" name="data_file" value="combined_training_data.txt" />
      <label>Epochs</label>
      <input type="number" name="epochs" min="1" value="3" />
      <label>Batch size</label>
      <input type="number" name="batch_size" min="1" value="8" />
      <label>Learning Rate</label>
      <input type="text" name="learning_rate" value="1e-4" />
      <label>Max length</label>
      <input type="number" name="max_length" min="16" value="256" />
      <label>Chunk size</label>
      <input type="number" name="chunk_size" min="50000" value="500000" />
      <button type="submit">Start Training</button>
    </form>
  </div>

  <div class="section">
    <h2>Generate Code</h2>
    <form action="/generate" method="post">
      <label>Prompt</label>
      <textarea name="prompt" rows="3">def </textarea>
      <label>Model path</label>
      <input type="text" name="model" value="checkpoints/minigpt_final.pt" />
      <label>Max length</label>
      <input type="number" name="length" value="100" />
      <label>Temperature</label>
      <input type="text" name="temp" value="0.7" />
      <button type="submit">Generate</button>
    </form>

    {% if generate_result %}
    <h3>Generated Output</h3>
    <div class="log" style="height: 400px;">{{ generate_result }}</div>
    {% endif %}
  </div>

  <div class="section" style="border-color: #B22222; background-color: #2b1111;">
    <h2 style="color: #ff6b6b;">Emergency Controls</h2>
    <div style="display: flex; gap: 10px;">
      <form action="/emergency" method="post" onsubmit="return confirm('Are you sure you want to terminate training and shut down the server?');">
        <button type="submit" style="background-color: #B22222; color: white; border: none; font-weight: bold; cursor: pointer; padding: 10px 20px; border-radius: 4px;">Terminate Everything</button>
      </form>
      <form action="/delete_data" method="post" onsubmit="return confirm('Are you sure you want to permanently delete all trained checkpoints, tokenizers, and logs? This cannot be undone.');">
        <button type="submit" style="background-color: #FF4500; color: white; border: none; font-weight: bold; cursor: pointer; padding: 10px 20px; border-radius: 4px;">Delete Trained Data</button>
      </form>
      <form action="/delete_datasets" method="post" onsubmit="return confirm('Are you sure you want to permanently delete all downloaded text datasets?');">
        <button type="submit" style="background-color: #f39c12; color: white; border: none; font-weight: bold; cursor: pointer; padding: 10px 20px; border-radius: 4px;">Delete Datasets</button>
      </form>
    </div>
  </div>

  <div class="dashboard-container">
    <div class="dashboard-card">
      <h3 style="margin-top:0; color:#4CAF50;">⚙️ System (CPU/RAM)</h3>
      <div class="stat-row"><span>CPU Clock:</span> <span id="cpu_clock">Loading...</span></div>
      <div class="stat-row"><span>CPU Temp:</span> <span id="cpu_temp">Loading...</span></div>
      <div class="stat-row"><span>CPU TDP:</span> <span id="cpu_tdp">Loading...</span></div>
      <div class="stat-row"><span>RAM Usage:</span> <span id="ram_usage">Loading...</span></div>
    </div>
    <div class="dashboard-card">
      <h3 style="margin-top:0; color:#2196F3;">🎮 GPU (<span id="gpu_name">Loading...</span>)</h3>
      <div class="stat-row"><span>GPU Clock:</span> <span id="gpu_clock">Loading...</span></div>
      <div class="stat-row"><span>GPU Temp:</span> <span id="gpu_temp">Loading...</span></div>
      <div class="stat-row"><span>GPU Power:</span> <span id="gpu_power">Loading...</span></div>
      <div class="stat-row"><span>VRAM Usage:</span> <span id="vram_usage">Loading...</span></div>
    </div>
  </div>

  <div class="section">
    <h2>Terminal Output</h2>
    <div class="log" id="terminal-log">{{ logs }}</div>
  </div>

  <script>
    function updateStats() {
        fetch('/api/stats')
            .then(r => r.json())
            .then(data => {
                document.getElementById('cpu_clock').innerText = data.cpu_clock;
                document.getElementById('cpu_temp').innerText = data.cpu_temp;
                document.getElementById('cpu_tdp').innerText = data.cpu_tdp;
                document.getElementById('ram_usage').innerText = data.ram_used + ' / ' + data.ram_total;
                
                document.getElementById('gpu_name').innerText = data.gpu.name;
                document.getElementById('gpu_clock').innerText = data.gpu.clock;
                document.getElementById('gpu_temp').innerText = data.gpu.temp;
                document.getElementById('gpu_power').innerText = data.gpu.power;
                document.getElementById('vram_usage').innerText = data.gpu.vram_used + ' / ' + data.gpu.vram_total;
            })
            .catch(e => console.log('Stats error:', e));
    }
    setInterval(updateStats, 2000);
    updateStats();
    var logDiv = document.getElementById("terminal-log");
    if(logDiv) {
        logDiv.scrollTop = logDiv.scrollHeight;
        setInterval(function() {
            fetch('/logs')
              .then(r => r.text())
              .then(text => {
                  if (text && text !== logDiv.innerText) {
                      var isBottom = (logDiv.scrollHeight - logDiv.scrollTop <= logDiv.clientHeight + 10);
                      logDiv.innerText = text;
                      if (isBottom) {
                          logDiv.scrollTop = logDiv.scrollHeight;
                      }
                  }
              });
        }, 1000);
    }
  </script>
</body>
</html>
'''

terminal_output = ""

def append_log(message):
    global terminal_output
    import time
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    msg = f"[{timestamp}] {message}\n"
    terminal_output += msg
    print(msg, end="")
    if len(terminal_output) > 50000:
        terminal_output = terminal_output[-50000:]


def training_worker(params):
    import subprocess
    global terminal_output, training_process
    try:
        training_state['status'] = 'running'
        training_state['message'] = 'Training started'
        training_state['start_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        training_state['end_time'] = None
        training_state['error'] = None

        append_log('Training started')

        cmd = [
            sys.executable, "train_complete.py",
            "--data", str(params['data_file']),
            "--epochs", str(params['epochs']),
            "--batch-size", str(params['batch_size']),
            "--lr", str(params['learning_rate']),
            "--max-length", str(params['max_length']),
            "--chunk-size", str(params['chunk_size']),
            "--name", "minigpt_web"
        ]
        
        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        
        training_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8', 
            errors='replace', 
            bufsize=1,
            env=env
        )
        process = training_process
        
        while True:
            char = process.stdout.read(1)
            if not char and process.poll() is not None:
                break
            if char:
                sys.stdout.write(char)
                sys.stdout.flush()
                if char == '\r':
                    idx = terminal_output.rfind('\n')
                    if idx != -1:
                        terminal_output = terminal_output[:idx+1]
                    else:
                        terminal_output = ""
                else:
                    terminal_output += char

                if len(terminal_output) > 50000:
                    terminal_output = terminal_output[-50000:]
                    
        if process.returncode != 0:
            raise Exception(f"Training failed with exit code {process.returncode}")

        training_state['status'] = 'completed'
        training_state['message'] = 'Training completed successfully'
        training_state['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        append_log('Training completed successfully')

    except Exception as ex:
        training_state['status'] = 'error'
        training_state['message'] = str(ex)
        training_state['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        training_state['error'] = str(ex)
        append_log(f'Error: {ex}')


@app.route('/logs')
def get_logs():
    return terminal_output or ""

@app.route('/')
def index():
    return render_template_string(
        TEMPLATE,
        state=training_state,
        logs=terminal_output,
        generate_result=None
    )


def prepare_worker(mode):
    import subprocess
    global terminal_output
    def run_cmd(cmd_list):
        global terminal_output
        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        process = subprocess.Popen(
            cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, encoding='utf-8', errors='replace', bufsize=1, env=env
        )
        while True:
            char = process.stdout.read(1)
            if not char and process.poll() is not None:
                break
            if char:
                sys.stdout.write(char)
                sys.stdout.flush()
                if char == '\r':
                    idx = terminal_output.rfind('\n')
                    if idx != -1:
                        terminal_output = terminal_output[:idx+1]
                    else:
                        terminal_output = ""
                else:
                    terminal_output += char
                if len(terminal_output) > 50000:
                    terminal_output = terminal_output[-50000:]
        if process.returncode != 0:
            append_log(f"Error: {' '.join(cmd_list)} failed with code {process.returncode}\n")

    if mode == '1':
        append_log('Running quick expand_data.py...\n')
        run_cmd([sys.executable, 'expand_data.py'])
    elif mode == '2':
        append_log('Running download_datasets.py...\n')
        run_cmd([sys.executable, 'download_datasets.py'])
        append_log('Running prepare_dataset.py...\n')
        run_cmd([sys.executable, 'prepare_dataset.py'])
    else:
        append_log('Skipping dataset preparation\n')
    append_log('Dataset preparation sequence done.\n')


@app.route('/prepare', methods=['POST'])
def prepare():
    mode = request.form.get('mode', '1')
    threading.Thread(target=prepare_worker, args=(mode,), daemon=True).start()
    return redirect(url_for('index'))


@app.route('/train', methods=['POST'])
def train():
    global training_thread

    if training_state['status'] == 'running':
        training_state['message'] = 'Training is already running.'
        return redirect(url_for('index'))

    data_file = request.form.get('data_file', 'combined_training_data.txt')
    epochs = int(request.form.get('epochs', 3))
    batch_size = int(request.form.get('batch_size', 8))
    learning_rate = float(request.form.get('learning_rate', 1e-4))
    max_length = int(request.form.get('max_length', 256))
    chunk_size = int(request.form.get('chunk_size', 500000))

    params = {
        'data_file': data_file,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_length': max_length,
        'chunk_size': chunk_size
    }

    training_thread = threading.Thread(target=training_worker, args=(params,), daemon=True)
    training_thread.start()
    append_log('Training thread launched')

    return redirect(url_for('index'))


@app.route('/generate', methods=['POST'])
def generate():
    model_path = request.form.get('model', 'checkpoints/minigpt_final.pt')
    prompt = request.form.get('prompt', 'def ')
    length = int(request.form.get('length', 100))
    temp = float(request.form.get('temp', 0.7))

    device = 'cuda' if bool(os.getenv('FORCE_CUDA', '1')) and os.path.exists('venv312') and os.path.exists('tokenizer') else 'cpu'
    model, tokenizer = load_model_and_tokenizer(model_path, device=device)

    if model is None or tokenizer is None:
        return render_template_string(TEMPLATE,
                                      state=training_state,
                                      logs=terminal_output,
                                      generate_result='Model or tokenizer not available.')

    generated = generate_code(model, tokenizer, prompt, max_length=length, temperature=temp, device=device)
    append_log('Generated output for prompt: ' + prompt)

    return render_template_string(TEMPLATE, state=training_state, logs=terminal_output, generate_result=generated)


@app.route('/emergency', methods=['POST'])
def emergency():
    global training_process
    append_log('EMERGENCY STOP TRIGGERED!')
    if training_process is not None:
        try:
            training_process.terminate()
            training_process.kill()
            append_log('Training process killed.')
        except Exception as e:
            append_log(f'Error killing training process: {e}')
    
    append_log('Shutting down server immediately...')
    def shutdown():
        import time
        time.sleep(1)
        import os
        os._exit(0)
    threading.Thread(target=shutdown).start()
    return render_template_string("<h1>Session Terminated</h1><p>The server has been shut down. You can close this window.</p>")


@app.route('/delete_data', methods=['POST'])
def delete_data():
    append_log('Running delete_trained_data.py...')
    os.system('python delete_trained_data.py')
    append_log('Deleted previously trained data.')
    return redirect(url_for('index'))


@app.route('/delete_datasets', methods=['POST'])
def delete_datasets():
    append_log('Deleting downloaded datasets...\n')
    datasets = ['combined_training_data.txt', 'python_training_data.txt', 'training_corpus.txt', 'data.txt']
    import os
    deleted = 0
    for file in datasets:
        if os.path.exists(file):
            try:
                os.remove(file)
                append_log(f'Deleted {file}\n')
                deleted += 1
            except Exception as e:
                append_log(f'Failed to delete {file}: {e}\n')
    if deleted == 0:
        append_log('No datasets found to delete.\n')
    append_log('Dataset deletion complete.\n')
    return redirect(url_for('index'))


@app.route('/api/stats')
def api_stats():
    import psutil
    
    cpu_freq = psutil.cpu_freq()
    cpu_clock = f"{cpu_freq.current:.0f} MHz" if cpu_freq else "N/A"
    
    ram = psutil.virtual_memory()
    ram_used = f"{ram.used / 1024**3:.1f} GB"
    ram_total = f"{ram.total / 1024**3:.1f} GB"
    
    cpu_temp = "N/A"
    if hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()
            if temps and 'coretemp' in temps:
                cpu_temp = f"{temps['coretemp'][0].current}°C"
        except:
            pass
            
    gpu_stats = {"name": "N/A", "temp": "N/A", "vram_used": "N/A", "vram_total": "N/A", "clock": "N/A", "power": "N/A"}
    
    global HAS_NVML
    if HAS_NVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            gpu_stats["name"] = name.decode('utf-8') if isinstance(name, bytes) else name
            
            temp_raw = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            gpu_stats["temp"] = f"{temp_raw}°C" if temp_raw < 200 else "N/A (Asleep/D3)"
            
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_stats["vram_used"] = f"{mem.used / 1024**3:.1f} GB"
            gpu_stats["vram_total"] = f"{mem.total / 1024**3:.1f} GB"
            
            clock_raw = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            gpu_stats["clock"] = f"{clock_raw} MHz" if clock_raw < 100000 else "N/A"
            
            try:
                power_raw = pynvml.nvmlDeviceGetPowerUsage(handle)
                gpu_stats["power"] = f"{power_raw / 1000.0:.1f} W" if power_raw < 10000000 else "N/A (Asleep/D3)"
            except pynvml.NVMLError:
                gpu_stats["power"] = "N/A"
        except Exception as e:
            gpu_stats["name"] = f"Error: {str(e)}"
            
    return {
        "cpu_clock": cpu_clock,
        "cpu_temp": cpu_temp,
        "cpu_tdp": "N/A (Win OS hidden)",
        "ram_used": ram_used,
        "ram_total": ram_total,
        "gpu": gpu_stats
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
