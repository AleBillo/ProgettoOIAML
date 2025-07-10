import subprocess
import threading

def launch_tensorboard(logdir, port=6006):
    def run_tb():
        command = ["tensorboard", "--logdir", logdir, "--port", str(port)]
        subprocess.run(command, check=True)

    thread = threading.Thread(target=run_tb, daemon=True)
    thread.start()
    print(f"TensorBoard is live at http://localhost:{port}")
