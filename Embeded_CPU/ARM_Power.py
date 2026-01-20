# Final benchmarking script to run ON THE ZYNQ

import onnxruntime as ort
import numpy as np
import timeit
from sklearn.metrics import mean_squared_error, r2_score
from reservoirpy.observables import rmse,nrmse, rsquare   # Example metric


import threading
import time
import os

class ZynqPowerMonitor:

    def __init__(self, hwmon_path=None, interval_sec=0.1):

        self.interval = interval_sec
        self.power_readings_watts = []
        
        self._stop_event = threading.Event()
        self._monitor_thread = None

        if hwmon_path:
            self.power_file_path = os.path.join(hwmon_path, 'power1_input')
        else:
            # Attempt to find the power file automatically
            self.power_file_path = self._find_power_file()

        if not os.path.exists(self.power_file_path):
            raise FileNotFoundError(f"Power monitor file not found at '{self.power_file_path}'. "
                                    "Please find the correct path using 'find /sys/bus/i2c/devices/ -name \"power1_input\"'")

    def _find_power_file(self):
        """Tries to locate the 'power1_input' file automatically."""
        for root, dirs, files in os.walk('/sys/bus/i2c/devices/'):
            if 'power1_input' in files:
                # This logic could be improved to pick a specific sensor if there are multiple.
                # For most boards, the first one found is for the main PS rails.
                return os.path.join(root, 'power1_input')
        return None

    def _get_power_reading_watts(self):
        """Reads the sysfs file and converts from microwatts to Watts."""
        try:
            with open(self.power_file_path, 'r') as f:
                power_microwatts = int(f.read())
            # Convert from microwatts to Watts
            return power_microwatts / 1_000_000.0
        except Exception as e:
            print(f"Warning: Could not read power file. Error: {e}")
            return None

    def _monitor_power_loop(self):
        """The main loop for the monitoring thread."""
        while not self._stop_event.is_set():
            power = self._get_power_reading_watts()
            if power is not None:
                self.power_readings_watts.append(power)
            time.sleep(self.interval)

    def start(self):
        """Starts the power monitoring background thread."""
        print("Starting Zynq power monitor...")
        self.power_readings_watts = []
        self._stop_event.clear()
        self.start_time = time.time()
        self._monitor_thread = threading.Thread(target=self._monitor_power_loop)
        self._monitor_thread.start()

    def stop(self):
        """Stops the monitoring thread and returns the calculated metrics."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join()
        self.end_time = time.time()
        print("Power monitor stopped.")
        
        duration_sec = self.end_time - self.start_time
        avg_power = np.mean(self.power_readings_watts) if self.power_readings_watts else 0.0
        energy_j = avg_power * duration_sec
        latency_ms = duration_sec * 1000
        
        return latency_ms, avg_power, energy_j

# --- Helper function for ONNX Runtime inference ---
def run_ort_inference(model_path, test_data, execution_provider):
    sess_options = ort.SessionOptions()
    ort_session = ort.InferenceSession(model_path, sess_options, providers=[execution_provider])
    input_name = ort_session.get_inputs()[0].name
    # The [0] at the end gets the 'output_sequence' tensor
    predictions = ort_session.run(None, {input_name: test_data})[0]
    return predictions

# --- Main Execution on Zynq ---
if __name__ == '__main__':
    # Load your test data on the Zynq
    X_test = np.load('X_test.npy').astype(np.float32)
    y_test = np.load('y_test.npy').astype(np.float32)
    start, end = 10, 100

    # Reshape test data for the model: (1, seq_len, 1)
    test_input_data_np = X_test.reshape(1, -1, 1)
    print(f"Test input size: {test_input_data_np.size}")
    PS_POWER_HWMON_PATH = "/sys/bus/i2c/devices/4-0043/hwmon/hwmon0/"

    try:
        # We explicitly provide the path to the ZynqPowerMonitor constructor.
        zynq_monitor = ZynqPowerMonitor(hwmon_path=PS_POWER_HWMON_PATH)
    except FileNotFoundError as e:
        print(e)
        print("\nERROR: The provided hwmon path is incorrect or the power monitoring driver is not loaded.")
        print(f"Please verify this path exists on your Zynq system: {PS_POWER_HWMON_PATH}")
        exit()

    models_to_benchmark = {
        "ONNX FP32": "regression_esn_fp32.onnx",
        "ONNX INT8": "regression_esn_uint8.onnx"
    }
    
    all_results = {}



    for name, path in models_to_benchmark.items():
        print(f"Benchmarking {name}...")
        
        exec_provider = 'CPUExecutionProvider'

        # Time the full inference run
        def workload():
            return run_ort_inference(path, test_input_data_np, exec_provider)
        
        # Warm-up run
        _ = workload()

        zynq_monitor.start()
        predictions_np = workload()
        latency, avg_power, energy = zynq_monitor.stop()
        start, end = 10, 100

        total_latency = latency / 1000.0
        cpu_latency = total_latency/test_input_data_np.size
        # Throughput in thousands of timesteps per second
        throughput = (test_input_data_np.size / total_latency)
        print(predictions_np.shape)
        # Calculate accuracy metrics
        y_true = y_test[start:end].flatten()
        y_pred_sliced = predictions_np.squeeze()[start:end]
        print(y_pred_sliced.shape)
        print(y_true.shape)
        rmse_final = rmse(y_true, y_pred_sliced)
        nrmse_final = nrmse(y_true, y_pred_sliced)
        rsquare_final = rsquare(y_true, y_pred_sliced)
        print (y_true.tolist())
        print (y_pred_sliced.tolist())
        # Store predictions for plotting
        all_results[name] = y_pred_sliced
        print("\n--- Zynq Performance & Accuracy Report (Regression Task) {name}... ---")
        print("="*100)
        header = f"{'Model':<15} | {'RMSE':<12} | {'NRMSE':<12} | {'R² Score':<12} | {'Total Latency (s)':<15} | {'CPU Latency (s)':<15} | {'Throughput (k_steps/s)':<22} | {'Avg Power (W)':<15} | {'Energy (J)':<12}"
        print(header)
        print("-" * 100)
        # Print results
        print(f"{name:<15} | {rmse_final:<12.6f} | {nrmse_final:<12.6f} | {rsquare_final:<12.6f} | {total_latency:<15.4f} | {cpu_latency:<22.2f} | {throughput:<22.2f} | {avg_power:<15.4f} | {energy:<12.4f}")
