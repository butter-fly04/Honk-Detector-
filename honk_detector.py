import pyaudio
import numpy as np
import librosa
from scipy import signal
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time

class HonkDetector:
    def __init__(self, alert_sound_path='/home/username/.local/share/honk-detector/alert.wav'):
        # Audio stream parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        
        # Honk detection parameters
        self.FREQ_MIN = 2500    # Typical car honk frequency range
        self.FREQ_MAX = 4000
        self.THRESHOLD = 0.3   # Adjust based on your needs
        
        # Alert limiting parameters
        self.MAX_CONSECUTIVE_ALERTS = 2
        self.ALERT_COOLDOWN = 10  # seconds to reset consecutive count
        self.consecutive_alerts = 0
        self.last_alert_time = 0
        
        # Load alert sound
        self.alert_sound, _ = sf.read(alert_sound_path)
        
        # Initialize audio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        # Queue for handling alerts
        self.alert_queue = queue.Queue()
        
    def _play_alert(self):
        while True:
            try:
                alert = self.alert_queue.get()
                if alert is None:
                    break
                sd.play(self.alert_sound, 44100)
                sd.wait()
            except queue.Empty:
                continue
    
    def detect_honk(self, audio_chunk):
        # Convert audio chunk to numpy array
        audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
        
        # Apply bandpass filter
        nyquist = self.RATE / 2
        low = self.FREQ_MIN / nyquist
        high = self.FREQ_MAX / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.lfilter(b, a, audio_data)
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(filtered**2))
        
        return rms > self.THRESHOLD
    
    def handle_honk_detection(self):
        current_time = time.time()
        
        # Reset consecutive count if enough time has passed
        if current_time - self.last_alert_time > self.ALERT_COOLDOWN:
            self.consecutive_alerts = 0
        
        # Increment consecutive alerts
        self.consecutive_alerts += 1
        self.last_alert_time = current_time
        
        # Decide whether to play alert
        if self.consecutive_alerts <= self.MAX_CONSECUTIVE_ALERTS:
            print(f"Honk detected! Alert {self.consecutive_alerts}/{self.MAX_CONSECUTIVE_ALERTS}")
            self.alert_queue.put(True)
        else:
            print(f"Honk detected but alert suppressed (limit reached). Wait {self.ALERT_COOLDOWN} seconds for reset.")
    
    def run(self):
        # Start alert player thread
        alert_thread = threading.Thread(target=self._play_alert)
        alert_thread.start()
        
        print("Listening for honks... Press Ctrl+C to stop")
        print(f"Alert will play maximum {self.MAX_CONSECUTIVE_ALERTS} times for consecutive honks")
        print(f"Alert count resets after {self.ALERT_COOLDOWN} seconds of no honks")
        
        try:
            while True:
                audio_chunk = self.stream.read(self.CHUNK, exception_on_overflow=False)
                if self.detect_honk(audio_chunk):
                    self.handle_honk_detection()
                    time.sleep(1)  # Prevent multiple detections of the same honk
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            # Cleanup
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            self.alert_queue.put(None)
            alert_thread.join()

if __name__ == "__main__":
    detector = HonkDetector()
    detector.run()
