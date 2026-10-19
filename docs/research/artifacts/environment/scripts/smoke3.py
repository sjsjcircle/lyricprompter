import sys, time, numpy as np
mp3, wav = sys.argv[1], sys.argv[2]
from torchcodec.decoders import AudioDecoder
t0=time.perf_counter(); s = AudioDecoder(mp3, sample_rate=16000, num_channels=1).get_all_samples(); print("torchcodec decode mp3->16k mono", s.data.shape, s.sample_rate, f"{(time.perf_counter()-t0)*1000:.0f} ms")
import librosa
t0=time.perf_counter(); y, sr = librosa.load(mp3, sr=16000); print("librosa.load mp3", y.shape, sr, f"{(time.perf_counter()-t0)*1000:.0f} ms")
import soundfile as sf
print("soundfile formats has MP3:", 'MP3' in sf.available_formats())
# numba-backed librosa DTW
X = np.random.rand(12, 750).astype(np.float32); Y = np.random.rand(12, 750).astype(np.float32)
t0=time.perf_counter(); D, wp = librosa.sequence.dtw(X, Y, metric='cosine'); t1=time.perf_counter()
D, wp = librosa.sequence.dtw(X, Y, metric='cosine'); t2=time.perf_counter()
print(f"librosa.sequence.dtw 750x750 first(JIT)={1000*(t1-t0):.0f} ms, second={1000*(t2-t1):.0f} ms")
import torchaudio
w, sr = torchaudio.load(wav); print("torchaudio.load (via torchcodec)", w.shape, sr)
