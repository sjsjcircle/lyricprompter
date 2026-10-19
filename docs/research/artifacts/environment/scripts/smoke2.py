import time, numpy as np
import samplerate, soxr, torch, torchaudio
print("samplerate", samplerate.__version__)
# torchcodec
try:
    import torchcodec; from torchcodec.decoders import AudioDecoder
    print("torchcodec OK", torchcodec.__version__)
except Exception as e:
    print("torchcodec FAIL", type(e).__name__, str(e)[:400])
# audio separator
from audio_separator.separator import Separator
print("audio_separator.Separator import OK")
# ctc-forced-aligner (Mahmoud main)
import ctc_forced_aligner as cfa
from ctc_forced_aligner import ctc_forced_aligner as ext
print("ctc_forced_aligner", cfa.__version__, "C++ ext:", ext.__file__.split('/')[-1])
# torchaudio forced_align on cpu and mps
T, C = 200, 30
em = torch.randn(1, T, C).log_softmax(-1)
tgt = torch.randint(1, C, (1, 40), dtype=torch.int32)
a, s = torchaudio.functional.forced_align(em, tgt, blank=0)
print("torchaudio.forced_align cpu OK", a.shape)
try:
    a2, s2 = torchaudio.functional.forced_align(em.to('mps'), tgt.to('mps'), blank=0)
    print("torchaudio.forced_align mps OK")
except Exception as e:
    print("torchaudio.forced_align mps FAIL:", type(e).__name__, str(e)[:200])
b = torchaudio.pipelines.MMS_FA
print("MMS_FA sample_rate", b.sample_rate, "labels", len(b.get_labels()))
# realtime resampling cost: 48k -> 16k, 10 ms blocks (480 samples)
x = np.random.randn(48000*10).astype(np.float32)
for name, fn in [
  ("soxr.ResampleStream", lambda: (lambda rs: [rs.resample_chunk(x[i:i+480]) for i in range(0, len(x), 480)])(soxr.ResampleStream(48000,16000,1,dtype='float32',quality='HQ'))),
  ("samplerate.Resampler sinc_fastest", lambda: (lambda r: [r.process(x[i:i+480], 16000/48000) for i in range(0,len(x),480)])(samplerate.Resampler('sinc_fastest',1))),
]:
    t0=time.perf_counter(); out=fn(); dt=time.perf_counter()-t0
    print(f"{name}: 10 s audio in {dt*1000:.1f} ms total, {dt/len(out)*1e6:.1f} us/block, out samples={sum(len(o) for o in out)}")
from scipy.signal import resample_poly
t0=time.perf_counter(); y=[resample_poly(x[i:i+480],1,3) for i in range(0,len(x),480)]; dt=time.perf_counter()-t0
print(f"scipy.resample_poly per-block (stateless): {dt*1000:.1f} ms total")
import sounddevice as sd
print(sd.query_hostapis())
for d in sd.query_devices():
    print(d['index'], repr(d['name']), 'in', d['max_input_channels'], 'out', d['max_output_channels'], 'sr', d['default_samplerate'], 'lowin', round(d['default_low_input_latency'],4))
