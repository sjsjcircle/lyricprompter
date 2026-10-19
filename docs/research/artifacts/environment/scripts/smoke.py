import time, importlib, sys, platform
print(sys.version, platform.machine(), platform.mac_ver()[0])
def t(name, fn=None):
    t0=time.time()
    try:
        m=importlib.import_module(name)
        extra = fn(m) if fn else ''
        print(f"OK   {name:22s} {getattr(m,'__version__','')!s:14s} {time.time()-t0:5.2f}s {extra}")
    except Exception as e:
        print(f"FAIL {name:22s} {type(e).__name__}: {str(e)[:300]}")
t('numpy'); t('scipy'); t('numba'); t('llvmlite')
t('torch', lambda m: f"mps_available={m.backends.mps.is_available()} mps_built={m.backends.mps.is_built()}")
t('torchaudio', lambda m: f"forced_align={hasattr(m.functional,'forced_align')} MMS_FA={hasattr(m.pipelines,'MMS_FA')} has_load={hasattr(m,'load')}")
t('torchcodec')
t('librosa'); t('soundfile', lambda m: f"libsndfile={m.__libsndfile_version__}")
t('sounddevice', lambda m: f"portaudio={m.get_portaudio_version()[1]} lib={m._libname}")
t('soxr', lambda m: f"libsoxr={m.__libsoxr_version__}")
t('samplerate'); t('diffq'); t('onnxruntime', lambda m: f"providers={m.get_available_providers()}")
t('audio_separator'); t('audio_separator.separator')
t('ctc_forced_aligner'); t('uroman'); t('pykakasi'); t('fugashi'); t('unidic_lite'); t('pyphen'); t('cmudict')
t('fastapi'); t('uvicorn'); t('pydantic'); t('transformers'); t('uvloop'); t('httptools'); t('websockets')
