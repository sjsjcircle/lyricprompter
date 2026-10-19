import time, torch, torch.nn as nn
torch.set_num_threads(4)
class Tiny(nn.Module):  # CRNN-ish phoneme posterior net, similar scale to the reference repo's models
    def __init__(s):
        super().__init__(); s.c=nn.Sequential(nn.Conv1d(128,128,3,padding=1),nn.ReLU(),nn.Conv1d(128,128,3,padding=1),nn.ReLU()); s.g=nn.GRU(128,128,batch_first=True,bidirectional=True); s.o=nn.Linear(256,40)
    def forward(s,x): y=s.c(x).transpose(1,2); y,_=s.g(y); return s.o(y).log_softmax(-1)
for dev in ['cpu','mps']:
    m=Tiny().to(dev).eval(); x=torch.randn(1,128,5,device=dev)
    with torch.inference_mode():
        for _ in range(20): m(x)
        if dev=='mps': torch.mps.synchronize()
        t0=time.perf_counter()
        for _ in range(200):
            y=m(x); y=y.cpu()
        dt=(time.perf_counter()-t0)/200
    print(f"{dev}: {dt*1000:.2f} ms per 5-frame hop inference (incl. .cpu())")
