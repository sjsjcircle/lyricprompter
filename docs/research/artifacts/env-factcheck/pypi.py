import json, sys, urllib.request
for pkg in sys.argv[1:]:
    try:
        d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=30))
    except Exception as e:
        print(pkg, "ERR", e); continue
    info = d["info"]
    v = info["version"]
    rel = d["releases"].get(v, [])
    up = min((f["upload_time"] for f in rel), default=None)
    print(f"== {pkg} {v} uploaded={up} requires_python={info.get('requires_python')} license={info.get('license_expression') or (info.get('license') or '')[:60]!r} home={info.get('home_page')} urls={ {k:v for k,v in (info.get('project_urls') or {}).items()} }")
    macs = sorted(f["filename"] for f in rel if "macosx" in f["filename"] and ("arm64" in f["filename"] or "universal2" in f["filename"]))
    other = [f["filename"] for f in rel if f["filename"].endswith((".tar.gz",".zip")) or "none-any" in f["filename"] or "abi3" in f["filename"]]
    print("   mac:", macs[:20])
    print("   other:", other[:5])
