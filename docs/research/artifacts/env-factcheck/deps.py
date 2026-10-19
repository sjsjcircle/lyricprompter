import json, sys, urllib.request
for spec in sys.argv[1:]:
    pkg, _, ver = spec.partition("==")
    url = f"https://pypi.org/pypi/{pkg}/{ver}/json" if ver else f"https://pypi.org/pypi/{pkg}/json"
    d = json.load(urllib.request.urlopen(url, timeout=30))
    print("==", pkg, d["info"]["version"], "rp=", d["info"].get("requires_python"))
    for r in d["info"].get("requires_dist") or []:
        print("   ", r)
