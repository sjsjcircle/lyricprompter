import json,sys,glob,os,re
for f in sorted(glob.glob('*.json')):
    d=json.load(open(f))
    info=d['info']; v=info['version']
    files=d['urls']
    upload=min((x['upload_time'] for x in files), default='?')
    wheels=[x['filename'] for x in files if x['filename'].endswith('.whl')]
    mac=[w for w in wheels if 'macosx' in w and ('arm64' in w or 'universal2' in w)]
    pure=[w for w in wheels if 'none-any' in w]
    cps=sorted(set(re.findall(r'-(cp3\d+t?)-', ' '.join(mac))))
    print(f"{info['name']} {v} ({upload[:10]}) req_py={info.get('requires_python')} lic={(info.get('license_expression') or (info.get('license') or '')[:40]).strip()!r}")
    print(f"   mac_arm64_cp={cps} pure={bool(pure)} sdist_only={not wheels}")
    if mac: print("   e.g.", mac[0])
