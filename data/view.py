import json

ifp = "data/results.json"
with open(ifp, "r", encoding="utf-8") as f:
    data = json.load(f)
    
print(data[0])
print(len(data))

a, b, c = 0, 0, 0

notalign_id = []
error = []
for d in data:
    if d[1] > 1 and d[1] != 267267267267267:
        a += 1
        notalign_id.append(d[0])
    elif d[1] < 1:
        b += 1
    elif d[1] == 267267267267267:
        error.append(d[0])
        c += 1
    else:
        print(d)
        
print(a, b, c)

ifp = "data/remain.json"
with open(ifp, "r", encoding="utf-8") as f:
    remain = json.load(f)
    
notalign = []
for r in remain:
    if r["id"] in notalign_id:
        notalign.append(r)
    if r["id"] in error:
        notalign.append(r)

print(len(notalign))

i = 0
for d in notalign:
    if len(d["output"]) == 0:
        i += 1
print(i)

ofp = "data/augdata.json"
with open(ofp, "w", encoding="utf-8") as f:
    json.dump(notalign, f, ensure_ascii=False, indent=4)