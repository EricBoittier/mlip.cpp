"""Split an extxyz into K contiguous parts (frame order kept): IN.xyz K -> IN.partK.xyz ... (for parallel eval)."""
import sys

fn, K = sys.argv[1], int(sys.argv[2])
frames, cur = [], []
with open(fn) as f:
    while True:
        l = f.readline()
        if not l:
            break
        n = int(l)
        frames.append(l + "".join(f.readline() for _ in range(n + 1)))
per = (len(frames) + K - 1) // K
for k in range(K):
    with open(fn[:-4] + f".part{k}.xyz", "w") as g:
        g.write("".join(frames[k * per:(k + 1) * per]))
print(len(frames), "frames ->", K, "parts of", per)
