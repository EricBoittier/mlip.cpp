#!/usr/bin/env python3
"""Sequential, matched CPU/Metal conservative-training benchmark (no concurrent runs)."""
import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--init', type=Path, required=True)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--steps', type=int, default=22)
p.add_argument('--warmup', type=int, default=2)
p.add_argument('--batches', type=int, nargs='+', default=[1, 3])
p.add_argument('--cpu-threads', type=int, nargs='+', default=[2, 4, 8])
a = p.parse_args()
if not 0 <= a.warmup < a.steps:
    p.error('warmup must be nonnegative and less than steps')
a.output.mkdir(parents=True, exist_ok=True)
report = dict(platform=platform.platform(), machine=platform.machine(),
              model=str(a.init.resolve()), model_sha256=hashlib.sha256(a.init.read_bytes()).hexdigest(),
              data=str(a.data.resolve()), steps=a.steps, warmup=a.warmup, runs=[])
for batch in a.batches:
    for backend, threads in [('cpu', t) for t in a.cpu_threads] + [('metal', 2)]:
        name = f'{backend}-t{threads}-b{batch}'
        command = [str(a.binary.resolve()), '--init', str(a.init.resolve()), '--data', str(a.data.resolve()),
                   '--backend', backend, '--threads', str(threads), '--batch-size', str(batch),
                   '--method', 'analytic', '--steps', str(a.steps), '--seed', '42', '--lr', '0.0001',
                   '--output', str((a.output / (name + '.gguf')).resolve())]
        start = time.perf_counter()
        with (a.output / (name + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        elapsed = time.perf_counter() - start
        text = (a.output / (name + '.log')).read_text()
        times = [float(t) for t in re.findall(r'^step=.* seconds=([\d.]+)$', text, re.M)]
        if len(times) != a.steps:
            raise RuntimeError(f'{name}: missing step timing records')
        steady = times[a.warmup:]
        r = dict(name=name, command=command, backend=backend, threads=threads, batch=batch,
                 process_seconds=elapsed, step_seconds=times,
                 median_step_seconds=statistics.median(steady), mean_step_seconds=statistics.mean(steady),
                 structures_per_second=batch/statistics.mean(steady))
        report['runs'].append(r)
        (a.output / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'{name}: {r["mean_step_seconds"]:.4f} s/update, {r["structures_per_second"]:.3f} structures/s', flush=True)
