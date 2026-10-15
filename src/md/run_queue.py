"""
run_queue.py - run the MD queue on ONE GPU machine, several simulations at the same time.

Runs every job in BASE/inputs/queue.json (made by gather_md_inputs.py) with run_md.py, N at a time.
Safe to stop (Ctrl+C or a dropped connection) and restart: finished jobs are skipped and unfinished ones resume
from their last checkpoint.

Typical use on the Jetstream2 machine (inside the 'md' conda environment):
  cd ~/md
  nohup python run_queue.py --base ~/md --workers 4 > queue.log 2>&1 &
  tail -f queue.log        (Ctrl+C stops watching, NOT the runs)
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="folder that contains run_md.py and inputs/queue.json")
ap.add_argument("--workers", type=int, default=4, help="simulations running at the same time")
ap.add_argument("--only", default="", help="only run jobs whose name contains this text")
ap.add_argument("--status-sec", type=int, default=300)
args = ap.parse_args()

base = os.path.expanduser(args.base)
queue = json.load(open(os.path.join(base, "inputs", "queue.json")))
stamp = lambda: time.strftime("%H:%M:%S")

def finished(job):
    n = job["name"]
    return os.path.exists(os.path.join(base, "runs", n, f"{n}_final.pdb"))

todo = [j for j in queue if args.only in j["name"] and not finished(j)]
print(f"[{stamp()}] {len(queue)} jobs in the queue, {len(todo)} still to do, {args.workers} at a time", flush=True)

def logfile(job):
    return os.path.join(base, "runs", job["name"], "stdout.log")

def run(job):
    out = os.path.join(base, "runs", job["name"])
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(base, "run_md.py"), "--pdb", os.path.join(base, "inputs", job["pdb"]),
           "--out-dir", out, "--name", job["name"], "--ns", str(job["ns"]), "--seed", str(job["seed"])] + job.get("extra", [])
    t0 = time.time()
    with open(logfile(job), "a") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    return job["name"], rc, (time.time() - t0) / 60

def last_line(job):
    try:
        lines = [l.strip() for l in open(logfile(job)) if l.strip()]
        return lines[-1] if lines else "(starting)"
    except OSError:
        return "(starting)"

failed = []
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    pending = {pool.submit(run, j): j for j in todo}
    started = set()
    while pending:
        done, _ = wait(list(pending), timeout=args.status_sec, return_when=FIRST_COMPLETED)
        for f in done:
            job = pending.pop(f)
            name, rc, mins = f.result()
            if rc != 0: failed.append(name)
            print(f"[{stamp()}] {name}: {'ok' if rc == 0 else 'FAILED (exit code %d, see %s)' % (rc, logfile(job))} after {mins:.0f} min", flush=True)
        running = [j for f, j in pending.items() if f.running()]
        if running:
            print(f"[{stamp()}] still running ({len(pending)} left incl. waiting):", flush=True)
            for j in running:
                print(f"     {j['name']:<28} {last_line(j)}", flush=True)

print(f"\n[{stamp()}] queue pass finished. " + (f"FAILED: {', '.join(failed)}" if failed else "No failures."), flush=True)
