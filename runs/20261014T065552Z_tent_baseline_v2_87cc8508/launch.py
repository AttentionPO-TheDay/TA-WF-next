import fcntl, json, os, subprocess
from pathlib import Path

R = Path(__file__).resolve().parent
C = json.loads((R / "config.json").read_text())
with (R / "artifacts/launch.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert C["status"] == "frozen"
    assert json.loads((R / "artifacts/preflight.json").read_text())["passed"]
    assert not (R / "artifacts/launch.json").exists()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(C["physical_gpu"]),
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    cmd = [C["python"], "-u", str(R / "runner.py")]
    with (R / "logs/runner.out").open("ab", buffering=0) as out:
        process = subprocess.Popen(cmd, cwd=R.parents[1], env=env, stdout=out,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    data = {"pid": process.pid, "command": cmd, "gpu": C["physical_gpu"],
            "new_tasks": 18, "inference_control_tasks": 9}
    (R / "artifacts/launch.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data))
