import subprocess, os
env = dict(os.environ); env["PYTHONPATH"]="/tmp/litert_py"
M = {
 "sparse":   "/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/face_detection_full_range_sparse.tflite",
 "gesture":  "/Users/hugocornellier/IdeaProjects/hand_detection/assets/models/gesture_embedder.tflite",
 "mobileface":"/Users/hugocornellier/IdeaProjects/flutter_litert/example/assets/mobilefacenet.tflite",
 "back_CTRL":"/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/face_detection_back.tflite",
}
ACC = {"GPU(2)":2, "GPU|CPU(3)":3}
print(f"{'model':12} {'accel':10} {'rc':>4}  outcome")
print("-"*70)
for name, path in M.items():
    for an, av in ACC.items():
        r = subprocess.run(["python3","/tmp/cm_worker.py",path,str(av)],
                           capture_output=True, text=True, env=env, timeout=120)
        out = (r.stdout.strip().splitlines() or [""])[-1]
        crash = r.returncode < 0
        tag = f"CRASH(sig{-r.returncode})" if crash else out
        print(f"{name:12} {an:10} {r.returncode:>4}  {tag}")
