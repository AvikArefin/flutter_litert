import sys
mp, accel = sys.argv[1], int(sys.argv[2])
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator as HA
try:
    m = CompiledModel.from_file(mp, hardware_accel=HA(accel))
    print("OK_CREATE")
except Exception as e:
    print(f"PYEXC {type(e).__name__}: {str(e)[:160]}")
