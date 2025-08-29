import time
import sys


def main() -> int:
    try:
        import torch
    except Exception as e:
        print(f"FAIL: Unable to import torch: {e}")
        return 1

    print(f"torch version: {torch.__version__}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return 2

    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(
        f"Current device: {torch.cuda.current_device() if torch.cuda.device_count() else 'N/A'}"
    )
    print(
        f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.device_count() else 'N/A'}"
    )

    try:
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        c = a.matmul(b)
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000
        print(f"SUCCESS: GPU matmul completed in {dt:.2f} ms; c.shape={tuple(c.shape)}")
        return 0
    except Exception as e:
        print(f"FAIL: GPU operation failed: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
