import time
import torch


def main():
    print("=" * 60)
    print("KARINA GPU TEST")
    print("=" * 60)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA build      : {torch.version.cuda}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "\nCUDA GPU를 사용할 수 없습니다.\n"
            "PyTorch가 CPU 버전으로 설치되었거나 "
            "CUDA/NVIDIA 드라이버 설정을 확인해야 합니다."
        )

    device_count = torch.cuda.device_count()
    print(f"GPU count       : {device_count}")

    for i in range(device_count):
        properties = torch.cuda.get_device_properties(i)
        capability = torch.cuda.get_device_capability(i)

        print()
        print(f"[GPU {i}]")
        print(f"Name            : {torch.cuda.get_device_name(i)}")
        print(
            f"VRAM            : "
            f"{properties.total_memory / (1024 ** 3):.2f} GB"
        )
        print(
            f"Compute         : "
            f"{capability[0]}.{capability[1]}"
        )
        print(
            f"Multiprocessors : "
            f"{properties.multi_processor_count}"
        )

    device = torch.device("cuda:0")

    print()
    print("=" * 60)
    print("GPU COMPUTATION TEST")
    print("=" * 60)

    # 실제 GPU 연산 확인
    matrix_size = 4096

    print(
        f"Creating {matrix_size} x {matrix_size} "
        f"FP16 matrices on GPU..."
    )

    a = torch.randn(
        matrix_size,
        matrix_size,
        device=device,
        dtype=torch.float16,
    )

    b = torch.randn(
        matrix_size,
        matrix_size,
        device=device,
        dtype=torch.float16,
    )

    # 첫 실행은 CUDA 초기화 시간이 포함되므로 워밍업
    for _ in range(3):
        _ = torch.matmul(a, b)

    torch.cuda.synchronize()

    start = time.perf_counter()

    iterations = 10

    for _ in range(iterations):
        c = torch.matmul(a, b)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print()
    print(f"Result device   : {c.device}")
    print(f"Result shape    : {tuple(c.shape)}")
    print(f"Iterations      : {iterations}")
    print(f"Total time      : {elapsed:.4f} sec")
    print(
        f"Average time    : "
        f"{elapsed / iterations * 1000:.2f} ms"
    )

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)

    print()
    print("=" * 60)
    print("GPU MEMORY")
    print("=" * 60)

    print(
        f"Allocated       : "
        f"{allocated / (1024 ** 2):.2f} MB"
    )
    print(
        f"Reserved        : "
        f"{reserved / (1024 ** 2):.2f} MB"
    )
    print(
        f"Peak allocated  : "
        f"{peak / (1024 ** 2):.2f} MB"
    )

    print()
    print("=" * 60)
    print("GPU TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()