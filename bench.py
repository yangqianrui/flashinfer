"""
python3 bench.py   --device cuda:0   --remote-device cuda:1   --batch-size 64   --seq-len 8192   --page-size 16   --num-qo-heads 32   --num-kv-heads 8   --head-dim 128   --dtype bfloat16   --kv-layout NHD   --ratios 0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.7,0.9,1   --warmup 20   --iters 1000   --repeats 5 --output-dir result/ --page-device-pattern prefix
"""

import argparse
import ctypes
import ctypes.util
import statistics
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch

import flashinfer
import flashinfer.decode as flashinfer_decode


PAGE_DEVICE_PATTERNS = ["random", "request-head", "request-tail", "prefix", "suffix"]


def parse_ratios(value: str) -> List[float]:
    ratios = [float(x) for x in value.split(",") if x.strip()]
    for ratio in ratios:
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"remote ratios must be in [0, 1], got {ratio}")
    return ratios


def parse_page_device_patterns(value: str) -> List[str]:
    patterns = [x.strip() for x in value.split(",") if x.strip()]
    if not patterns:
        raise ValueError("at least one page-device pattern is required")
    for pattern in patterns:
        if pattern not in PAGE_DEVICE_PATTERNS:
            raise ValueError(
                f"unknown page-device pattern {pattern}; expected one of {PAGE_DEVICE_PATTERNS}"
            )
    return patterns


def enable_cuda_peer_access(src_device: int, peer_device: int) -> None:
    libcudart_name = ctypes.util.find_library("cudart") or "libcudart.so"
    cudart = ctypes.CDLL(libcudart_name)
    cuda_set_device = cudart.cudaSetDevice
    cuda_set_device.argtypes = [ctypes.c_int]
    cuda_set_device.restype = ctypes.c_int
    cuda_enable_peer = cudart.cudaDeviceEnablePeerAccess
    cuda_enable_peer.argtypes = [ctypes.c_int, ctypes.c_uint]
    cuda_enable_peer.restype = ctypes.c_int

    err = cuda_set_device(src_device)
    if err != 0:
        raise RuntimeError(f"cudaSetDevice({src_device}) failed with CUDA error {err}")
    err = cuda_enable_peer(peer_device, 0)
    if err not in (0, 704):
        raise RuntimeError(
            f"cudaDeviceEnablePeerAccess({peer_device}) from {src_device} failed with CUDA error {err}"
        )


def maybe_enable_peer_access(device: torch.device, remote_device: torch.device) -> None:
    if device == remote_device:
        return
    if device.type != "cuda" or remote_device.type != "cuda":
        raise ValueError("peer-memory benchmark requires CUDA devices")
    if not torch.cuda.can_device_access_peer(device.index, remote_device.index):
        raise RuntimeError(f"{device} cannot access peer memory on {remote_device}")

    enable_peer = getattr(torch.cuda.cudart(), "cudaDeviceEnablePeerAccess", None)
    if enable_peer is not None:
        try:
            with torch.cuda.device(device):
                enable_peer(remote_device.index, 0)
            return
        except RuntimeError as err:
            if "peer access is already enabled" in str(err).lower():
                return
    enable_cuda_peer_access(device.index, remote_device.index)


def make_page_device(
    total_num_pages: int,
    ratio: float,
    device: torch.device,
    seed: int,
    pattern: str,
    batch_size: int,
    num_pages_per_seq: int,
) -> torch.Tensor:
    num_remote_pages = int(round(total_num_pages * ratio))
    page_device = torch.zeros(total_num_pages, dtype=torch.int32, device=device)
    if num_remote_pages == 0:
        return page_device
    if num_remote_pages == total_num_pages:
        page_device.fill_(1)
        return page_device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if pattern == "random":
        perm = torch.randperm(total_num_pages, device=device, generator=generator)
        page_device[perm[:num_remote_pages]] = 1
        return page_device

    if batch_size * num_pages_per_seq != total_num_pages:
        raise ValueError("batch_size * num_pages_per_seq must equal total_num_pages")

    if pattern == "request-head":
        page_device[:num_remote_pages] = 1
        return page_device

    if pattern == "request-tail":
        page_device[-num_remote_pages:] = 1
        return page_device

    pages_by_request = page_device.view(batch_size, num_pages_per_seq)
    pages_per_request, extra_pages = divmod(num_remote_pages, batch_size)
    for batch_idx in range(batch_size):
        pages_this_request = pages_per_request + (1 if batch_idx < extra_pages else 0)
        if pages_this_request == 0:
            continue
        if pattern == "prefix":
            pages_by_request[batch_idx, :pages_this_request] = 1
        elif pattern == "suffix":
            pages_by_request[batch_idx, -pages_this_request:] = 1
        else:
            raise ValueError(f"unknown page-device pattern: {pattern}")
    return page_device


def bench_cuda_events(fn, warmup: int, iters: int, device: torch.device) -> float:
    with torch.cuda.device(device):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end) / iters


def write_plot(
    rows: List[dict],
    baseline_rows: List[dict],
    png_path: Path,
    patterns: List[str],
) -> None:
    plt.figure(figsize=(8, 4.8), dpi=160)
    for pattern_idx, pattern in enumerate(patterns):
        pattern_rows = [row for row in rows if row["page_device_pattern"] == pattern]
        ratios = [100.0 * float(row["remote_ratio"]) for row in pattern_rows]
        latencies = [float(row["latency_ms"]) for row in pattern_rows]
        plt.plot(ratios, latencies, marker="o", linewidth=2, label=f"local/remote ({pattern})")
        for x, y in zip(ratios, latencies):
            plt.annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 7 + pattern_idx * 8),
                ha="center",
                fontsize=8,
            )
    for baseline in baseline_rows:
        latency = float(baseline["latency_ms"])
        plt.axhline(latency, linestyle="--", linewidth=1.6, label=baseline["kind"])
    plt.xlabel("Remote page ratio (%)")
    plt.ylabel("Latency (ms)")
    plt.title("Batch decode local/remote page ratio")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--kv-layout", choices=["NHD", "HND"], default="NHD")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--remote-device", default=None)
    parser.add_argument("--ratios", default="0,0.1,0.25,0.5,0.75,0.9,1")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--page-device-pattern",
        "--page-device-patterns",
        default="random",
        help=(
            "Comma-separated page-device placement patterns. Supported values: "
            f"{','.join(PAGE_DEVICE_PATTERNS)}. random scatters remote pages uniformly; "
            "prefix places old-prefix pages on the remote device in every request and is a "
            "pessimistic non-random pattern for the current split-kv schedule; suffix does the "
            "same for recent pages; request-head/request-tail pack remote pages into whole "
            "requests to create request-level stragglers. Example: --page-device-pattern "
            "random,prefix"
        ),
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not benchmark the original wrapper.run implementation.",
    )
    parser.add_argument(
        "--skip-correctness-check",
        action="store_true",
        help="Do not compare ratio=0/1 outputs against the original wrapper.run implementation.",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    remote_device = torch.device(args.remote_device) if args.remote_device else device
    torch.cuda.set_device(device)
    ratios = parse_ratios(args.ratios)
    page_device_patterns = parse_page_device_patterns(args.page_device_pattern)
    maybe_enable_peer_access(device, remote_device)
    torch.manual_seed(args.seed)

    if args.num_qo_heads % args.num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")

    num_pages_per_seq = (args.seq_len + args.page_size - 1) // args.page_size
    total_num_pages = args.batch_size * num_pages_per_seq
    last_len = args.seq_len % args.page_size or args.page_size

    kv_indptr = (
        torch.arange(args.batch_size + 1, device=device, dtype=torch.int32) * num_pages_per_seq
    )
    kv_indices = torch.arange(total_num_pages, device=device, dtype=torch.int32)
    kv_last_page_len = torch.full((args.batch_size,), last_len, device=device, dtype=torch.int32)
    q = torch.randn(args.batch_size, args.num_qo_heads, args.head_dim, device=device, dtype=dtype)

    if args.kv_layout == "NHD":
        kv_shape = (total_num_pages, 2, args.page_size, args.num_kv_heads, args.head_dim)
    else:
        kv_shape = (total_num_pages, 2, args.num_kv_heads, args.page_size, args.head_dim)
    kv_local = torch.randn(kv_shape, device=device, dtype=dtype)
    kv_remote = torch.randn(kv_shape, device=remote_device, dtype=dtype)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer_decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout=args.kv_layout,
        use_cuda_graph=False,
        use_tensor_cores=False,
        backend="fa2",
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
        args.page_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
    )

    if not args.skip_correctness_check:
        page_device_local = torch.zeros(total_num_pages, dtype=torch.int32, device=device)
        ref_local = wrapper.run(q, kv_local, enable_pdl=False)
        out_local = wrapper.run_local_remote(
            q, kv_local, kv_remote, page_device_local, enable_pdl=False
        )
        torch.cuda.synchronize()
        print(f"check local max_diff={(ref_local - out_local).abs().max().item()}")

        if remote_device != device:
            page_device_remote = torch.ones(total_num_pages, dtype=torch.int32, device=device)
            ref_remote = wrapper.run(q, kv_remote, enable_pdl=False)
            out_remote = wrapper.run_local_remote(
                q, kv_local, kv_remote, page_device_remote, enable_pdl=False
            )
            torch.cuda.synchronize()
            print(f"check remote max_diff={(ref_remote - out_remote).abs().max().item()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "result.png"

    baseline_rows = []
    if not args.skip_baseline:
        local_latencies = [
            bench_cuda_events(
                lambda: wrapper.run(q, kv_local, enable_pdl=False),
                args.warmup,
                args.iters,
                device,
            )
            for _ in range(args.repeats)
        ]
        baseline_rows.append(
            {
                "kind": "original_local",
                "remote_ratio": 0.0,
                "latency_ms": statistics.median(local_latencies),
                "latency_repeats_ms": ";".join(f"{x:.6f}" for x in local_latencies),
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "page_size": args.page_size,
                "num_qo_heads": args.num_qo_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_dim": args.head_dim,
                "dtype": args.dtype,
                "kv_layout": args.kv_layout,
                "device": str(device),
                "remote_device": str(device),
            }
        )
        print(
            f"baseline=original_local, latency_ms={baseline_rows[-1]['latency_ms']:.6f}, "
            f"repeats={local_latencies}"
        )
        if remote_device != device:
            remote_latencies = [
                bench_cuda_events(
                    lambda: wrapper.run(q, kv_remote, enable_pdl=False),
                    args.warmup,
                    args.iters,
                    device,
                )
                for _ in range(args.repeats)
            ]
            baseline_rows.append(
                {
                    "kind": "original_remote",
                    "remote_ratio": 1.0,
                    "latency_ms": statistics.median(remote_latencies),
                    "latency_repeats_ms": ";".join(f"{x:.6f}" for x in remote_latencies),
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                    "page_size": args.page_size,
                    "num_qo_heads": args.num_qo_heads,
                    "num_kv_heads": args.num_kv_heads,
                    "head_dim": args.head_dim,
                    "dtype": args.dtype,
                    "kv_layout": args.kv_layout,
                    "device": str(device),
                    "remote_device": str(remote_device),
                }
            )
            print(
                f"baseline=original_remote, latency_ms={baseline_rows[-1]['latency_ms']:.6f}, "
                f"repeats={remote_latencies}"
            )

    rows = []
    for pattern_idx, pattern in enumerate(page_device_patterns):
        for ratio in ratios:
            page_device = make_page_device(
                total_num_pages,
                ratio,
                device,
                args.seed + pattern_idx * 100000 + int(ratio * 10000),
                pattern,
                args.batch_size,
                num_pages_per_seq,
            )
            remote_pages = int(page_device.sum().item())
            latencies = [
                bench_cuda_events(
                    lambda: wrapper.run_local_remote(
                        q, kv_local, kv_remote, page_device, enable_pdl=False
                    ),
                    args.warmup,
                    args.iters,
                    device,
                )
                for _ in range(args.repeats)
            ]
            latency_ms = statistics.median(latencies)
            row = {
                "kind": "local_remote",
                "remote_ratio": ratio,
                "latency_ms": latency_ms,
                "latency_repeats_ms": ";".join(f"{x:.6f}" for x in latencies),
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "page_size": args.page_size,
                "num_qo_heads": args.num_qo_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_dim": args.head_dim,
                "dtype": args.dtype,
                "kv_layout": args.kv_layout,
                "device": str(device),
                "remote_device": str(remote_device),
                "page_device_pattern": pattern,
                "remote_pages": remote_pages,
                "total_pages": total_num_pages,
            }
            rows.append(row)
            print(
                f"pattern={pattern}, remote_ratio={ratio:.2f}, "
                f"actual_remote_ratio={remote_pages / total_num_pages:.6f}, "
                f"remote_pages={remote_pages}/{total_num_pages}, latency_ms={latency_ms:.6f}, "
                f"repeats={latencies}"
            )
    write_plot(rows, baseline_rows, png_path, page_device_patterns)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
