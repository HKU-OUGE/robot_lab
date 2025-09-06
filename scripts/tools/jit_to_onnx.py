# jit2onnx_min.py
import argparse, os, torch

def main():
    p = argparse.ArgumentParser("JIT (.pt) → ONNX (minimal, no Isaac)")
    p.add_argument("--jit", required=True, help="Path to TorchScript .pt")
    p.add_argument("--onnx_out", default=None, help="Output .onnx path (default: alongside .pt)")
    p.add_argument("--obs_dim", type=int, required=True, help="Observation feature dim, e.g., 42")
    p.add_argument("--act_dim", type=int, default=None, help="(Optional) expected action dim, e.g., 16 (sanity check)")
    p.add_argument("--device", default="cuda:0", help="cuda:0 | cpu | cuda:1 ...")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset (>=13)")
    p.add_argument("--batch", type=int, default=1, help="Dummy batch for export (batch is dynamic anyway)")
    args = p.parse_args()

    if args.onnx_out is None:
        base = os.path.dirname(os.path.abspath(args.jit))
        args.onnx_out = os.path.join(base, "policy.from_jit.onnx")

    # pick device safely
    want_cuda = args.device.startswith("cuda")
    if want_cuda and not torch.cuda.is_available():
        print("[WARN] CUDA not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # load scripted model
    policy = torch.jit.load(args.jit, map_location=device)
    policy.eval()

    # build dummy input
    x = torch.zeros(max(1, args.batch), int(args.obs_dim), device=device, dtype=torch.float32)

    # dry-run & (optional) sanity check
    with torch.inference_mode():
        y = policy(x)
    if not torch.is_tensor(y):
        raise RuntimeError(f"JIT returned {type(y)}; this minimal exporter expects a Tensor.")
    if y.ndim != 2:
        raise RuntimeError(f"Expected model output [B, A]; got shape={tuple(y.shape)}")
    if args.act_dim is not None and y.shape[1] != int(args.act_dim):
        raise RuntimeError(f"Action dim mismatch: got {y.shape[1]} vs expected {args.act_dim}")

    # export
    os.makedirs(os.path.dirname(os.path.abspath(args.onnx_out)), exist_ok=True)
    dynamic_axes = {"observations": {0: "batch"}, "actions": {0: "batch"}}

    torch.onnx.export(
        policy,
        x,
        args.onnx_out,
        input_names=["obs"],
        output_names=["actions"],
        opset_version=int(args.opset),
        do_constant_folding=True,
        export_params=True,
    )
    print(f"[OK] ONNX exported to: {args.onnx_out}")

if __name__ == "__main__":
    main()
