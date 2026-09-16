# Step 2 — independent checkout and venv

Date: 2026-09-16. Execution-plan step 2 of
[`docs/SGLANG_MAIN_MIGRATION_PLAN.md`](../../../SGLANG_MAIN_MIGRATION_PLAN.md).
The control (`/root/sglang`, `/root/quant/venv-sglang`, the rollback launchers
and `/root/quant/serve-3090.sh`) is untouched.

## Checkout

| Item | Value |
|---|---|
| Path | `/root/quant/sglang-main` |
| Origin | `https://github.com/sgl-project/sglang.git` |
| Pinned commit | `b02e16a895add01a0cfe24bb74922de92ab4d895` (matches `/root/sglang` `origin/main`) |
| Model-support commit | `52fecfdf0908dca24f4c6799ff5967125cc4110e` present |
| Method | `git clone --local --no-checkout /root/sglang` then detached checkout; 508 MB, hardlinked objects |

## Venv

| Item | Value |
|---|---|
| Path | `/root/quant/venv-sglang-main` (Python 3.12.3) |
| sglang | editable, `0.5.20.dev788+gb02e16a89`, `/root/quant/sglang-main/python` |
| Build log | `venv-main-build.log` |
| Installer | mirrors `/root/quant/build_venv2.sh` (`SGLANG_BUILD_RUST_EXTS=none`, nvcc/cuda-runtime pinned, `flash-attn==2.8.3.post1`, `FLASH_ATTN_CUDA_ARCHS=86`, `MAX_JOBS=4`) |
| torch arch list | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120` (includes SM86) |
| Full freeze | `venv-main-pip-freeze.txt` (206 packages) |

Separate caches/control files are required when serving from this venv: use a
distinct `SGLANG_CACHE_DIR` and a distinct `SGLANG_MOE_ELASTIC_CTL`/status path,
and do not point the main server at `/root/quant/logs/server.log`.

## Dependency delta

Full table in `dependency-delta.txt`. Material differences:

| Package | Control | Main | Note |
|---|---|---|---|
| `flashinfer-python` | 0.6.17 | **0.6.18** | main's pyproject pin; must be validated on SM86 |
| `sglang-kernel` | 0.4.6.post1 | 0.4.7 | wheel bump pulled by main |
| `tilelang` | 0.1.11 | 0.1.12 | minor bump |
| `sgl-deep-gemm` | 0.1.5.post3 | 0.2.0 | minor bump |
| `sgl-deep-ep` | 0.1.0 | 0.1.2 | minor bump |
| `nvshmem4py-cu13` | (absent) | 0.3.1 | new transitive dep |
| `Cython` | (absent) | 3.3.0 | new build/runtime dep |
| `pyelftools` | 0.33 | (absent) | dropped |
| `anthropic` / `urllib3` / `yarl` / `propcache` / `modelscope*` / `humming-kernels` / `kernels-data` | — | — | unrelated minor bumps |

Toolchain pinned to the control: `nvidia-cuda-nvcc 13.3.73`,
`nvidia-cuda-runtime 13.3.29`, `nvidia-cuda-crt 13.3.73`, `nvidia-nvvm 13.3.73`,
`nvidia-cuda-nvdisasm 13.3.73`. Main's editable install had first brought
`nvidia-cuda-nvcc 13.4.59` (and 13.4.x crt/nvvm); those were downgraded to match
the control so the JIT toolchain is identical. This is the "validate the
supported SM86 toolchain separately" item: the venv's `nvcc` reports release
13.3, V13.3.73, and `torch.cuda.get_arch_list()` contains `sm_86`.

## Server status note

The promoted control server received an external `SIGTERM` at 20:20:36 and shut
down cleanly (its last log lines are the normal drain). This happened before any
venv command was issued (first command 20:21:07). Nothing in this step starts or
stops a server; the rollback launcher `/root/quant/serve-3090.sh` remains
available for a maintenance-window run.

## Follow-up: cu13 JIT link layout (2026-09-16 20:54)

First candidate bring-up crashed in the JIT toolchain before any model code:
`ninja`/`ld: cannot find -lcudart` while linking
`sgl_kernel_jit_gptq_marlin_repack`, because the main venv's
`nvidia/cu13` lacked two layout entries the control venv has:

```
ln -sf lib                $VENV/lib/python3.12/site-packages/nvidia/cu13/lib64
ln -sf libcudart.so.13    $VENV/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so
```

After the fix `_jit_gptq_marlin_repack_module()` builds successfully. This is a
venv layout fix, not a source-port change. The same two entries must be part
of any rebuilt main venv.
