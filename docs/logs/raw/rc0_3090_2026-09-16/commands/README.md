# RC0 commands (read-only)

Date: 2026-09-16 UTC. No serving source was modified and no server was restarted.
The accepted TG0 server (scope `sglang-1789569284.scope`, pid 368271 scheduler,
port 30001) was left running throughout; all measurements are of that idle server.

```bash
# host memory
cat /proc/meminfo
SCOPE=/sys/fs/cgroup/user.slice/user-0.slice/user@0.service/app.slice/sglang-1789569284.scope
cat $SCOPE/memory.max $SCOPE/memory.current $SCOPE/memory.stat $SCOPE/memory.events

# scheduler process footprint, per-mapping attribution
grep -E 'VmRSS|RssAnon|RssFile|RssShmem|VmHWM' /proc/368271/status
cat /proc/368271/smaps_rollup
# 135 x '/dev/zero (deleted)' mappings ~= 25.25 GiB pinned host expert memory

# PLE mmap working set / residency
grep ple.f8 /proc/368271/smaps
cat /mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang/ple/ple.json

# GPU
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,clocks.sm,power.draw,temperature.gpu --format=csv

# model architecture facts
python3 -c 'import json;c=json.load(open("/mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang/config.json"));print(c["text_config"]["layer_types"])'

# serving revision identity
cd /root/sglang && git rev-parse HEAD && git diff | sha256sum
for f in python/sglang/srt/layers/moe/expert_elastic.py \
         python/sglang/srt/layers/moe/expert_gemv.py \
         python/sglang/srt/layers/moe/expert_stream.py \
         python/sglang/srt/layers/moe/row_arena.py \
         python/sglang/srt/mem_cache/int4_kv_pool.py \
         python/sglang/srt/mem_cache/int8_kv_pool.py \
         python/sglang/srt/mem_cache/tiered_kv_pool.py; do sha256sum "$f"; done

# accepted-boot pool accounting (quoted in the report)
grep -nE 'PLE: mmap|Mamba Cache|KvVmmArena|KV Cache is allocated|KV tiers|Memory pool end|available_gpu_mem' \
  docs/logs/raw/tg0_3090_2026-09-16/pp14-diag/server-accepted-boot4.log
```

Source inspection references are quoted inline in
[`../../rc0_feasibility_3090_2026-09-16.md`](../../rc0_feasibility_3090_2026-09-16.md).
