# pre-PP15 MoE config arm (non-promotion validation)

Accepted stack with `SGLANG_MOE_CONFIG_DIR` pointed at the pre-PP15 configs
(`adc57b8`): the `M=4565` entries are absent from the E=384/E=432 buckets, so the
canonical/4,608 chunk resolves to the `M=1024` entry instead. Everything else,
including PP14 prefetch, is identical to `accepted/`.

`oracle-check.txt` is one `tools/long_logprob_oracle.py check accepted-run1` run.
`server.json` is the live server snapshot; `config-keys.txt` shows the reduced
key set. JSON logprobs were not retained for this arm (the check text is the raw
evidence).
