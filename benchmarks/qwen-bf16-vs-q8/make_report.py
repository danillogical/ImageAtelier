"""Assemble the measured comparison after both benchmark processes have finished."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    runs = {name:json.loads((HERE/f'{name}.json').read_text()) for name in ('bf16','q8')}
    audit = json.loads((HERE/'quantization-audit.json').read_text())
    assert audit['passed'] and all(run['status']=='complete' for run in runs.values())
    a,b = runs['bf16'],runs['q8']
    assert a['raw_prompt']==b['raw_prompt'] and a['image_seed']==b['image_seed'] and a['steps']==b['steps']==40
    def stage(run):
        t=run['timings']; pe=run['enhancer']['result']; image=run['image']['result']
        return [pe['load_s'],t['pe_prefill_to_first_token_s'],t['pe_thinking_decode_s'],t['pe_answer_decode_s'],
                max(0,t['pe_job_before_cleanup_s']-pe['load_s']-sum(t[k] for k in ('pe_prefill_to_first_token_s','pe_thinking_decode_s','pe_answer_decode_s'))),
                pe['teardown_s'],max(0,run['pe_wall_s']-t['pe_job_before_cleanup_s']-pe['teardown_s']),
                run['pe_wall_s'],image['load_s'],t['prompt_encoding_s'],t['text_encoder_offload_s'],
                t['denoising_s'],t['vae_decode_and_pixel_conversion_s'],t['image_setup_png_and_other_overhead_s'],
                image['teardown_s'],max(0,run['image_wall_s']-t['image_job_before_cleanup_s']-image['teardown_s']),
                run['image_wall_s'],run['pipeline_wall_s']]
    names=['PE-T2I model load','PE prefill → first token','PE thinking-token decode','PE answer-token decode',
           'PE template/setup/other overhead','PE GPU synchronization and cleanup','PE worker startup, imports, idle guard and polling','**PE job wall time**',
           'Image model load (transformer + text/vision encoder + VAE)','Image prompt encoding',
           'Text-encoder offload','40 denoising steps','VAE decode + pixels','Image setup, model transition, PNG save and other overhead',
           'Image GPU synchronization and cleanup','Image idle guard and polling','**Image job wall time**','**Complete PE → image wall time**']
    timing='| Stage | BF16 (s) | Q8 (s) | BF16 ÷ Q8 |\n|---|---:|---:|---:|\n'
    for name,av,bv in zip(names,stage(a),stage(b)):
        timing+=f'| {name} | {av:.3f} | {bv:.3f} | {av/bv:.2f}× |\n' if bv>0 else f'| {name} | {av:.3f} | {bv:.3f} | — |\n'
    memory='| Metric | BF16 | Q8 |\n|---|---:|---:|\n'
    for label,fn in [('PE thinking tokens',lambda r:r['enhancer']['result']['thinking_tokens']),
                     ('PE answer tokens',lambda r:r['enhancer']['result']['answer_tokens']),
                     ('PE decode throughput (tokens/s)',lambda r:r['enhancer']['result']['tok_s']),
                     ('PE peak MLX allocation (GB)',lambda r:round(r['enhancer']['result']['peak_mem_bytes']/1e9,3)),
                     ('Image peak MLX allocation (GB)',lambda r:round(r['image']['result']['peak_mem_bytes']/1e9,3)),
                     ('Mean denoising step (s)',lambda r:round(r['timings']['denoising_s']/40,3)),
                     ('First denoising step (s)',lambda r:round(r['step_timings'][0]['seconds'],3)),
                     ('Mean steps 2–40 (s)',lambda r:round(sum(s['seconds'] for s in r['step_timings'][1:])/39,3))]:
        memory+=f'| {label} | {fn(a)} | {fn(b)} |\n'
    quant='| Component | Source weights (GB) | Q8 weights (GB) | Q8 packed matrices checked | Mean sampled relative RMS error |\n|---|---:|---:|---:|---:|\n'
    for name,part in audit['parts'].items():
        error=f"{part['mean_relative_rms_error_percent']:.3f}%" if part['quantized_layers'] else 'Full precision retained'
        quant+=f"| {name} | {part['bf16_weight_bytes']/1e9:.3f} | {part['q8_weight_bytes']/1e9:.3f} | {part['quantized_layers']} | {error} |\n"
    original_size=sum(p['bf16_weight_bytes'] for p in audit['parts'].values())/1e9
    converted_size=sum(p['q8_weight_bytes'] for p in audit['parts'].values())/1e9
    packed_transformer=audit['parts']['transformer']['q8_packed_weight_bytes']
    q8_decode=b['image']['result']['phases']['decode']['peak']
    memory_policy='| Suite | Peak model memory | With 6 GB overhead | Recommended RAM | Evidence |\n|---|---:|---:|---:|---|\n'
    estimates=[('BF16',a['image']['result']['peak_mem_bytes']/1e9,48,'Measured'),
               ('Q8',b['image']['result']['peak_mem_bytes']/1e9,32,'Measured')]
    estimates += [(f'Q{bits}',(q8_decode-packed_transformer*(8-bits)/8)/1e9,32,'Estimated') for bits in (6,4)]
    for label,peak,ram,evidence in estimates:
        memory_policy+=f'| {label} | {peak:.1f} GB | {peak+6:.1f} GB | {ram} GB | {evidence} |\n'
    versions=', '.join(f'{key} {value}' for key,value in a['versions'].items())
    analysis=(HERE/'visual-analysis.md').read_text()
    content=f'''# Qwen-Image-2.1: BF16 versus Q8

Measured on {a['started_at'][:10]} on an **{a['hardware']['chip']} MacBook Pro with 64 GiB unified memory**.

| Measured outcome | BF16 | Q8 |
|---|---:|---:|
| Complete PE-T2I → 40-step image | {a['pipeline_wall_s']:.2f} s | {b['pipeline_wall_s']:.2f} s |
| PE-T2I job | {a['pe_wall_s']:.2f} s | {b['pe_wall_s']:.2f} s |
| Image job | {a['image_wall_s']:.2f} s | {b['image_wall_s']:.2f} s |
| Image peak MLX allocation | {a['image']['result']['peak_mem_bytes']/1e9:.2f} GB | {b['image']['result']['peak_mem_bytes']/1e9:.2f} GB |

Q8 used less memory and finished the complete pipeline **{(1-b['pipeline_wall_s']/a['pipeline_wall_s'])*100:.1f}% sooner** in this sample. BF16 preserved the requested subject placement and three pumpkins; Q8's enhancer changed the placement, and its image shows only two pumpkins. Both checkpoints passed the quantization audit and completed all 40 steps.

## Test setup

One complete end-to-end run per suite: its own PE-T2I enhancer rewrites the same source prompt, then its own image model generates from that rewrite. **1024 × 1024, 40 steps, seed {a['image_seed']}**, no input images, no LoRA, no art-style suffix, opaque output. The app's standard Euler linear scheduler and guidance 1.0 are used. Both suites use the same memory settings: text-encoder offload enabled and VAE tiling disabled.

PE settings: English, thinking enabled, maximum 8,192 new tokens, temperature 1.0, top-p 0.95, top-k 20. Sampling is seeded **after loading** with {a['pe_seed']} for both runs. Rewrites may still differ because model precision changes token probabilities. This compares complete pipelines; it does **not** isolate the image transformer's quantization from prompt-enhancer changes.

Each suite runs in a separate fresh Python process, with no other ImageAtelier server running, using the real ImageAtelier engine and downloaded local models. No downloads or conversion time are included. This is a single-run comparison, not a statistical benchmark. Filesystem cache, kernel compilation, power/thermal state, and background desktop activity are not fully controlled. BF16 ran first. An earlier BF16 PE initialization was aborted while fixing seed placement; it is excluded from all measurements and may have warmed the filesystem cache. GPU stage boundaries are synchronized. Instrumentation and file persistence add a small amount of overhead.

Software: {versions}.

### Shared source prompt

> {a['raw_prompt']}

## Generated images

### BF16

![Image generated with the BF16 suite at 40 steps]({(HERE/'bf16.png').relative_to(ROOT).as_posix()})

### Q8

![Image generated with the Q8 suite at 40 steps]({(HERE/'q8.png').relative_to(ROOT).as_posix()})

## Stage timings

All values are seconds. A ratio above 1 means Q8 was faster. Bold rows are independently measured wall-clock totals; they include scheduling/polling overhead and are not additional stages to sum again. Model loading includes materialization of the respective checkpoint weights. The image-load row also loads the text/vision encoder and VAE. The separate image overhead row includes PE unloading during the model transition, runtime-format auditing, PNG serialization, and Python bookkeeping.

{timing}
PE thinking decode starts after the first emitted token and ends when `</think>` appears; prefill includes producing the first token. PE answer decode includes the final JSON answer. VAE decode includes latent unpacking and conversion to image pixels. Peak allocation means MLX's recorded peak, not total macOS RAM usage.

{memory}
## Quantization verification

The five weight components total **{original_size:.3f} GB** in the original suite and **{converted_size:.3f} GB** in the converted suite, a **{(1-converted_size/original_size)*100:.1f}%** reduction (decimal GB; processor/config files excluded).

{quant}
Checkpoint checks covered all packed matrices: shape, unsigned 32-bit packing containing four 8-bit values per word, scale/bias shapes, and group size 64. Every source vision tensor was accounted for, including the convolution layout change required by MLX. The numerical audit dequantized the first and middle rows of every packed matrix and compared those values with the corresponding BF16 source rows. This is a sampled numerical check, not a full-tensor equality test; Q8 is lossy by design.

The image transformer and eligible text/vision encoder layers are Q8; **90 vision layers** in the image encoder are quantized. PE-T2I and PE-I2I have Q8 language-model layers, while **their vision towers remain BF16 under the MLX-VLM converter's policy**. The reference convolutional **VAE remains F32 in both suites**. Thus “Q8 suite” does not mean every tensor uses eight bits. BF16 runtime models were checked for the absence of quantized modules; Q8 runtime image/PE-T2I models were checked for 8-bit modules with group size 64. PE-I2I was audited on disk but was not used for this text-to-image benchmark.

## RAM budget and Q6/Q4 estimates

Using a conservative **6 GB allowance for macOS and other apps**, these are the recommendations for this workload, with text-encoder offload enabled and **VAE tiling disabled**:

{memory_policy}
The Q8 decoding peak is the limiting stage. Its transformer contains {packed_transformer/1e9:.3f} GB of packed 8-bit matrix data. Assuming unchanged activations, full-precision VAE, and group-64 scale/bias storage, Q6 saves another 25% of those packed bytes and Q4 saves 50%. Their estimated peaks therefore subtract those savings from the measured Q8 decoding peak; they do not scale all memory linearly with bit depth. The text encoder is already offloaded by this stage. Neither lower-bit peak has been measured here.

The app recommends BF16 at **48 GB RAM or more**, Q8 at **32–47 GB**, and no suite below 32 GB under this headroom policy. Q6 and Q4 remain available as manual choices. For a conservative 24 GB target, the model budget would be 18 GB; Q6 would need about 3.2 GB of additional peak savings and Q4 about 1.4 GB. VAE tiling is a candidate for achieving those savings, but requires a separate measurement before a 24 GB recommendation. Measured allocations use decimal GB; physical RAM thresholds use the Mac's GiB capacity tiers, with conservative rounding for the recommendation.

## Analysis

{analysis}

## Full enhanced prompts

### BF16 PE-T2I

{a['enhancer']['result']['rewritten_prompt']}

### Q8 PE-T2I

{b['enhancer']['result']['rewritten_prompt']}

## Reproducibility and raw results

- [BF16 measurements](benchmarks/qwen-bf16-vs-q8/bf16.json)
- [Q8 measurements](benchmarks/qwen-bf16-vs-q8/q8.json)
- [Checkpoint numerical audit](benchmarks/qwen-bf16-vs-q8/quantization-audit.json)
- [Benchmark runner](benchmarks/qwen-bf16-vs-q8/run_benchmark.py)
- [Audit script](benchmarks/qwen-bf16-vs-q8/audit_quantization.py)

From the repository root, run the audit and then each suite sequentially (these commands replace the corresponding benchmark outputs):

```sh
HF_HUB_OFFLINE=1 .venv/bin/python benchmarks/qwen-bf16-vs-q8/audit_quantization.py
caffeinate -i .venv/bin/python benchmarks/qwen-bf16-vs-q8/run_benchmark.py bf16
caffeinate -i .venv/bin/python benchmarks/qwen-bf16-vs-q8/run_benchmark.py q8
.venv/bin/python benchmarks/qwen-bf16-vs-q8/make_report.py
```

Pinned repository revisions are recorded in the numerical audit. Logs and per-step timings are stored alongside the images. Enhanced prompts are also saved separately as plain text.
'''
    (ROOT/'qwen-bf16-vs-q8.md').write_text(content)
    print(ROOT/'qwen-bf16-vs-q8.md')

if __name__=='__main__':
    main()
