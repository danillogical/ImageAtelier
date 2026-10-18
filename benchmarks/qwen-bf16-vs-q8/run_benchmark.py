"""Run one suite in a fresh process through ImageAtelier's real engine, with synchronized stage timings."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TQDM_DISABLE'] = '1'

from imageatelier import engine as engine_module, models, suites

PROMPT = ('An editorial photograph of a small neighborhood bakery at blue hour after rain. '
          'A woman in a mustard-yellow coat stands on the left holding a red umbrella; '
          'a friendly golden retriever sits on the right beside exactly three orange pumpkins. '
          'Warm light spills through the window. Above the door, a sign clearly reads "NIGHT OWL BAKERY". '
          'Wet cobblestones reflect the amber lights. Natural proportions, detailed textures, '
          'cinematic yet realistic lighting.')
SEED = 20261016
PE_SEED = 314159
OUT = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


class TimedEngine(engine_module.Engine):
    def __init__(self, suite):
        quant = models.SUITE_QUANTS[suite]
        self.timings = {}
        self.runtime = {}
        self.steps = []
        self.mark = None
        self.step_mark = None
        self.phase_active = False
        super().__init__(settings_provider=lambda: {
            'resolution': 1024, 'low_memory': True, 'vae_tiling': False,
            'image_model': f'official-{quant}', 'text_encoder': f'official-{quant}',
            'pe_quant': quant, 'pe_thinking': True, 'pe_max_new_tokens': 8192,
            'pe_language': 'english',
        }, cache_gb=2)
        self.suite = suite

    def _ensure(self, job, key):
        result = super()._ensure(job, key)
        engine_module._synchronize_all()
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        model = self._mm.model
        components = {name: getattr(model, name) for name in ('transformer', 'text_encoder', 'vae')} if key[0] == 'image' else {key[0]: model}
        audit = {}
        for name, component in components.items():
            quantized = []
            dtype_counts = {}
            for path, layer in component.named_modules():
                if isinstance(layer, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
                    quantized.append({'path': path, 'bits': layer.bits, 'group_size': layer.group_size})
            for path, array in tree_flatten(component.parameters()):
                dtype_counts[str(array.dtype)] = dtype_counts.get(str(array.dtype), 0) + 1
            if self.suite == 'bf16':
                assert not quantized, (name, 'BF16 model unexpectedly quantized')
                assert not any('uint' in dtype for dtype in dtype_counts), (name, dtype_counts)
            elif name != 'vae':
                assert quantized and all(layer['bits'] == 8 and layer['group_size'] == 64 for layer in quantized), (name, quantized)
            if name == 'vae':
                assert not quantized, 'The reference VAE must retain full precision'
            audit[name] = {'quantized_layers': len(quantized), 'bits': sorted(set(x['bits'] for x in quantized)),
                           'groups': sorted(set(x['group_size'] for x in quantized)), 'parameter_dtypes': dtype_counts,
                           'vision_quantized_layers': sum(any(s in x['path'] for s in ('visual.', 'vision_tower.')) for x in quantized)}
        self.runtime[key[0]] = audit
        print('RUNTIME_QUANTIZATION', key, json.dumps(audit), flush=True)
        return result

    def _run(self, job):
        self.phase_active = True
        start = time.perf_counter()
        try:
            result = super()._run(job)
            engine_module._synchronize_all()
            self.timings['image_job_before_cleanup_s'] = time.perf_counter() - start
            if result[0] == engine_module.DONE:
                accounted = result[1]['result']['load_s'] + sum(self.timings.get(k, 0) for k in ('prompt_encoding_s', 'text_encoder_offload_s', 'denoising_s', 'vae_decode_and_pixel_conversion_s'))
                self.timings['image_setup_png_and_other_overhead_s'] = max(0, self.timings['image_job_before_cleanup_s'] - accounted)
            return result
        finally:
            self.phase_active = False

    def _end_phase(self, name):
        engine_module._synchronize_all()
        stamp = time.perf_counter()
        if name == 'load':
            self.mark = self.step_mark = stamp
        elif name == 'encode':
            self.timings['prompt_encoding_s'] = stamp - self.mark
            self.mark = self.step_mark = stamp
        elif name == 'denoise':
            self.timings['denoising_s'] = stamp - self.mark
            self.mark = stamp
        elif name == 'decode':
            self.timings['vae_decode_and_pixel_conversion_s'] = stamp - self.mark
            self.mark = stamp
        print('IMAGE_PHASE', name, json.dumps(self.timings), flush=True)
        return super()._end_phase(name)

    def _maybe_offload_text_encoder(self):
        did_offload = self._offload_te and self._mm.model is not None and self._mm.model.text_encoder is not None
        start = time.perf_counter()
        result = super()._maybe_offload_text_encoder()
        if did_offload and self.phase_active:
            engine_module._synchronize_all()
            self.timings['text_encoder_offload_s'] = time.perf_counter() - start
            self.mark = self.step_mark = time.perf_counter()
        return result

    def _on_step(self, step, total):
        stamp = time.perf_counter()
        self.steps.append({'step': step, 'seconds': stamp - self.step_mark})
        self.step_mark = stamp
        print('STEP', step, total, self.steps[-1]['seconds'], flush=True)
        return super()._on_step(step, total)

    def _run_enhance(self, job):
        import mlx.core as mx
        import mlx_vlm
        mx.random.seed(PE_SEED)
        original = mlx_vlm.stream_generate
        timestamps = {}
        def stream(*args, **kwargs):
            # Model construction can consume RNG values; seed sampling after loading.
            mx.random.seed(PE_SEED)
            started = time.perf_counter()
            timestamps['stream_start'] = started
            text = ''
            count = 0
            for token in original(*args, **kwargs):
                stamp = time.perf_counter()
                if count == 0:
                    timestamps['first_token'] = stamp
                count += 1
                text += token.text
                if '</think>' in text and 'thinking_end' not in timestamps:
                    timestamps['thinking_end'] = stamp
                yield token
            timestamps['stream_end'] = time.perf_counter()
        mlx_vlm.stream_generate = stream
        start = time.perf_counter()
        try:
            result = super()._run_enhance(job)
            engine_module._synchronize_all()
            self.timings['pe_job_before_cleanup_s'] = time.perf_counter() - start
            first = timestamps.get('first_token')
            if first:
                end = timestamps['stream_end']
                split = timestamps.get('thinking_end', end)
                self.timings['pe_prefill_to_first_token_s'] = first - timestamps['stream_start']
                self.timings['pe_thinking_decode_s'] = max(0, split - first)
                self.timings['pe_answer_decode_s'] = end - split
            return result
        finally:
            mlx_vlm.stream_generate = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=['bf16', 'q8'])
    args = parser.parse_args()
    suite = args.suite
    quant = models.SUITE_QUANTS[suite]
    assert suites.parts(suite), f'{suite} is not ready'
    output = OUT / f'{suite}.json'
    report = {'suite': suite, 'raw_prompt': PROMPT, 'image_seed': SEED, 'pe_seed': PE_SEED, 'steps': 40,
              'resolution': 1024, 'thinking': True, 'max_new_tokens': 8192, 'temperature': 1.0, 'top_p': .95, 'top_k': 20,
              'low_memory': True, 'vae_tiling': False, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'hardware': {'chip': subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                           'ram_bytes': suites.physical_memory()},
              'versions': {name:importlib.metadata.version(name) for name in ('mlx','mlx-vlm','mflux','huggingface_hub')}}
    save(output, report)
    logging.basicConfig(level=logging.INFO)
    engine = TimedEngine(suite)
    last_tokens = 0
    def event(snapshot):
        nonlocal last_tokens
        if snapshot.get('type') in ('state','model'):
            print('EVENT', snapshot.get('type'), snapshot.get('state'), snapshot.get('kind'), snapshot.get('phase'), flush=True)
        elif snapshot.get('kind') == 'enhance' and snapshot.get('tokens', 0) >= last_tokens + 100:
            last_tokens = snapshot['tokens']
            print('PE_PROGRESS', snapshot.get('phase'), last_tokens, flush=True)
    engine.on_event(event)
    engine.start()
    pipeline_start = time.perf_counter()
    def wait(jid):
        while True:
            snap = engine.get(jid)
            if snap['state'] in engine_module.FINAL_STATES:
                if snap['state'] != engine_module.DONE:
                    raise RuntimeError(json.dumps(snap, default=str))
                return snap
            time.sleep(.1)
    try:
        start = time.perf_counter()
        jid = engine.submit({'kind':'enhance','mode':'t2i','prompt':PROMPT,'quant':quant,'thinking':True,
                             'language':'english','max_new_tokens':8192})
        pe = wait(jid)
        report['pe_wall_s'] = time.perf_counter() - start
        report['enhancer'] = pe
        report['timings'] = dict(engine.timings)
        report['runtime_quantization'] = dict(engine.runtime)
        save(output, report)
        if pe['result']['fell_back']:
            raise RuntimeError('Enhancer failed to produce a usable rewrite')
        enhanced = pe['result']['rewritten_prompt']
        (OUT/f'{suite}-enhanced-prompt.txt').write_text(enhanced+'\n')
        print('ENHANCED_PROMPT', enhanced, flush=True)
        start = time.perf_counter()
        jid = engine.submit({'mode':'t2i','prompt':enhanced,'seed':SEED,'steps':40,
                             'image_model':f'official-{quant}','text_encoder':f'official-{quant}',
                             'output_path':str(OUT/f'{suite}.png')})
        image = wait(jid)
        report['image_wall_s'] = time.perf_counter() - start
        report['pipeline_wall_s'] = time.perf_counter() - pipeline_start
        report['image'] = image
        report['timings'] = dict(engine.timings)
        report['step_timings'] = engine.steps
        report['runtime_quantization'] = engine.runtime
        report['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        report['status'] = 'complete'
        save(output, report)
        print('COMPLETE', suite, 'seconds', report['pipeline_wall_s'], flush=True)
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        report['timings'] = engine.timings
        save(output, report)
        raise
    finally:
        engine.stop()

if __name__ == '__main__':
    main()
