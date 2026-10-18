### Speed and memory

The Q8 pipeline finished in **128.38 s (2 min 8 s)** versus **161.65 s (2 min 42 s)** for BF16: **33.27 seconds saved, 20.6% less elapsed time, or 1.26× throughput** for this one complete run. Most of the gain came from PE-T2I: its wall time fell from 71.12 to 40.34 seconds. The measured image-job saving was much smaller, 90.53 to 88.04 seconds, or 2.8%.

PE decoding ran at **55.7 tokens/s for Q8 versus 32.1 tokens/s for BF16**. Q8 generated 965 thinking tokens and 988 answer tokens; BF16 generated 919 thinking tokens and 1,207 answer tokens. Q8 therefore both decoded faster and wrote a shorter answer. The shorter answer contributes to its elapsed-time advantage; the 43.3% PE wall-time reduction is not a comparison at identical output length. The shared 70-word input expanded to 951 words under BF16 and 763 under Q8.

Q8 did **not** accelerate every stage. Its 40 denoising steps took **82.90 s versus 79.26 s**, about **4.6% longer**. Its smaller image checkpoint loaded faster (1.73 versus 4.80 seconds), and its shorter enhanced prompt encoded faster (0.46 versus 2.57 seconds), largely offsetting that denoising slowdown. These timings establish what happened on this M5 Max in these runs; they do not isolate kernel performance from prompt length, compilation, cache, or thermal effects. The VAE also timed differently despite retaining identical F32 precision, illustrating why a single-run stage difference should not automatically be credited to quantization.

Image-job peak MLX allocation fell from **32.43 GB to 22.94 GB**, a **29.3% reduction**. BF16 peaked during prompt encoding; Q8 peaked during VAE decoding. PE peak allocation fell from **20.63 GB to 12.12 GB**, a **41.3% reduction**. The shared full-precision VAE and activation buffers help explain why image peak memory did not fall as much as checkpoint size. Offloading the text encoder after encoding was enabled for both runs.

### Prompt fidelity and visible results

| Requested property | BF16 result | Q8 result |
|---|---|---|
| “NIGHT OWL BAKERY” signage | All three words are clearly legible, arranged across two lines. | All three words are legible on a single sign; spacing is tighter. |
| Woman on the left; dog on the right | Preserved: woman left of center, retriever to her right. | Woman and dog are both on the right; the requested placement of the woman was lost. |
| Mustard coat and red umbrella | Both present. | Both present; a knit beanie was also added by the enhancer. |
| Exactly three orange pumpkins | Three distinct pumpkins are visible beside the dog. | Only two distinct pumpkins are visible in front of the dog. |
| Rainy blue hour, amber interior, wet cobblestones | Coherent cool street/warm interior contrast, wet paving, detailed reflections. | Coherent cool dusk/warm interior contrast, strong golden paving reflections and window moisture. |
| Photographic appearance | Convincing cloth, wood, foliage and fur textures; a wider street-side composition. | Convincing raincoat, fur, rain and storefront textures; a tighter composition with a larger dog and stronger bokeh. |

Both outputs are coherent, detailed, photo-like images, with plausible umbrella handling, shop lighting and wet-surface reflections. I do not see a gross corruption failure such as noise, broken decoding, or widespread malformed objects. For this particular source prompt, **BF16 was more faithful to the layout and count constraints**. That is a constraint-following observation from one scene, not a general quality ranking across prompts.

The position error is already explicit in the **Q8 enhanced prompt**: “On the right side of the storefront, an adult woman stands…”. Its image follows that changed instruction. The BF16 enhancer instead says “slightly left of center,” which preserves the source placement. The Q8 pumpkin omission is different: its enhancer still explicitly requests exactly three, so the final image fails a constraint that remained in its input. These are two distinct failure locations in the pipeline.

Both enhancers also invented secondary text and decorative details. BF16 added “FRESH BAKED”, “BY LIGHT” and the odd menu item “LATE SADDLE”; the generated image reproduces those additions. Q8 added “Apple Cake” and a door label “B”, which also appear in its image. Extra signs, planters, clothing details and architectural choices make the scenes richer, but weaken strict fidelity to the original short prompt. When exact composition or wording matters, reviewing the enhanced prompt before generating would catch some of these changes.

### What this comparison supports

**Q8 worked correctly as a quantized model and delivered substantial memory savings plus a faster complete PE-to-image pipeline. BF16 preserved more of this prompt's requested constraints.** The measured Q8 denoising slowdown means reduced checkpoint size should not be treated as a guarantee of faster image sampling on this hardware.

Quantized weights change token probabilities; even with the same PE sampling seed, the enhancers produced different text. The generated images therefore reflect both enhancer divergence and image-model precision. A clean test of the image model alone would feed both image suites one identical, fixed enhanced prompt and repeat several seeds/runs. The current images answer the requested complete-pipeline comparison. No image-reference/I2I generation was performed, and the numerical weight errors reported above are not perceptual image-quality scores.
