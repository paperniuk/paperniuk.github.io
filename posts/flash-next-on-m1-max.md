---
title: Qwen3.8-Flash-Next on a 2021 M1 Max: 44 tok/s, and 35 tok/s with 400K tokens in the context
date: 2026-10-04
slug: flash-next-on-m1-max
summary: A 126B mixture-of-experts model on a 64 GB laptop, with kernels written for the M1, and how little the speed falls as the context grows.
tags: Metal, ds4, MoE, benchmarks
reddit: 
---

A few weeks ago I posted my M1 port of Splash. The most common request in the comments and DMs was the same: bring the M1 optimizations to Qwen3.8-Flash-Next, put it in Splash. I started looking at it and realized that adding a new architecture to Splash from scratch would take me forever: the hybrid recurrent layers, the indexed sparse attention, the n-gram tables, MTP. So I took an engine that already runs this model, antirez's ds4 (DwarfStar), and spent the time on making it fast on the M1 instead.

So this time it is a different engine and a bigger model: **Qwen3.8-Flash-Next** in ISTA-DASLab's small GSQ-RCO quants, running on my fork of ds4 with Metal kernels written for the M1. Same laptop: M1 Max, 32 GPU cores, 64 GB.

The number I care about this time is not the peak, it is how little it falls. Long agent sessions are where local models usually die: the context grows, decode drops to single digits, the Mac heats up and throttles. So I ran it for a night and measured exactly that.

TL;DR, M1 Max 64 GB, MTP speculative decoding on:

- **One chat grown to its full context, reasoning xhigh:** Q2_0 (35 GiB of weights) decodes at **44 tok/s** at 4K, **38.8** at 256K and **35.4 at 398K**. Prefill goes from 328 to 292 tok/s over the same range.
- **Five prompts on a loop for 5 minutes, reasoning xhigh:** **43.2 tok/s** in the first quarter of the run and **43.1** in the last. GPU settles at 73 to 75 C, the clock does not move, 0.70 J per token.
- **Real OpenCode sessions, reasoning medium:** 162 requests, context up to 371K, 143K generated tokens. Median decode per turn: 44 tok/s under 64K, 42 at 64 to 128K, 38 at 128 to 256K.
- **IQ3_XXS (44 GiB, better answers):** 35.2 tok/s at 4K, 31.7 at 259K, same flat shape.
- **Against my Splash port of Qwen3.8-27B, same OpenCode task:** at 128 to 192K the 27B decodes at 14 tok/s, Flash-Next Q2_0 at 38; prefill is 4 to 7x faster.
- **For scale: a DGX Spark** with the popular vLLM NVFP4 + MTP recipe decodes this model at 36.8 tok/s on prose and 45.8 on code, single stream. A 2021 laptop is now in the same range for decode (the Spark still prefills 5x faster).
- **Bonus: DeepSeek V4 Flash (81 GiB) on the same 64 GB Mac**, streamed from the SSD: 12 tok/s at 4K, 9 at 127K.

![One chat growing to its full context](img/flash-next-on-m1-max/4_all_models_depth.png)

## How it works

Where it started: stock ds4 could not open ISTA's files at all. With the n-gram patch that it needs just to load the model, it decoded at 21 to 23 tok/s on the M1, 17.7 at 262K. llama.cpp gave 27 tok/s short and 13 at 64K. Here is what changed, in the order the problems came up.

**1. Making the files load at all.**

- *The quant types.* ISTA's GSQ-RCO files are mixed precision. The experts are Q2_0 (or IQ3_XXS plus IQ2_XS/IQ2_S/IQ3_S in the bigger file), and the rest is Q5_0, IQ4_NL/XS, Q3_K to Q6_K and BF16. ds4 had no Metal path for most of these, so each one got a decoder, and each is checked bit-exact against the CPU dequantizer (and llama.cpp for the IQ types).
- *The n-gram table.* Qwen3.8-Flash-Next has a per-layer n-gram embedding table. In BF16 it is 95 GiB, more than the whole Mac. ISTA ships it as a separate 28.8 GB IQ4_NL shard. The fork never loads that table into memory: each token needs only 16 rows of it, so those rows are read from the SSD as the token goes through. ds4 also had the IQ4_NL block size wrong, which is fixed too.
- *The MTP head.* ISTA's release has no MTP block, so speculative decoding was off the table. The fork borrows it from a second GGUF (I put a 1.5 GB file with only that block on Hugging Face; ggml-org's MTP Q8_0 file works too) and grafts it on as layer 48 at load time.

**2. Decode kernels for the M1.** Decode is memory-bound: every token reads about 4 GB of weights through about 1000 GPU dispatches. So the work went into matvec kernels for the M1 (Apple7) GPU, one per quant type. A few things that mattered on this GPU:

- Q2_0 decodes through half2 arithmetic with a magic-number trick instead of converting bytes to floats (uchar4 to float4 was much slower).
- Reading through typed block pointers instead of `char*` was worth 5 to 15% per kernel.
- For MTP, every type also has a 2-token and a 3-token variant that decodes each weight block once and uses it for all tokens. Verifying a draft then costs about one weight read instead of two or three. Four tokens run as two pairs, because a 4-token kernel spills registers on the M1. This is what makes MTP pay off here: **24 to 45 tok/s on code** with the multi-token kernels.

Plain decode at 4K went from 21 to 23 up to 35.5 tok/s.

**3. Why the curve stays flat.** This is the part I am proudest of. The attention layers (every fourth layer) do not read the whole cache. A small indexer scores blocks of keys, a top-k pass picks the blocks, and only those get full attention. So the cost per token should grow slowly with context, as long as scoring and selection are cheap. On the M1 they were not:

- *The indexer scorer.* Upstream had a fast vector scorer only for the M5. Everything else fell back to a scalar loop, which took **10.7 ms of a 42.5 ms token at 128K**. The fork rewrote both scorers around one shared helper that reads each key once and runs the four head dot products as independent chains in the same order. The vector scorer is now bit-identical to the scalar one, so it is on for every Apple GPU: 853 to 63 µs per layer at 32K blocks.
- *The top-k selection.* The radix select now counts runs of one digit in a register before it touches the shared counter, since the scores share few exponents (360 to 207 µs at 64K). Its gather gave each thread a contiguous chunk, so every load of a simdgroup touched 32 cache lines. Now each simdgroup owns a segment its lanes read together (206 to 110 µs), with the same order and the same output.
- *The attention loop.* It loaded one key, scored it, rescaled, then went to the next key, so every key paid the full memory latency. It now issues the loads for four keys before the first score and does one running-max correction per round. Attention per token went from 1.74 to 1.06 ms.

Together: **decode at 128K went from 23.8 to 33.8 tok/s, and at 262K from 17.7 to 29.7**, with byte-identical output. With MTP on top, that is the 35 to 44 tok/s curve above.

**4. Keeping the GPU busy.** At 25 ms per token, every millisecond the GPU waits for the CPU shows up in the result.

- The 16 n-gram rows per token are uncached SSD reads, 0.45 ms with the GPU idle. Only layer 1 needs them, so layer 0 now goes to the GPU first and the read runs while it computes.
- The sampler took 1.34 ms per token on the 248K vocabulary at the default settings (temperature above 0, min-p). It now takes 0.13 ms and returns the same token for the same seed.

**5. Prefill.** Agents send a few hundred new tokens after every tool call, then sometimes 30K at once, so both sizes matter.

- Small prompts: the expert GEMMs now use 8- and 16-token tiles for each expert's leftover tokens instead of padding to a big tile. 182 to 223 tok/s at 300 tokens.
- Big prompts, measured at 16K: the Q6_K decoder reads four weights at a time instead of branching per weight (it was 8x slower than Q4_K, 276 to 291). The gate/up tile reads the next expert's weights before the multiply, so the wait for cold memory overlaps with the math (to 303). And the Q2_0 expert tiles keep weights and accumulators in registers, using the same simdgroup layout as my Splash M1 port, with the activations staged once per threadgroup (to **327 tok/s**).

**6. Agent sessions that do not replay.** Qwen3.8 is a hybrid: three of every four layers are recurrent (Gated DeltaNet), and a recurrent state cannot be rewound to an earlier token. A normal KV cache can be cut back; this one cannot. So when an agent retried or edited a turn, the server had to prefill the whole transcript again. The server now keeps a copy of the recurrent state at the last turn marker. A retried turn on a 31K prompt went from **109 s to 0.3 s**. A second copy is saved to disk right before the first user message, so a new session with the same system prompt and tools starts from disk instead of prefilling them.

**7. Past the trained 262K window.** The model was trained on 262K. The fork applies YaRN with factor 2 above that. On 64 GB, 400K and 524K fit (about 33 KiB of cache per token). In a needle test with 20 facts spread over the document, it found 20 of 20 at 400K and 18 of 20 at 524K. The second half of a 524K document scores the same NLL (0.265) as the same text fed from position zero (0.276).

**How I checked it did not get dumber.** Every kernel has a test against the CPU reference path. Most of the changes above give byte-identical output, and the commit messages say which. ds4's `score_official` on Alibaba's reference continuations gives the same average NLL before and after the kernel work (0.45044 vs 0.45045 short, 0.17410 vs 0.17410 long). MTP only adds speed: in my greedy checks the text matched decoding without it.

And `dstar`, the launcher: it picks the context that fits your Mac, turns on MTP and vision, and prints the one `sysctl` line you need for long contexts.

**Benchmark 1: one chat to the full context.** A single conversation, the way an agent session grows: every turn adds about 30K tokens of C source and asks for a small function, reasoning xhigh, 800 tokens out. The answer and its reasoning go back into the history, so the server only appends. Speed is measured on the client from the stream.

| Context | Q2_0 decode | Q2_0 prefill | IQ3_XXS decode | IQ3_XXS prefill |
|---|---:|---:|---:|---:|
| 4K | 44.0 tok/s | 328 tok/s | 35.2 tok/s | 241 tok/s |
| 128K | 43.0 | 320 | 34.2 | 278 |
| 256K | 38.8 | 313 | 31.7 (259K) | 258 (259K) |
| 398K | 35.4 | 292 | | |

**Benchmark 2: sustained load and temperature.** npanj's five Splash prompts on a loop for 5 minutes, the same script as in my Splash posts, reasoning xhigh, 250 tokens each. Temperature, power and fans from `macmon`; my fans follow a Macs Fan Control curve from 50 C to full speed at 80 C.

![Five prompts for 5 minutes](img/flash-next-on-m1-max/2_sustained.png)

| | Decode | First quarter / last quarter | GPU while generating (max) | Energy per token |
|---|---:|---:|---:|---:|
| Q2_0 | 43.2 tok/s | 43.2 / 43.1 | 73.1 C (75.0) | 0.70 J |
| IQ3_XXS | 35.0 tok/s | 35.1 / 35.0 | 73.9 C (75.1) | 0.95 J |

No throttling: the GPU clock is the same at the start and the end of the run.

**Benchmark 3: real OpenCode sessions.** The same seven-step Three.js galaxy prompt as last time, reasoning medium, then more steps in the same session, and for Q2_0 a second task on top: read a 23K-line C file from start to end and write an architecture document. That one pushed the context to 371K before OpenCode compacted it.

![Real OpenCode sessions](img/flash-next-on-m1-max/3_opencode.png)

| Context | Q2_0, decode per turn (median) | IQ3_XXS |
|---|---:|---:|
| 0-64K | 44.4 tok/s | 33.8 tok/s |
| 64-128K | 42.2 | 34.7 |
| 128-192K | 38.5 | 33.8 |
| 192-256K | 38.1 | 31.4 |

Prefill in the sessions: 334 down to 298 tok/s for Q2_0, 282 down to 260 for IQ3_XXS.

**Against my Splash port (Qwen3.8-27B).** Same Mac, same benchmark script, same OpenCode galaxy task at reasoning medium, Splash 1.0.2-m1.1 with the 4-bit 27B and its DFlash draft, i.e. the numbers from my last post:

![Splash 27B vs Flash-Next on ds4 in OpenCode](img/flash-next-on-m1-max/5_vs_splash.png)

| | Qwen3.8-27B, Splash M1 | Flash-Next Q2_0, ds4 fork | Flash-Next IQ3_XXS |
|---|---:|---:|---:|
| Five prompts for 5 minutes, xhigh | 41.5 tok/s | 43.2 tok/s | 35.0 tok/s |
| Slowest / fastest prompt | 26.6 / 59.6 | 40.9 / 47.5 | 33.1 / 37.5 |
| GPU while generating | 75.3 C | 73.1 C | 73.9 C |
| Energy per token | 1.06 J | 0.70 J | 0.95 J |
| OpenCode decode, 0-64K | 30.6 tok/s | 44.7 tok/s | 34.2 tok/s |
| OpenCode decode, 64-128K | 19.8 | 42.7 | 34.6 |
| OpenCode decode, 128-192K | 14.2 | 38.1 | 32.7 |
| OpenCode prefill of new tokens | 79 down to 42 tok/s | 334 down to 302 | 282 down to 260 |
| GPU memory | 21 GB | about 48 GiB at 262K | about 57 GiB at 262K |

On short prompts they are even. In an agent session they are not: at 128K the 27B decodes at 14 tok/s and prefills at 42, Flash-Next at 38 and 316. A 30K tool result that took the 27B over 10 minutes to read takes Flash-Next about 90 seconds.

Why: the 27B is dense, so every token reads all of its 4-bit weights, and its attention layers read the whole cache, so both decode and prefill slow down as the session grows. Splash's DFlash draft hides that on short, predictable text (59.6 tok/s on the math prompt) and much less on prose (26.6). Flash-Next reads about 4 GB per token whatever the context, three of its four layers are recurrent, and the attention layers only look at the blocks the indexer picks. Its speed depends less on the text and much less on the context.

To be fair to the 27B: these are different models, and I am not claiming a 2-bit Flash-Next answers as well as a 4-bit 27B. The 27B fits in 21 GB, so it runs on a 32 GB Mac; Flash-Next needs the 64 GB. The Splash prefill numbers are new tokens over the client's wait for the first token, the ds4 ones the server's own prefill average, so treat that row as roughly comparable.

Along the way I also found and fixed a few bugs in the engine itself. They are in stock ds4 too, but I cannot send everything upstream at once, so the PRs to antirez will go gradually, small and one at a time.

Tip: if your OpenCode system prompt changes between runs (mine did, the same skills were installed in two folders), every run prefills the whole context again.

**For scale: a DGX Spark.** The most common single-Spark setup for this model is vLLM with the NVFP4 checkpoint and MTP. madeye's write-up measures it at **36.8 tok/s on prose and 45.8 on code**, single stream, and MiaAI-Lab's recipe reports about 37 on prose. The M1 Max with the fork: 43.2 tok/s on npanj's mixed prompts, 44 to 45 on code. So for one user decoding, a 2021 laptop is now in the same range as NVIDIA's AI box.

Two honest caveats. The Spark prefills at 1600 to 1800 tok/s against my ~320, so on a long fresh prompt it is 5x faster, and it has 128 GB, so it runs 4-bit weights where I run 2-bit. And people with tuned INT4 builds on the Spark report much higher decode than the vLLM recipe. Still, I did not expect this comparison to be close at all.

**DeepSeek V4 Flash on 64 GB.** The 81 GiB Q2 file does not fit, so ds4 keeps the dense weights and a cache of 5800 experts in memory and reads the rest from the SSD. Same benchmarks, 128K context:

| | DeepSeek V4 Flash Q2 |
|---|---:|
| Five prompts for 5 minutes | 11.8 tok/s (11.9 / 11.6), 66.5 C, 1.77 J per token |
| One chat, decode | 12.1 tok/s at 4K, 9.1 at 127K |
| One chat, prefill | 94 tok/s for the first chunk, 34 to 47 after it |
| OpenCode, 90 minutes | 23 requests to 85K, 9.3 tok/s median |

Decode holds up; prefill is the weak spot, since every new chunk of prompt pulls in experts from the disk. Usable for chat on a 64 GB Mac, slow for agents.

**Try it.** `dstar` is one launcher for every model ds4 runs, not only these quants. It picks the context that fits your Mac, turns on MTP and vision for Flash-Next, streams a model larger than your RAM from the SSD, and tells you the one `sysctl` line when a long context needs it.

The prebuilt release is the easy way: macOS 15 or newer, nothing to compile. It installs into `~/dstar` and puts `dstar` on your PATH.

```bash
curl -fsSL https://raw.githubusercontent.com/paperniuk/ds4/m1-flash-next/install.sh | bash
dstar doctor                    # what your Mac fits: memory, files, the max context per quant
dstar pull                      # Qwen3.8-Flash-Next Q2_0 + MTP block + vision encoder, 67 GB
dstar serve                     # OpenAI/Anthropic compatible server on http://127.0.0.1:8010/v1
dstar opencode                  # provider block to paste into OpenCode
```

Or build it from source and run the launcher from the repo folder:

```bash
git clone https://github.com/paperniuk/ds4.git
cd ds4 && make
./dstar doctor                  # the same commands, as ./dstar
```

Other quants, other models, other settings:

```bash
dstar pull --quant iq3          # IQ3_XXS: better answers, a quarter slower (iq3s for 96 GB+ Macs)
dstar serve --quant iq3 --ctx 262k
dstar serve --ctx 400k          # past the trained window, YaRN is set for you
dstar models                    # everything else ds4 runs: DeepSeek V4 / V4.1 Flash, GLM, ...
dstar pull ds4f-q2              # DeepSeek V4 Flash Q2, 81 GB
dstar serve deepseek            # any model by a word of its file name, or a path to a GGUF
dstar serve ~/models/some.gguf --ctx 65536   # anything after -- goes straight to ds4-server
dstar serve --power 50          # a cooler, quieter Mac at about half the speed
dstar chat                      # the same, in the terminal
```

**64 GB and up, for now.** Flash-Next needs a 64 GB Mac: the Q2_0 weights alone are 35 GiB, and ds4 cannot stream Qwen3.8 from the SSD yet. I do have plans for 32 and 48 GB Macs, but until then my Splash M1 port is the better choice there: its Qwen3.8-27B fits in 21 GB and runs at about 40 tok/s on short prompts.

**Raising the GPU memory limit.** By default macOS lets the GPU wire about three quarters of the RAM, 48 GiB on a 64 GB Mac. That is enough for Q2_0 up to 131K; for the long contexts you raise it (no reboot needed, and it resets at every reboot):

```bash
sudo sysctl iogpu.wired_limit_mb=57344   # Q2_0 at 262K and 400K
sudo sysctl iogpu.wired_limit_mb=61440   # IQ3_XXS at 262K, Q2_0 at 524K
```

You do not have to work this out: `dstar doctor` shows which contexts fit, and when the one you ask for does not, `dstar serve` starts with a smaller one and prints the exact line. At 61440 the server holds up to ~57 GiB of the 64, so close the browser and other heavy apps first.

**Not only M1/M2.** I wrote and measured everything on an M1 Max, but almost nothing is M1-only. The kernels, the long-context attention work, MTP, the prompt anchor and YaRN run on any Apple Silicon Mac; only three prefill tweaks are switched on by chip name, and the docs list the one variable each needs on other chips. So on M3, M4 and M5 this could well be one of the fastest ways to run Qwen3.8-Flash-Next right now, especially on 64 GB Macs and in long agent sessions. I have not tested that, though: on M5 stock ds4 already uses the new Metal 4 tensor path, so the gain there is unknown. If you have one of these Macs, `dstar doctor` plus one `dstar chat` run is enough for a useful report, and I would love to see the numbers.

## What this says about the M1 Max

To me this is the real result, more than any single number above. A laptop from 2021 runs a mixture-of-experts model with about 126B parameters (6.7B active per token, counted from the GGUF) at 35 to 44 tok/s, holds that across 400K tokens of context, and decodes in the same range as NVIDIA's 2025 AI box. The chip was never the bottleneck; the software was.

How much of the hardware it actually uses, counted from the tensor sizes and measured on this Mac:

- **Bandwidth.** The M1 Max has 400 GB/s, more than the DGX Spark's 273. Every token reads about 4.0 GB of weights plus about 0.2 GB of recurrent state. Plain decode at 35.5 tok/s is about **150 GB/s, under 40% of the peak**. The big matvecs alone run at 200 to 300 GB/s from DRAM (up to 75% of the peak); the rest of the time goes to small work, about 1000 GPU dispatches per token and 30 expert matrices of under 0.5 MB per layer. MTP then gets 44 tok/s out of the same traffic by checking 2 to 3 tokens per pass over the weights.
- **Compute.** The M1 Max matrix units peak at about 10 TFLOPS in fp16 (I measured 9.9 with simdgroup_matrix). Prefill does about 12 GFLOP per token, so 327 tok/s is about **4 TFLOPS, roughly 40% of the peak**. Dense tiles reach 6 to 8 TFLOPS; the 512 small experts are what holds the average down.

So this is not the ceiling. Even on a five-year-old laptop there is still headroom left in both decode and prefill, and that is what I find most exciting about this whole thing.

## Links
- The fork: https://github.com/paperniuk/ds4 (branch `m1-flash-next`)
- The prebuilt release: https://github.com/paperniuk/ds4/releases/latest
- ds4 (DwarfStar) by antirez: https://github.com/antirez/ds4
- The GSQ-RCO quants by ISTA-DASLab: https://huggingface.co/ISTA-DASLab
- npanj's benchmark prompts: https://github.com/npanj/splash-plus
- DGX Spark numbers: https://madeye.github.io/qwen38-flash-next-on-dgx-spark/ and https://forums.developer.nvidia.com/t/miaai-lab-new-qwen3-8-flash-next-nvfp4-recipe-for-1x-dgx-spark-1m-context-vision-video-37-tok-s-c1/382446

The engine is antirez's and the quants are ISTA-DASLab's; this is an unofficial fork for Apple Silicon. Model weights are under the Qwen community license.
