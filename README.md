# Egocentric Activity Agent

Multisensor activity recognition for first-person recordings (camera + IMU), built in PyTorch, with an LLM agent on top that answers questions like *"When was I stirring in session 24?"* by calling tools over a searchable index of moments.

It is a small, end-to-end version of the problem that physical-AI products face: several sensor streams on different clocks, a deep model that fuses them, classical baselines to keep the deep model honest, unsupervised structure in the embeddings, and an agent layer that turns predictions into answers.

```
camera 10 Hz ─┐                      ┌─ CNN per frame ─ Transformer over time ─┐
              ├─ clock-offset fix ─ windows                                    ├─ fusion ─ embedding ─ classifier
IMU 50 Hz ────┘   (cross-correlation)  └─ 1D CNN over 6 IMU channels ───────────┘            │
                                                                                             ▼
                        question ─▶ agent ─▶ tools (find / timeline / stats / similar) ─▶ moment index
```

## What's inside

| Part | File | What it does |
|---|---|---|
| Synthetic sessions | `egoagent/synthetic.py` | Generates first-person video (32×32, 10 fps) and 6-axis IMU (50 Hz) for 5 activities with known ground truth and a configurable IMU clock offset |
| Sensor alignment | `egoagent/sensors.py` | Estimates the IMU clock offset by cross-correlating activity-change signals from both sensors, then cuts aligned 2 s windows |
| Deep models | `egoagent/models.py` | CNN + Transformer video encoder, 1D CNN IMU encoder, late fusion; video-only / IMU-only / fusion for ablations |
| Training | `egoagent/training.py` | Single-process training and **DistributedDataParallel** over several processes (gloo on CPU, nccl on GPU) |
| Classical ML | `egoagent/classical.py` | Hand-crafted features (FFT dominant frequency, spectral energy, motion energy), **gradient boosting**, **RBF SVM**, **k-means** clustering quality (ARI, NMI) |
| Moment index | `egoagent/index.py` | Merges window predictions into timelines; finds activities, totals time, finds similar moments by embedding cosine |
| Agent | `egoagent/agent.py`, `egoagent/llm.py` | Function-calling loop with JSON-schema tools; tool errors come back as data so the model can retry; works with any OpenAI-compatible endpoint, plus an offline rule-based planner |
| Real video | `egoagent/video_io.py`, `egoagent/vlm.py` | OpenCV loader and a **CLIP** zero-shot labeller for real first-person footage |

## Results

All numbers come from `python -m egoagent demo` (seed 0, 32 sessions of 60 s, split **by session**: 24 train / 8 test, 1,416 / 472 windows), and are saved in `artifacts/results.json`. CPU only, 2 cores.

**Activity recognition on held-out sessions**

| Model | Input | Accuracy | Macro F1 |
|---|---|---|---|
| Gradient boosting | hand-crafted IMU + video features | 0.970 | 0.971 |
| RBF SVM | hand-crafted IMU + video features | 0.945 | 0.942 |
| 1D CNN | IMU only | 0.979 | 0.981 |
| CNN + Transformer | video only, raw frames | 0.968 | 0.972 |
| CNN + Transformer | video only, background-removed + frame-difference input | 0.947 | 0.943 |
| **Fusion** | video + IMU | **0.981** | **0.983** |
| Fusion, DDP over 2 processes | video + IMU | 0.977 | – |

**Unsupervised structure (k-means, k = 5, labels only used for scoring)**

| Features | Adjusted Rand index | NMI |
|---|---|---|
| Hand-crafted features | 0.655 | 0.668 |
| Learned fusion embeddings | **0.945** | **0.929** |

**Clock alignment.** True IMU offsets ranged from −0.80 s to +0.80 s; the estimate was off by 48 ms on average and 180 ms at worst across 32 sessions.

### What the experiments actually showed

- **The deep model only just beats the baselines here.** Gradient boosting on 41 hand-crafted features reaches 0.970; fusion reaches 0.981. On data this clean, a strong classical baseline is most of the way there, which is exactly why it is in the pipeline.
- **The biggest win from deep learning is the embedding, not the classifier.** k-means on the learned embeddings recovers the activities far better (ARI 0.945) than on hand-crafted features (0.655). That matters for finding activities nobody labelled.
- **My background-removal idea did not work.** I expected that subtracting each clip's mean frame would stop the video model from memorising per-session backgrounds. The ablation says otherwise: 0.947 vs 0.968 for raw frames on the full run, and 0.746 vs 0.869 on a 16-session run. Raw frames are now the default; the other input stays as an option so the ablation can be rerun.
- **Clock sync helps a little, and not every time.** On the full run, fusion with and without offset correction both scored 0.981. On the smaller 16-session setting over three seeds (`experiments/clock_sync_seeds.py`), corrected clocks were never worse and averaged 0.984 vs 0.966, but one seed showed no difference. Offsets under a second are small next to 2 s windows and multi-second activities; I would expect the gap to grow with shorter windows or faster actions.
- **DDP gives the same model, not a faster one, on this machine.** Both ranks finish with identical weights (checked by an all-gather of parameter checksums), and accuracy matches single-process training. With 2 CPU cores and a small model, communication overhead makes it slower (50 s vs 42 s); the point is that the training code is ready for multiple GPUs.

### Honest limits

- The data is synthetic. It has known ground truth, which makes the experiments clean, but real egocentric video is far harder (occlusion, lighting, camera shake, long-tail activities). Treat the accuracies as a test of the pipeline, not of real-world performance.
- Every result is from one seed unless stated otherwise.
- The default agent planner (`RuleBasedLLM`) is a keyword router so the demo runs offline. It is not a language model. Use `--llm openai` with a real endpoint for free-form questions.
- The CLIP labeller is unit-tested with a fake backend but not evaluated here, because 32×32 synthetic blobs give CLIP nothing to recognise. It is meant for real footage.

## Quick start

```bash
git clone https://github.com/Akshitaa-V/ego-activity-agent
cd ego-activity-agent
python -m venv .venv
# Linux/macOS: source .venv/bin/activate     Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,video]"

python -m egoagent demo            # full experiment, about 5 minutes on 2 CPU cores
python -m egoagent demo --quick    # 30-second smoke run
python -m egoagent ask "When was I stirring in session 24?"
python -m egoagent ask "How long did I spend on each activity in session 25?"
python -m egoagent ask "Show me moments similar to session 24 at 10s"
pytest -q                          # 41 tests
```

Example from the full run (ground truth for that session: stirring from 38.3 s to 43.6 s):

```
$ python -m egoagent ask "When was I stirring in session 24?"
[tool] find_moments({'session_id': 24, 'activity': 'stirring'}) -> ok in 0.3 ms
Found 1 moment(s): 38.0-43.0s stirring (session 24).
```

### Using a real LLM

Any endpoint that speaks the OpenAI chat-completions format with tool calling works (OpenAI, vLLM, Ollama, a university GPU cluster):

```bash
export EGOAGENT_BASE_URL=https://your-endpoint/v1
export EGOAGENT_MODEL=your-model
export EGOAGENT_API_KEY=...
python -m egoagent ask "what did I do right after typing in session 26?" --llm openai
```

### Running on real video

```python
from egoagent.video_io import load_video
from egoagent.vlm import ZeroShotLabeller   # pip install -e ".[vlm]"

times, frames = load_video("my_clip.mp4", target_fps=2, size=224, rgb=True)
labeller = ZeroShotLabeller()               # CLIP ViT-B/32 from Hugging Face
print(labeller.score_clip(frames[:8]))       # {'stirring': 0.71, 'typing': 0.12, ...}
```

## Design notes

- **Split by session, never by window.** Neighbouring windows overlap and share a background, so a random window split would leak test data into training.
- **Tools return errors instead of raising.** A wrong activity name or unknown session comes back as `{"ok": false, "error": "..."}`, the agent passes it to the model, and the model can correct itself. A test covers exactly that recovery.
- **Normalisation lives inside the model.** IMU mean and standard deviation are stored as buffers, so a saved model normalises inputs the same way at inference.
- **Windows on Windows:** DDP uses `torch.multiprocessing.spawn`, so call it from a script guarded by `if __name__ == "__main__":` (the CLI already is).

## Project layout

```
egoagent/        package (synthetic, sensors, models, training, classical, index, agent, llm, vlm, video_io, pipeline, cli)
experiments/     clock_sync_seeds.py
tests/           41 pytest tests, including DDP, the CLI end to end and the video loader
artifacts/       results.json from the last full run
.github/         CI: ruff lint + format check and pytest on Ubuntu, Python 3.10 and 3.12
```

## Licence

MIT
