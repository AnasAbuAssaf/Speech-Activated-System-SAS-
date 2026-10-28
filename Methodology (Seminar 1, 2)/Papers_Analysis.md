# Analysis of Research Papers

---

## Paper 1: Benchmarking Google Speech Commands with a New Data Extension

<https://wydawnictwo.pan.pl/index.php/ijet/article/view/2928/2254>

**Reference:** Kucharski, Idzikowski, Sarnicka. *International Journal of Electronics and Telecommunications*, 2026, Vol. 72, No. 3.

**Goal:** Reproduce and verify the results of the best keyword-spotting (KWS) models that were published on Papers with Code (after it was shut down), and test them on new data to measure generalization.

### Tools and Techniques

- **Data:** Google Speech Commands (GSC) plus a new extension from Wrocław University: 3,710 recordings from 106 speakers (non-native English speakers, mostly Polish and some Ukrainian), one utterance of each of the 35 words (GSC v2) per speaker, i.e. $106 \times 35 = 3{,}710$.
- **Recording:** A room conforming to the ITU-T P.800 standard, an MXL 770 condenser microphone, and a Focusrite Scarlett Solo sound card. Recorded at $512\ \text{kHz}$, then downsampled to $16\ \text{kHz}$ and cropped to one second.
- **The four repositories tested:** ML-KWS-for-MCU (DNN, CNN, LSTM, GRU, CRNN, DS-CNN in S/M/L sizes), Keyword Transformer (KWT-1/2/3 and MHAtt-RNN), Howl (Res8), and SpeechCmdRecognition (Attention RNN).
- **Infrastructure:** NVIDIA DGX Station A100 (4×A100 80GB), TensorFlow and TFLite, the `soundfile` library, and Python evaluation scripts.
- **Metrics:** Precision, Recall, $F_1$, Support, and the confusion matrix.

### Methodology

Of the 42 entries listed on Papers with Code, 32 were original papers, 21 of those came with code, 12 actually worked, and only 4 frameworks were testable:

$$
42 \rightarrow 32 \rightarrow 21 \rightarrow 12 \rightarrow 4
$$

Frameworks with pretrained models were tested directly.

- **Howl** was trained from scratch: AdamW, learning rate $0.01$ multiplied by $0.8$ every epoch, 20 epochs, batch size 64, cross-entropy loss, on 10 commands.
- **Attention RNN:** Conv1D followed by a two-layer BiLSTM, Adam with learning rate $0.001$ multiplied by $0.4$ every 10 epochs, up to 40 epochs with early stopping after 10 epochs without improvement.

All models were evaluated on the official GSC test set and on the new extension.

### Results

| Model | Reported accuracy | On GSC | On the extension |
|---|---|---|---|
| DNN (S / M / L) | 0.846 / 0.864 / 0.867 | 0.853 / 0.880 / 0.897 | **0.567 / 0.594 / 0.651** |
| CNN-L | 0.927 | 0.933 | 0.950 |
| CRNN-L | 0.950 | 0.945 | 0.976 |
| DS-CNN-L | 0.954 | 0.947 | 0.963 |
| GRU-L | 0.947 | 0.948 | 0.974 |
| KWT-1 | 0.9856 | 0.9922 | 0.9972 |
| KWT-2 | 0.9843 | 0.9988 | **1.0000** |
| KWT-3 | 0.9808 | 0.9995 | 0.9991 |
| MHAtt-RNN | 0.9836 | 0.9947 | 0.9981 |
| Attention RNN | 0.939 | **0.7845** | 0.9353 |
| Howl (Res8) | — | Close to 99% (from the confusion matrix) | Close to 99% |

- Most models generalized well to new speakers, and some improved by 2–3 points. The KWT models were the most stable ($> 99\%$).
- The simple DNN collapsed (a drop of $\approx 28$ points), which shows its sensitivity to acoustic variability. The authors saw this as a sign of overfitting.
- Attention RNN reached only 78% on the original GSC despite 94% on their own data. This was explained by the homogeneity of its training data.
- **The most important message for our project: testing a model on newly recorded data with new speakers is the most honest measure of generalization.**

### Critical Notes

- Confusion-matrix tables XVI and XVII (Attention RNN on their data and on GSC) are numerically identical despite the different stated accuracies, which looks like a copy error.
- The extension contains one recording per word per speaker, so the matrices show 106 samples per class, and the `silence` and `unknown` classes are effectively empty.

---

## Paper 2: HArnESS — Lightweight Arabic Foundation Models via Distillation

<https://aclanthology.org/2026.speakable-1.12.pdf>

**Reference:** Sukhadia & Chowdhury. SPEAKABLE @ LREC 2026, pp. 109–117.

**Goal:** Build a family of self-supervised learning (SSL) models for Arabic speech, trained from scratch, then compress them into small models using iterative self-distillation.

### Tools and Techniques

- **Architecture:** A CNN feature extractor (7 layers) plus a Transformer, in the style of HuBERT. Masked prediction of pseudo-labels from K-means ($K = 1000$).
- **Models:** HArnESS-L (24 layers, 316M parameters), HArnESS-S (4 layers, 65M), and HArnESS-ST (4 layers, thin, 28M). Also an experiment with PCA (dimension 512) to compress the supervision signal.
- **Data:** About 4,001 hours of Arabic plus 3,565 hours of English (QASR, MGB3, LibriSpeech, Common Voice, GigaSpeech, and YouTube videos from 15 Arab countries), augmented up to 23,000 hours (speed change 0.9/1.1, noise, SpecAugment). Distillation used about 1,100 hours of Arabic from QASR.
- **Hardware and tools:** fairseq, $24 \times$ H100 for pretraining and $8 \times$ H100 for distillation, and ESPnet for speech recognition.

### Methodology

Three iterations: (1) labels from MFCC, (2) labels from layer 9, (3) distillation and compression by reducing depth, width, and number of heads, with either random initialization or block-averaging initialization.

Evaluation freezes the encoder and places a light classifier on top:

- **DID and SER:** 3 convolutional layers plus attention pooling, batch size 4, 10k steps.
- **ASR:** A Conformer/Transformer model fine-tuned on 300 hours of QASR and tested on MGB2 and MGB3.

Tasks: ASR, dialect identification (ADI5), and emotion recognition (KSUEmotion). Metrics: WER and accuracy.

### Results

| Model | WER on MGB2 (lower is better) | WER on MGB3 | SER (KSUEmotion) | DID (ADI5) |
|---|---|---|---|---|
| HuBERT-L (English) | 22.6 | 51.2 | 91.92% | 64.14% |
| XLS-R (multilingual) | 22.60 | 51.80 | 73.32% | 42.35% |
| **HArnESS-L** | **15.50** | **41.60** | **94.66%** | **84.98%** |
| HArnESS-S | 20.20 | 52.80 | 91.15% | 70.84% |
| HArnESS-ST | 23.20 | 58.20 | 89.02% | 69.77% |
| HArnESS-ST + PCA | 22.50 | 55.60 | 87.34% | 61.64% |

- HArnESS-L outperforms HuBERT and XLS-R on all Arabic tasks. A compression of 79.4% (the S model) costs $+4.7$ points of WER and $-14.4$ points in DID, meaning dialect cues are the first thing affected by reducing depth.
- Extreme compression in dimension ($\approx 96.5\%$) drops performance sharply. PCA speeds up convergence but does not clearly improve the final result.

### Critical Notes

- Evaluation uses a frozen encoder only, and the authors acknowledge that full fine-tuning might give different results. Their comparison with published results (Fanar) is indirect and does not represent SOTA.
- Figure 2 plots WER with negative bars, so it is better to rely on the tables to read the values.

---

## Paper 3: SynTTS-Commands — Synthetic Command Data Generated with TTS

<https://arxiv.org/pdf/2511.07821>

**Reference:** Gan & Li. arXiv:2511.07821v2 (November 2025).

**Goal:** Show that synthetically generated speech can replace human recordings for training voice-command classifiers on tiny devices (TinyML).

### Tools and Techniques

- **Generation:** The CosyVoice 2 model (from Alibaba: an LLM that converts text to semantic tokens, plus flow-matching, with FSQ), using speaker embeddings from VoxCeleb1/2 (5,994 speakers from VoxCeleb2) and Free ST Chinese Mandarin (855 speakers).
- **Selection:** A quality score computed from the duration $D$ and the signal-to-noise ratio $S$, used to pick the best reference clip for each speaker:

$$
Q = 100 \times \left( 0.5 \times \frac{D}{5} + 0.5 \times \frac{S}{30} \right)
$$

- **Verification:** A bilingual ASR model (Paraformer) computes text similarity, and anything below 60% is reviewed manually (clarity, completeness, correctness, accuracy).
- **Reference models:** MicroCNN (4K parameters), DS-CNN, TC-ResNet, CRNN, MobileNetV1, and EfficientNet-B0.

### Methodology

48 commands (25 Chinese and 23 English) in the media domain (playback, navigation, volume, calls, wake words) produce 384,621 clips (111.33 hours, 35.76 GB), with a 70/15/15 speaker-disjoint split and separate test sets for each language.

### Results

| Model | English accuracy | Chinese accuracy | Parameters (EN) |
|---|---|---|---|
| MicroCNN | 93.22% | 80.14% | 4,189 |
| DS-CNN | 99.46% | 97.18% | 30,103 |
| TC-ResNet | 98.87% | 96.56% | 68,431 |
| CRNN | **99.50%** | 97.42% | 1,083,031 |
| MobileNet-V1 | 99.50% | 97.92% | 2,651,783 |
| EfficientNet-B0 | 99.41% | 97.93% | 4,717,248 |

- The DS-CNN model, with only 30 thousand parameters, approaches the best result, making it a practical choice for tiny devices.

### Critical Notes

- Both evaluation and training use synthetic data (the same generator), and there is no test on real human speech. The claim that it is a "superior alternative" therefore goes beyond the evidence presented. There are also no negative (out-of-vocabulary) samples and no noise.
- The available commands are English and Chinese only; Arabic is not currently supported (it is part of the future plan).

---

## Paper 4: Automatic Speech Recognition Challenges for Adults with Cognitive Impairment

<https://arxiv.org/pdf/2602.23436>

**Reference:** Cohn et al. Accepted at CHI 2026 (UC Davis and others).

**Goal:** Measure the performance of ASR (Whisper) on Alexa commands read aloud by older adults, according to cognitive status, and link the errors to acoustic characteristics.

### Tools and Techniques

- **Data:** The VAS set from DementiaBank: 83 participants after exclusions (30 normal, 28 mild cognitive impairment (MCI), 25 dementia), $30 + 28 + 25 = 83$, and 30 commands (29 were used).
- **ASR:** Whisper in two versions, small (244M) and medium (769M), run locally (HIPAA privacy). The JiWER library computes WER.
- **Acoustic analysis:** Praat (speech and articulation rate, silence ratio, jitter, shimmer, intensity, pitch and its variability).
- **Statistics:** Mixed regression models using `lme4` in R, `MuMIn` for model comparison (AICc), and the `performance` package to check collinearity (VIF).

### Methodology

Text cleaning (removing punctuation, lowercasing, converting numbers), then WER per clip, then a mixed model with these variables: cognitive status, command type (IoT / reminder / question), Whisper model, age, gender, ethnicity, education, and prior experience.

### Results

- Mean WER for dementia patients is **$\approx 0.32$**, more than double that of normal participants ($\approx 0.12$), with $p < 0.001$. There is no significant difference between normal participants and MCI ($\approx 0.14$).
- The worst performance for dementia patients was on Internet-of-Things commands (lighting control and similar), which matter most for smart-home applications.
- The larger model (medium) reduced WER for everyone, and more so for dementia patients.
- Acoustic characteristics that predict errors: **higher shimmer, lower loudness, and a lower silence ratio.**
- Recommendations: fine-tune or personalize models per speaker, asynchronous human review of transcripts assisted by caregivers, delayed end-of-utterance detection, and alternatives to the wake word such as a button.

### Critical Notes

- The speech is read aloud, not spontaneous (a conservative estimate), the data is American English only, and the MCI classification was based on a MoCA score, not a clinical diagnosis.

---

## Paper 5: Evaluating Commercial ASR Systems on Code-Switching (Arabic, Persian, German)

**Reference:** Abdoli et al. (Perle AI). arXiv:2605.19069v3, May 2026.

**Goal:** Build a benchmark for evaluating five commercial systems on mixed-language speech (code-switching), and show that BERTScore is more suitable than WER for measuring the size of differences in Arabic and Persian.

### Tools and Techniques

- **Systems:** ElevenLabs Scribe v2, OpenAI gpt-4o-transcribe, Google Chirp 3, Azure Speech (CLID), and Deepgram Nova-3 (German only).
- **Data:** 1,200 clips (300 per pair: Egyptian–English, Saudi–English, Persian–English, German–English), $4 \times 300 = 1{,}200$, recorded by native speakers with headset/USB microphones in a quiet environment.
- **Two-stage selection:**
  1. A heuristic filter, `H_Score`, built from five signals (script-mixing ratio, switching rate, morphological fusion, length, vocabulary diversity).
  2. Evaluation by GPT-4o and Gemini 1.5 Pro on six linguistic dimensions, averaged. This cut LLM cost by about 91%.
- **Metrics:** WER (`jiwer` library), BERTScore with multilingual mBERT, Kendall's $\tau$ correlation, and UMAP analysis of sentence embeddings (Sentence-BERT).

### Methodology

Text normalization (lowercasing, removing punctuation) while keeping Arabic and Persian letters, deliberately without unifying hamza forms or diacritics. Analysis by difficulty quartile (Q1–Q4) and by language pair.

### Results

| System | Mean WER | Mean BERTScore |
|---|---|---|
| **ElevenLabs Scribe v2** | **13.2%** | **0.936** |
| OpenAI gpt-4o-transcribe | 38.6% | 0.856 |
| Google Chirp 3 | 39.4% | 0.862 |
| Azure CLID | 43.6% | 0.839 |
| Deepgram Nova-3 (German only) | 5.0% | 0.959 |

- Egyptian–English: ElevenLabs 13.1% versus 45.9% for the nearest system (OpenAI) and 59.7% for Google.
- The system ranking by WER and by BERTScore is identical on the Arabic and Persian pairs ($\tau = 1.0$), but WER inflates the differences about 3×, because it penalizes spellings that are phonetically and semantically correct (*feature* versus فيتشر).
- Performance degrades with difficulty: the gap between ElevenLabs and Google is 2.4 points in Q1 and 41.5 in Q4.

### Critical Notes

- The authors are a commercial company (Perle AI), and the speakers read pre-written texts (not fully spontaneous speech).
- No confidence intervals or significance tests are reported for the differences between systems. A sample of 300 clips per pair is relatively small, and the samples were deliberately selected to be hard, so they do not represent general everyday speech.
- Useful for our project: when evaluating Arabic commands, it is preferable to report WER and BERTScore together.

---

## Paper 6: SM-STMC — Short-Term Memory Convolutions with Submodels for Keyword Spotting

<https://arxiv.org/pdf/2609.35005>

**Reference:** Warlewski et al. (Samsung R&D Poland). arXiv:2609.35005v1, September 2026.

**Goal:** Reduce the computation and memory of CNN models for streaming keyword spotting on tiny devices (earbuds, watches) without retraining.

### Tools and Techniques

- **Data:** Google Speech Commands (10 commands + a negative class), with an extended version that adds 0.5 seconds of silence on each side to test window boundaries.
- **Models:** A VGG-type backbone (two receptive fields: 41 and 38 frames) plus an MLP classifier, compared with an LSTM. Mel feature extraction (window 1024, hop 256).
- **The idea:** STMC stores intermediate activations so they can be reused. SM-STMC splits the network into independent submodels executed on a static schedule (the deepest one runs only when needed), which eliminates redundant states and needs no conditional controller inside the model.
- **Hardware:** ARM Cortex-M55 at $196\ \text{MHz}$ (Helium extension), int8 quantization, TensorFlow Lite Micro, and CMSIS-NN.

### Results

| Metric | Result |
|---|---|
| SM-STMC1 on GSC (1 s) | Accuracy 0.9382 and $F_1 = 0.9403$, versus VGG1: 0.9278 and 0.9292 |
| SM-STMC1 accuracy on the 2 s version | $F_1 = 0.9715$, whereas VGG1 with a single window reaches only 0.5798 |
| MCPS reduction | Up to 82% versus the sliding window, and 46% versus conventional STMC |
| Memory | 65% reduction compared with STMC (9,280 versus 17,184 bytes) |
| Latency | Classification every 4 or 8 frames; stays within real-time requirements |
| LSTM reference | Accuracy 0.9511 on GSC and a cost of 10.35 MCPS |

### Critical Notes

- The summary numbers (93.8% and 97.1%) match the recall (≈ accuracy) in Tables 3 and 4 for SM-STMC1. The results here are on GSC with only ten commands, not on Arabic commands.
- The one-second VGG1 setup fails because it relies on a fixed window. This justifies continuous streaming, which matters for the live mode in our project.

---

## Quick Comparison Table

| Paper | Type | Key takeaway for the project |
|---|---|---|
| 1. Kucharski | KWS benchmark | Test on new data; KWT, CRNN, and DS-CNN are the best; DNN is ruled out |
| 2. HArnESS | Arabic SSL models | A strong future Arabic foundation; dialects are affected by compression |
| 3. SynTTS | Synthetic data | Data augmentation via TTS for Arabic (with verification on real speech) |
| 4. Cohn (CHI) | Accessibility | Older adults and dementia: delayed end-of-utterance detection, button alternative, personalized tuning |
| 5. Perle | Code-switching | Arabic–English in real commands; WER + BERTScore |
| 6. SM-STMC | Streaming KWS | Continuous running on a tiny device with less computation |
