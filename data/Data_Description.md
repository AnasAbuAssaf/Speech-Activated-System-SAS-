# Description of the Data Used in the Project

**Arabic Speech Commands** (Arabic) + **Google Speech Commands** (English)

**Sources:** the GitHub repository of the Arabic Speech Commands Dataset (abdulkaderghandoura), the Kaggle page of Google Speech Commands (neehakurelli), the official dataset card, the Warden paper, and the Kucharski paper.

---

## 1. Overview and Comparison

| Item | Arabic Speech Commands | Google Speech Commands (v0.02) |
|---|---|---|
| Language | Arabic (Modern Standard) | English |
| Number of words | 40 keywords | 35 words (30 in v0.01) |
| Number of files | 12,000 | 105,829 (v0.01: 64,727) |
| Speakers | 30 participants, each recorded 10 repetitions of every word | Thousands of speakers, collected by crowdsourcing |
| Files per word | Exactly 300 (perfectly balanced) | About 1,500–4,000 depending on the word (partially balanced) |
| File format | WAV, 1 second, $16\ \text{kHz}$ | WAV, $\le 1$ second (usually padded to 1 s), $16\ \text{kHz}$, mono |
| Size | ≈ 384 MB (+ 49 MB noise) | Original v0.02 is a few GB compressed |
| License | CC BY 4.0 | CC BY 4.0 |
| Split | Fixed 60/20/20 via CSV files, separated by speaker | 80/10/10 via `validation_list` and `testing_list` (filename hashing) |

---

## 2. Arabic Speech Commands (source: GitHub)

Created by Ghandoura, Hjabo, and Al Dakkak (*Engineering Applications of Artificial Intelligence*, 2021) for training simple educational and research models for keyword spotting.

### 2.1 Properties

- **Size and structure:** 12,000 (signal, word) pairs $= 30 \text{ speakers} \times 10 \text{ repetitions} \times 40 \text{ words}$. There are 40 folders (one per word), each containing 300 files.
- **Naming:** example `rotate/00000021_NO_06.wav`. The first eight digits are the speaker ID, and the last two digits are the round number (repetition).
- **Noise:** a `background_noise` folder with a variety of natural noise recordings (≈ 49 MB) for data augmentation.
- **Files:** the folders `dataset`, `background_noise`, and `images`, and the files `train.csv`, `val.csv`, and `test.csv` (columns: `file` and `class`).

### 2.2 Split (verified from the CSV files)

| Set | Files | Speakers | Files per word |
|---|---|---|---|
| Train | 7,200 | 18 | 180 |
| Val | 2,400 | 6 | 60 |
| Test | 2,400 | 6 | 60 |

The counts follow directly from the number of speakers in each set:

$$
\begin{aligned}
\text{Train} &= 18 \times 10 \times 40 = 7{,}200 \\
\text{Val} &= 6 \times 10 \times 40 = 2{,}400 \\
\text{Test} &= 6 \times 10 \times 40 = 2{,}400
\end{aligned}
$$

There is no speaker overlap between the three sets, so the test result truly measures generalization to new speakers. However, the total number of speakers is very small (30).

### 2.3 The 40 Words

The "Shared" column means the word has a literal counterpart in Google Speech Commands v0.02.

| Arabic | English | Shared | Arabic | English | Shared |
|---|---|---|---|---|---|
| صفر | Zero | Yes | تفعيل | Enable | — |
| واحد | One | Yes | تعطيل | Disable | — |
| اثنان | Two | Yes | موافق | Ok | — |
| ثلاثة | Three | Yes | إلغاء | Cancel | — |
| أربعة | Four | Yes | فتح | Open | — |
| خمسة | Five | Yes | إغلاق | Close | — |
| ستة | Six | Yes | تكبير | Zoom in | — |
| سبعة | Seven | Yes | تصغير | Zoom out | — |
| ثمانية | Eight | Yes | السابق | Previous | — |
| تسعة | Nine | Yes | التالي | Next | — |
| يمين | Right | Yes | إرسال | Send | — |
| يسار | Left | Yes | استقبال | Receive | — |
| أعلى | Up | Yes | تحريك | Move | — |
| أسفل | Down | Yes | تدوير | Rotate | — |
| أمام | Forward | Yes | تسجيل | Record | — |
| خلف | Backward | Yes | إدخال | Enter | — |
| نعم | Yes | Yes | رقم | Digit | — |
| لا | No | Yes | اتجاه | Direction | — |
| ابدأ | Start | — | خيارات | Options | — |
| توقف | Stop | Yes | تراجع | Undo | — |

The 19 shared words are: the digits 0–9 (10), yes/no (2), up/down/left/right (4), forward/backward (2), and stop (1), which gives $10 + 2 + 4 + 2 + 1 = 19$.

The English words *on*, *off*, and *go* have no literal counterpart in the Arabic set. "Enable" and "Disable" are the closest in meaning to *on* and *off*, but they are different words phonetically.

---

## 3. Google Speech Commands (source: Kaggle)

Created by Pete Warden (Google) in 2018 for training and evaluating keyword-spotting systems. The linked page is a copy published on Kaggle by the user neehakurelli.

> [!WARNING]
> **Important notice:** I could not read the content of the Kaggle page itself (it relies on JavaScript), so I do not know for certain which version it contains (v0.01 or v0.02) or the size of its files. The information below is taken from the official dataset card and the Kucharski paper. Open the page and verify the file name and the number of folders before starting.

### 3.1 The Two Versions

| Item | v0.01 (August 2017) | v0.02 (April 2018) |
|---|---|---|
| Number of files | 64,727 | 105,829 |
| Number of words | 30 | 35 (adds: *backward, forward, follow, learn, visual*) |
| The ten core commands | yes, no, up, down, left, right, on, off, stop, go | Same |

### 3.2 The Words

- **Core commands (10):** yes, no, up, down, left, right, on, off, stop, go.
- **Auxiliary words (20 in v0.01):** zero–nine, bed, bird, cat, dog, happy, house, marvin, sheila, tree, wow.
- **Added in v0.02 (5):** backward, forward, follow, learn, visual.

### 3.3 Technical Properties

- WAV files of one second or less (the word itself is shorter, so the clip contains silence at the edges), sampled at $16\ \text{kHz}$ with a single channel.
- Crowdsourced recordings with varied microphones and environments. Speakers are anonymous and attempting to identify them is not permitted, so the dataset is unsuitable for speaker-recognition tasks.
- A `_background_noise_` folder with noise recordings (household noise, white/pink noise, and others) that are mixed with the words for training.
- The files `validation_list.txt` and `testing_list.txt` define the standard split. The split relies on hashing the file name so that each speaker stays within a single set.
- License: CC BY 4.0. Cite the Warden paper (arXiv:1804.03209).

### 3.4 The Common Standard Task

12-class classification: the ten core commands, plus `silence`, plus `unknown` (all remaining words), i.e. $10 + 1 + 1 = 12$ classes. This is the task used in most papers (for example, the SM-STMC paper with 10 commands and a negative class).

---

## 4. Differences and Their Impact on Training

- **Size imbalance:** The English set is about 9 times larger ($105{,}829 / 12{,}000 \approx 8.8$), and its speakers number in the thousands versus 30. The Arabic model is therefore prone to overfitting to the 30 speakers, which is exactly what the Kucharski paper warned about (performance drops on new speakers).
- **Vocabulary mismatch:** Only 19 words are shared. The rest of the Arabic words (e.g., open, close, send) have no counterpart in GSC, and the rest of GSC (e.g., cat, dog, on, off, go) is outside the control domain.
- **Recording conditions:** The Arabic data was mostly recorded with more homogeneous microphones and environments, while GSC is a diverse crowdsourced recording (noise, devices). This causes a generalization gap when the two are combined.
- **Dialects:** The Arabic is Modern Standard (Fusha). The data does not cover local dialects (Syrian, Egyptian, Gulf), which means commands spoken in a dialect may be misrecognized.

---

## 5. Practical Recommendations for the Project

- **Unified preprocessing:** $16\ \text{kHz}$ mono, crop or pad to 1 second, then Log-Mel or MFCC features.
- **Use the provided splits:** the CSV files for Arabic and the two GSC lists for English. Do not mix speakers across sets.
- **Two models or one shared model:** start with one model per language (a comparison baseline), then train a shared model with a unified vocabulary (40 Arabic + 19 shared English, or the ten core commands) and compare.
- **Data augmentation for Arabic:** background noise, time shifting, and speed and pitch changes. The SynTTS idea (generating extra voices) is for experimentation only and requires human verification.
- **Additional classes:** add `silence` and `unknown` classes (GSC words not used as negatives for Arabic) to reduce false alarms.
- **A team-recorded test set:** record a number of new speakers (different dialects and ages) saying the same words to measure generalization, as Kucharski did with 106 speakers.
- **Evaluation:** accuracy, $F_1$, and a confusion matrix for each language, plus an analysis of similar-sounding pairs (right/left, open/close, up/down).
- **Citation and license:** both datasets are CC BY 4.0, so cite the references in the report.

---

## 6. Verify Before Starting

- [ ] Confirm the Kaggle version (v0.01 or v0.02), its size, and that it includes the list files and the noise folder.
- [ ] Confirm that all Arabic files are $16\ \text{kHz}$ and 1 second long (as described in the repository) using a quick checking script.
- [ ] Check for silent or truncated clips before training.
