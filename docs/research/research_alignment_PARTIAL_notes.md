# research:alignment — PARTIAL (workflow stopped before the agent finished)

Key tool outputs salvaged from the alignment research agent transcript (2026-10-12). Scripts it wrote are in artifacts/alignment/scripts/.

### [2] List recent torchaudio and torch releases and wheels
```
== torchaudio
2.3.0 2024-04-24T15:49:10
2.3.1 2024-06-05T16:45:07
2.4.0 2024-07-24T15:31:51
2.4.1 2024-09-04T19:16:05
2.5.0 2024-10-17T18:06:21
2.5.1 2024-10-29T17:41:14
2.6.0 2025-01-29T16:29:43
2.7.0 2025-04-23T14:47:09
2.7.1 2025-06-04T17:44:12
2.8.0 2025-08-06T14:58:32
2.9.0 2025-10-15T15:51:53
2.9.1 2025-11-12T15:25:47
2.10.0 2026-01-21T16:28:49
2.11.0 2026-03-23T18:13:46
== torch
2.5.1 2024-10-29T17:33:38
2.6.0 2025-01-29T16:19:12
2.7.0 2025-04-23T14:35:10
2.7.1 2025-06-04T17:38:48
2.8.0 2025-08-06T14:53:15
2.9.0 2025-10-15T15:45:59
2.9.1 2025-11-12T15:19:56
2.10.0 2026-02-10T21:44:59
2.11.0 2026-03-23T18:12:06
2.12.0 2026-05-13T14:55:53
2.12.1 2026-06-17T21:08:43
2.13.0 2026-07-08T16:05:49
2.14.0 2026-09-02T13:42:48
2.14.1 2026-09-30T17:43:48
None
['torchaudio-2.11.0-cp310-cp310-macosx_11_0_arm64.whl', 'torchaudio-2.11.0-cp310-cp310-manylinux_2_28_aarch64.whl', 'torchaudio-2.11.0-cp310-cp310-manylinux_2_28_x86_64.whl', 'torchaudio-2.11.0-cp310-cp310-win_amd64.whl', 'torchaudio-2.11.0-cp311-cp311-macosx_11_0_arm64.whl', 'torchaudio-2.11.0-cp311-cp311-manylinux_2_28_aarch64.whl', 'torchaudio-2.11.0-cp311-cp311-manylinux_2_28_x86_64.whl', 'torchaudio-2.11.0-cp311-cp311-win_amd64.whl', 'torchaudio-2.11.0-cp312-cp312-macosx_11_0_arm64.whl', 'torchaudio-2.11.0-cp312-cp312-manylinux_2_28_aarch64.whl', 'torchaudio-2.11.0-cp312-cp312-manylinux_2_28_x86_64.whl', 'torchaudio-2.11.0-cp312-cp312-win_amd64.whl', 'torchaudio-2.11.0-cp313-cp313-macosx_12_0_arm64.whl', 'torchaudio-2.11.0-cp313-cp313-manylinux_2_28_aarch64.whl', 'torchaudio-2.11.0-cp313-cp313-manylinux_2_28_x86_64.whl', 'torchaudio-2.11.0-cp313-cp313t-macosx_12_0_arm64.whl', 'torchaudio-2.11.0-cp313-cp313t-manylinux_2_28_aarch64.whl', 'torchaudio-2.11.0-cp313-cp313t-manylinux_2_28_x86_64.whl', 'torchaudio-2.11.0-cp313-cp313t-win_amd64.whl', 'torchaudio-2.11.0-cp313-cp313-win_amd64.whl', 'torchaudio-2.11.0-cp314-cp314-macosx_12_0_arm64.whl', 'torchaudio-2.11.0-cp314-cp314-manylinux_2_28_aarch64.whl', 'torchaudio-2.11.0-cp314-cp314-manylinux_2_28_x86_64.whl', 'torchaudio-2.11.0-cp314-cp314t-macosx_12_0_arm64.whl', 'torchaudio-2.11.0-cp314-cp314t-manylinux_2_28_aarch64.whl', 'torchaudio-2.11.0-cp314-cp314t-manylinux_2_28_x86_64.whl', 'torchaudio-2.11.0-cp314-cp314t-win_amd64.whl', 'torchaudio-2.11.0-cp314-cp314-win_amd64.whl']
```

### [3] Check pytorch/audio repo status and README
```
{"archived":false,"default_branch":"main","description":"Data manipulation and transformation for audio signal processing, powered by PyTorch","pushed_at":"2026-10-11T11:40:03Z"}
{"name":"TorchAudio 2.11","published_at":"2026-03-23T18:40:32Z","tag_name":"v2.11.0"}
{"name":"TorchAudio 2.10","published_at":"2026-01-21T17:05:17Z","tag_name":"v2.10.0"}
{"name":"TorchAudio 2.9.1 Release","published_at":"2025-11-12T20:07:48Z","tag_name":"v2.9.1"}
{"name":"TorchAudio 2.9 release","published_at":"2025-10-15T17:17:43Z","tag_name":"v2.9.0"}
{"name":"TorchAudio 2.8.0 release","published_at":"2025-08-06T17:08:37Z","tag_name":"v2.8.0"}
{"name":"TorchAudio 2.7.1 Release","published_at":"2025-06-04T18:21:13Z","tag_name":"v2.7.1"}
torchaudio: an audio library for PyTorch
========================================

[![Documentation](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchaudio%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pytorch.org/audio/main/)
[![Anaconda Badge](https://anaconda.org/pytorch/torchaudio/badges/downloads.svg)](https://anaconda.org/pytorch/torchaudio)
[![Anaconda-Server Badge](https://anaconda.org/pytorch/torchaudio/badges/platforms.svg)](https://anaconda.org/pytorch/torchaudio)

![TorchAudio Logo](docs/source/_static/img/logo.png)

> [!IMPORTANT]
> **TorchAudio 2.11 works with `torch` 2.11 and with every future `torch`
> release (2.12, 2.13, etc.).** It is the latest release, and it is the only
> one you need: there is nothing to upgrade in TorchAudio when you upgrade
> `torch`, and installing TorchAudio does not pin `torch` to a specific
> version.

> [!NOTE]
> We have transitioned TorchAudio into a maintenance phase. This migration
> process removed some user-facing features. These features were deprecated
> from TorchAudio 2.8 and removed in 2.9.
> Our main goals were to reduce redundancies with the rest of the
> PyTorch ecosystem, make it easier to maintain, and create a version of
> TorchAudio that is more tightly scoped to its strengths: processing audio
> data for ML. Please see
> [our community message](https://github.com/pytorch/audio/issues/3902)
> for more details.

The aim of torchaudio is to apply [PyTorch](https://github.com/pytorch/pytorch) to
the audio domain. By supporting PyTorch, torchaudio follows the same philosophy
of providing strong GPU acceleration, having a focus on trainable features through
the autograd system, and having consistent style (tensor names and dimension names).
Therefore, it is primarily a machine learning library and not a general signal
processing library. The benefits of PyTorch can be seen in torchaudio through
having all the computations be through PyTorch operations which makes it easy
to use and feel like a natural extension.

- [Dataloaders for common audio datasets](http://pytorch.org/audio/main/datasets.html)
- Audio and speech processing functions
  - [forced_align](https://pytorch.org/audio/main/generated/torchaudio.functional.forced_align.html)
- Common audio transforms
  - [Spectrogram, AmplitudeToDB, MelScale, MelSpectrogram, MFCC, MuLawEncoding, MuLawDecoding, Resample](http://pytorch.org/audio/main/transforms.html)
- Compliance interfaces: Run code using PyTorch that align with other libraries
  - [Kaldi: spectrogram, fbank, mfcc](https://pytorch.org/audio/main/compliance.kaldi.html)

Installation
------------

```bash
pip install torchaudio
```

TorchAudio 2.11 requires `torch` 2.11 or newer, includin
```

### [7] Check torchaudio aligner and ctc-forced-aligner repo status
```
from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torchaudio.functional as F
from torch import Tensor
from torchaudio.functional import TokenSpan


class ITokenizer(ABC):
    @abstractmethod
    def __call__(self, transcript: List[str]) -> List[List[str]]:
        """Tokenize the given transcript (list of word)

        .. note::

           The toranscript must be normalized.

        Args:
            transcript (list of str): Transcript (list of word).

        Returns:
            (list of int): List of token sequences
        """


class Tokenizer(ITokenizer):
    def __init__(self, dictionary: Dict[str, int]):
        self.dictionary = dictionary

    def __call__(self, transcript: List[str]) -> List[List[int]]:
        return [[self.dictionary[c] for c in word] for word in transcript]


def _align_emission_and_tokens(emission: Tensor, tokens: List[int], blank: int = 0):
    device = emission.device
    emission = emission.unsqueeze(0)
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)

    aligned_tokens, scores = F.forced_align(emission, targets, blank=blank)

    scores = scores.exp()  # convert back to probability
    aligned_tokens, scores = aligned_tokens[0], scores[0]  # remove batch dimension
    return aligned_tokens, scores


class IAligner(ABC):
    @abstractmethod
    def __call__(self, emission: Tensor, tokens: List[List[int]]) -> List[List[TokenSpan]]:
        """Generate list of time-stamped token sequences

        Args:
            emission (Tensor): Sequence of token probability distributions in log-domain.
                Shape: `(time, tokens)`.
            tokens (list of integer sequence): Tokenized transcript.
                Output from :py:class:`torchaudio.pipelines.Wav2Vec2FABundle.Tokenizer`.

        Returns:
            (list of TokenSpan sequence): Tokens with time stamps and scores.
        """


def _unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


def _flatten(nested_list):
    return [item for list_ in nested_list for item in list_]


class Aligner(IAligner):
    def __init__(self, blank):
        self.blank = blank

    def __call__(self, emission: Tensor, tokens: List[List[int]]) -> List[List[TokenSpan]]:
        if emission.ndim != 2:
            raise ValueError(f"The input emission must be 2D. Found: {emission.shape}")

        aligned_tokens, scores = _align_emission_and_tokens(emission, _flatten(tokens), self.blank)
        spans = F.merge_tokens(aligned_tokens, scores)
        return _unflatten(spans, [len(ts) for ts in tokens])

{"archived":false,"license":"BSD-2-Clause","open_issues_count":20,"pushed_at":"2026-09-07T13:35:26Z","stargazers_count":567}
{"date":"2026-09-07T13:35:25Z","msg":"Update license information in README.md"}
{"date":"2026-09-07T13:34:59Z","msg":"Add BSD 2-Clause License file"}
{"date":"2026-07-12T15:18:15Z","msg":"Change score calculation from sum to mean"}
{"date":"2026-07-09T14:02:34Z","msg":"Increase Alignment Accuracy"}
{"date":"2026-04-15T15:12:18Z","msg":"fix extra delimiter token"}
{"date":"2026-02-23T13:31:18Z","msg":"Replace Torchaudio with FFMPEG (#88)"}
{"date":"2025-11-26T09:43:56Z","msg":"Add torchcodec to requirements"}
{"date":"2025-08-13T12:45:46Z","msg":"fix python version (#77)"}
```

### [22] Check Qwen3-ForcedAligner HF config, license and file sizes
```
{'architectures': ['Qwen3ASRForConditionalGeneration'], 'timestamp_token_id': 151705, 'timestamp_segment_time': 80, 'model_type': 'qwen3_asr', 'support_languages': ['Chinese', 'Cantonese', 'English', 'German', 'Spanish', 'French', 'Italian', 'Portuguese', 'Russian', 'Korean', 'Japanese'], 'transformers_version': '4.57.6'}
dict_keys(['model_type', 'classify_num', 'architectures', 'audio_config', 'audio_end_token_id', 'audio_start_token_id', 'audio_token_id', 'dtype', 'initializer_range', 'text_config'])
apache-2.0 399247 ['.gitattributes', 'README.md', 'chat_template.json', 'config.json', 'generation_config.json', 'merges.txt', 'model.safetensors', 'preprocessor_config.json', 'tokenizer_config.json', 'vocab.json']
.gitattributes 1519
README.md 57456
chat_template.json 1161
config.json 5982
generation_config.json 115
merges.txt 1671853
model.safetensors 1835544544
preprocessor_config.json 330
tokenizer_config.json 12666
vocab.json 2776833
```

### [66] Verify torchaudio forced_align and MMS_FA on torch 2.14
```
2.14.1 2.11.0 mps True
forced_align cpu ok torch.Size([1, 200]) [TokenSpan(token=5, start=38, end=81, score=-3.761936664581299), TokenSpan(token=6, start=86, end=89, score=-2.7777843475341797), TokenSpan(token=7, start=91, end=114, score=-3.27968430519104)]
mps forced_align: NotImplementedError The operator 'torchaudio::forced_align' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://github.com/pytorch/pytorch/issues/1
MMS_FA 16000 29 ('-', 'a', 'i', 'e', 'n', 'o', 'u', 't', 's', 'r', 'm', 'k', 'l', 'd', 'g', 'h', 'y', 'b', 'p', 'w', 'c', 'v', 'j', 'z', 'f', "'", 'q', 'x', '*')
Albert              en_US    # Hello! My name is Albert.
Bad News            en_US    # Hello! My name is Bad News.
Bahh                en_US    # Hello! My name is Bahh.
Bells               en_US    # Hello! My name is Bells.
Boing               en_US    # Hello! My name is Boing.
Bubbles             en_US    # Hello! My name is Bubbles.
Cellos              en_US    # Hello! My name is Cellos.
Wobble              en_US    # Hello! My name is Wobble.
```

### [75] Run MMS_FA global alignment on multilingual test audio
```
{'load_s': 110.57, 'emission_s': 1.19, 'viterbi_s': 0.01, 'audio_s': 18.42, 'frames': 920, 'device': 'mps'}
[1.48-3.7] conf=0.101 빛나는 별처럼 너를 기다려
     빛@1.48(0.116) 나@1.7(0.176) 는@1.86(0.127) 별@2.1(0.056) 처@2.32(0.051) 럼@2.5(0.111) 너@2.88(0.133) 를@3.0(0.106) 기@3.24(0.081) 다@3.44(0.075) 려@3.58(0.077)
[3.88-5.14] conf=0.179 사랑해 오늘 밤에
     사@3.88(0.107) 랑@4.04(0.129) 해@4.24(0.173) 오@4.52(0.24) 늘@4.6(0.149) 밤@4.84(0.221) 에@5.0(0.231)
[8.92-11.52] conf=0.816 君の名前を呼んだ 夜空に響く声
     君@8.92(0.701) の@9.22(0.942) 名前@9.38(0.907) を@9.8(0.948) 呼ん@9.9(0.823) だ@10.16(0.955) 夜空@10.32(0.757) に@10.72(0.916) 響く@10.88(0.6) 声@11.34(0.616)
[14.64-16.86] conf=0.639 Shine on me tonight, hold me close
     Shine@14.64(0.608) on@14.98(0.637) me@15.14(0.48) tonight@15.28(0.711) hold@15.96(0.94) me@16.24(0.489) close@16.4(0.605)
      117.63 real         4.98 user         2.39 sys
          2691170304  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
              290962  page reclaims
               17229  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                  14  messages sent
              177113  messages received
                   0  signals received
               12401  voluntary context switches
               46591  involuntary context switches
         80998389188  instructions retired
         30017207865  cycles elapsed
          3166160384  peak memory footprint
```

### [76] Benchmark MMS_FA emission on MPS vs CPU for 4-min audio
```
mps emission 240s audio: 2.78 s torch.Size([11974, 29])
cpu emission 240s audio: 14.34 s torch.Size([11974, 29])
viterbi 2500 tokens x 11974 frames: 0.067 s
```

### [78] Run Qwen3-ForcedAligner with syllable units on test audio
```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights:   0%|          | 0/708 [00:00<?, ?it/s]Loading weights:  40%|███▉      | 283/708 [00:00<00:00, 2808.40it/s]Loading weights:  80%|███████▉  | 564/708 [00:00<00:00, 788.73it/s] Loading weights: 100%|██████████| 708/708 [00:00<00:00, 1107.59it/s]
{'load_s': 152.8, 'run_s': 1.29, 'device': 'mps', 'n_units': 35}

LINE 0: 빛나는 별처럼 너를 기다려
  빛@1.52-1.6(0.436)  나@1.6-1.76(0.621)  는@1.84-2.08(0.529)  별@2.08-2.24(0.894)  처@2.32-2.48(0.588)  럼@2.48-2.8(0.835)  너@2.88-2.96(0.633)  를@2.96-3.2(0.819)  기@3.2-3.44(0.72)  다@3.44-3.52(0.716)  려@3.52-3.84(0.831)
LINE 1: 사랑해 오늘 밤에
  사@3.84-4.0(0.714)  랑@4.0-4.24(0.865)  해@4.24-4.4(0.776)  오@4.48-4.56(0.859)  늘@4.56-4.8(0.897)  밤@4.8-5.04(0.605)  에@5.04-5.28(0.584)
LINE 2: 君の名前を呼んだ 夜空に響く声
  君@8.88-9.2(0.799)  の@9.2-9.36(0.984)  名前@9.36-9.76(0.977)  を@9.76-9.84(0.919)  呼ん@9.84-10.16(0.825)  だ@10.16-10.24(0.704)  夜空@10.24-10.72(0.714)  に@10.72-10.88(0.948)  響く@10.88-11.28(0.861)  声@11.28-11.6(0.714)
LINE 3: Shine on me tonight, hold me close
  Shine@14.56-14.96(0.802)  on@14.96-15.12(0.836)  me@15.12-15.28(0.691)  tonight@15.28-15.68(0.543)  hold@16.0-16.24(0.836)  me@16.24-16.4(0.814)  close@16.4-16.88(0.751)
```

### [79] Test Qwen3-ForcedAligner on 4-minute repeated audio
```
{'load_s': 2.0, 'run_s': 3.43, 'device': 'mps', 'n_units': 455}
line-start err (s): median 3.632 max 36.777 n 52
       14.96 real         8.45 user         3.39 sys
          6132727808  maximum resident set size
          9827961280  peak memory footprint
```

### [80] Run MMS_FA on same 4-minute audio for comparison
```
{'load_s': 0.87, 'emission_s': 2.7, 'viterbi_s': 0.056, 'audio_s': 239.49, 'frames': 11974, 'device': 'mps'}
MMS line-start err (s): median 0.027 max 3.058
per-line err [-0.02, 0.02, 0.02, 0.04, -0.0, 0.02, 0.02, 0.06, -0.0, 0.02, 0.04, 0.06, -0.01, 0.01, 0.03, 0.05, -0.01, 0.03, 0.03, 0.05, -0.01, 0.03, 0.03, 0.05, -0.01, 0.03, 0.03, 0.05, -3.06, 0.74, 0.02, 0.04, -3.06, 0.02, 0.02, 0.04, -0.02, 0.02, 0.02, 0.06, -0.0, 0.02, 0.04, 0.06, -0.01, 0.01, 0.03, 0.05, -0.01, 0.97, 0.03, 0.05]
conf [0.63, 0.723, 0.817, 0.428, 0.608, 0.735, 0.82, 0.679, 0.467, 0.526, 0.098, 0.682]
```

### [81] Test Qwen aligner error vs audio length 60/120/240s
```
60 s: {'load_s': 1.8, 'run_s': 0.83, 'device': 'mps', 'n_units': 116} median 0.02 max 0.1
    [0.1, -0.02, -0.02, -0.04, -0.0, 0.04, 0.04, 0.02, -0.02, 0.02, 0.02, -0.0, 0.03]
Output audio/long240.wav same as Input #0 - exiting
Error opening output file audio/long240.wav.
Error opening output files: Invalid argument
120 s: {'load_s': 2.7, 'run_s': 1.2, 'device': 'mps', 'n_units': 228} median 0.032 max 2.99
    [0.02, -0.02, -0.02, -0.04, -0.0, 0.04, 0.04, 0.02, -0.02, 0.02, 0.02, -0.0, -0.93, -0.01, -2.81, -2.99, 0.01, -0.03, 0.05, 2.97, -0.93, -2.45, -2.22, -2.34, -0.03, 0.01]
240 s: {'load_s': 3.2, 'run_s': 2.6, 'device': 'mps', 'n_units': 455} median 3.632 max 36.78
    [0.02, -0.02, -0.02, -0.04, -0.0, 0.04, 0.04, 0.02, -0.02, 0.02, 0.02, -0.0, -2.41, -0.01, -2.97, -3.07, 0.01, 2.53, 0.05, 3.15, 2.15, 0.17, -0.72, 9.32, 4.05, 1.77, -3.21, -4.7, -4.7, -3.14, -8.18, -13.88, -19.2, -20.36, -20.28, -18.46, 33.95, 35.59, 35.11, 35.28, 35.24, 33.36, 35.9, 36.78, 31.45, 29.09, 24.05, 18.35, 17.83, 17.15, 12.11, 6.41]
```

### [82] Inspect MMS_FA outlier lines unit spans
```
27 Shine on me tonight, hold me close 0.757
    [('Shine', 125.18, 125.5, 0.695), ('on', 125.52, 125.62, 0.66), ('me', 125.66, 125.74, 0.648), ('tonight', 125.82, 126.22, 0.789), ('hold', 126.5, 126.74, 0.994)]
28 빛나는 별처럼 너를 기다려 0.051
    [('빛', 127.4, 130.6, 0.001), ('나', 130.66, 132.6, 0.003), ('는', 132.6, 132.68, 0.005), ('별', 132.68, 132.78, 0.007), ('처', 132.78, 132.84, 0.006)]
29 사랑해 오늘 밤에 0.091
    [('사', 133.56, 133.6, 0.006), ('랑', 133.6, 133.76, 0.034), ('해', 133.8, 133.86, 0.026), ('오', 133.86, 133.88, 0.121), ('늘', 133.88, 133.98, 0.008)]
32 빛나는 별처럼 너를 기다려 0.239
    [('빛', 145.82, 145.88, 0.001), ('나', 145.9, 145.94, 0.0), ('는', 145.94, 146.16, 0.001), ('별', 148.88, 149.64, 0.0), ('처', 150.06, 150.2, 0.002)]
49 사랑해 오늘 밤에 0.007
    [('사', 225.9, 225.94, 0.003), ('랑', 225.94, 226.1, 0.024), ('해', 226.14, 226.2, 0.011), ('오', 226.2, 226.22, 0.003), ('늘', 226.22, 226.32, 0.006)]
```

### [84] Compare MMS_FA vs native Korean CTC on Korean test audio
```
Exit code 1
Argument 'ko_long.wav' provided as input filename, but 'kogap.wav' was already specified.
```

### [85] Compare MMS_FA vs native Korean CTC on Korean test audio
```
7.343688
60.249500
MMS_FA uroman+g2pk {'load_s': 1.85, 'emission_s': 3.2, 'viterbi_s': 0.011, 'audio_s': 60.25, 'frames': 3012, 'device': 'mps'} median 0.018 max 0.04 conf [0.716, 0.596, 0.696, 0.682, 0.693, 0.697]
kresnik native {'total_s': 294.1, 'align_s': 3.41, 'frames': 3012} median 0.059 max 0.07 conf [0.149, 0.503, 0.063, 0.514, 0.113, 0.382]
kresnik units line0 [('빛', 1.52), ('나', 1.72), ('는', 1.88), ('별', 2.14), ('처', 2.36), ('럼', 2.52), ('너', 2.92), ('를', 3.02), ('기', 3.26), ('다', 3.46), ('려', 3.6)]
```

### [86] Install ctc-forced-aligner from git and test C++ extension
```
Resolved 42 packages in 3.73s
Installed 2 packages in 16ms
 + ctc-forced-aligner==0.3.0 (from git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git@64293cc6d711e57666c4a8b098e9fd93b381fd88)
 + unidecode==1.4.0
version 0.3.0
cpp forced_align ok (1, 300) (300,)
['<star>', 'j u n', '<star>', 'n o', '<star>', 'm i n g', '<star>', 'q i a n', '<star>', 'o', '<star>', 'h u']
['<star>', '君', '<star>', 'の', '<star>', '名', '<star>', '前', '<star>', 'を', '<star>', '呼']
['<star>', 's a r a n g h a e', '<star>', 'o n e u l', '<star>', 'b a m e']
['<star>', '사랑해', '<star>', '오늘', '<star>', '밤에']
```

### [87] Compare ctc-forced-aligner PyPI vs git versions
```
1.0.2 ['requests', 'librosa>=0.10.2.post1', 'numpy', 'onnxruntime', 'onnxruntime-gpu; extra == "gpu"', 'torch; extra == "torch"', 'torchaudio; extra == "torch"', 'torch; extra == "all"', 'torchaudio; extra == "all"', 'onnxruntime-gpu; extra == "all"'] None
[('1.0.1', '2025-02-07'), ('1.0.2', '2025-02-09')]
v0.3.0
v0.2
v0.1
```

### [98] Evaluate aligners on 4 JamendoLyrics English songs
```
Command did not complete within its 600s timeout and was moved to the background (ID: beubbwqbj). Output is being written to: /private/tmp/claude-501/-Users-sunjaewon-projects-Performance-live-lyrics-tracking/a1551f14-1847-4d65-a2d8-137d02c33533/tasks/beubbwqbj.output. If it exits while you are still working you will be notified, but it is terminated when you give your final response and no notification can follow that — so do not end your turn to wait for it; if you need its result, wait for it before giving your final response. To check interim output, use Read on that file path.
Session cwd remains /Users/sunjaewon/projects/Performance/live-lyrics-tracking; directory changes made by the backgrounded command do not apply to subsequent commands.
```

### [100] Wait and check evaluation output
```
<tool_use_error>Blocked: sleep 60 followed by: cat /private/tmp/claude-501/-Users-sunjaewon-projects-Performance-live-lyrics-tracking/a1551f14-1847-4d65-a2d8-137d02c33533/tasks/beubbwqbj.output. To wait for a condition, use Monitor with an until-loop (e.g. `until <check>; do sleep 2; done`). To wait for a command you started, use run_in_background: true. Do not chain shorter sleeps to work around this block.</tool_use_error>
```

### [104] Print per-line errors for failing songs
```
Avercage_-_Embers
  L00 gt=  32.45 pred=   4.94 err= -27.51 conf=0.072 | through days of thunders
  L01 gt=  34.26 pred=  11.00 err= -23.26 conf=0.067 | continuing stolid smiles to wear
  L02 gt=  40.55 pred=  15.36 err= -25.19 conf=0.044 | we're going on
  L03 gt=  43.04 pred=  17.18 err= -25.86 conf=0.059 | we nearly carried it to the end
  L04 gt=  47.32 pred=  21.64 err= -25.68 conf=0.106 | and every whisper
  L05 gt=  49.06 pred=  23.52 err= -25.54 conf=0.067 | and every beat of windy heart
  L06 gt=  55.30 pred=  26.48 err= -28.82 conf=0.044 | we're holding on
  L07 gt=  57.76 pred=  27.70 err= -30.06 conf=0.030 | for the moment when we unite
  L08 gt=  64.51 pred=  29.74 err= -34.77 conf=0.141 | no face
  L09 gt=  66.06 pred=  33.08 err= -32.98 conf=0.231 | is seen at dusk of a raining day
  L10 gt=  69.67 pred=  36.90 err= -32.77 conf=0.132 | no name
  L11 gt=  71.66 pred=  40.72 err= -30.94 conf=0.246 | is spoken in a loud voice
  L12 gt=  75.49 pred=  45.04 err= -30.45 conf=0.287 | still still
  L13 gt=  77.37 pred=  47.60 err= -29.77 conf=0.198 | one day it all
  L14 gt=  80.36 pred=  49.18 err= -31.18 conf=0.159 | will gone forever will gone forever
  L15 gt=  84.77 pred=  58.42 err= -26.35 conf=0.072 | don't close your eyes
  L16 gt=  87.81 pred=  77.52 err= -10.29 conf=0.061 | you'll gone forever you'll gone forever
  L17 gt=  95.15 pred=  82.34 err= -12.81 conf=0.037 | just reach the border
  L18 gt=  96.91 pred=  84.92 err= -11.99 conf=0.032 | to throw your past and go ahead
  L19 gt= 103.25 pred=  89.78 err= -13.47 conf=0.037 | hey light your heart
  L20 gt= 105.57 pred=  95.30 err= -10.27 conf=0.053 | as we're led by a winding thread
  L21 gt= 112.47 pred=  97.66 err= -14.81 conf=0.100 | no face
  L22 gt= 114.11 pred=  99.58 err= -14.53 conf=0.071 | is seen at dusk of a raining day
  L23 gt= 117.51 pred= 108.42 err=  -9.09 conf=0.071 | no name
  L24 gt= 119.67 pred= 112.10 err=  -7.57 conf=0.038 | is spoken with the loud voice
  L25 gt= 123.63 pred= 114.72 err=  -8.91 conf=0.074 | still still
  L26 gt= 125.40 pred= 116.36 err=  -9.04 conf=0.088 | one day it all
  L27 gt= 128.38 pred= 118.18 err= -10.20 conf=0.143 | will gone forever will gone forever
  L28 gt= 132.82 pred= 132.90 err=   0.08 conf=0.168 | don't close your eyes
  L29 gt= 135.83 pred= 135.58 err=  -0.25 conf=0.071 | you'll gone forever you'll gone forever
  L30 gt= 169.21 pred= 170.14 err=   0.93 conf=0.084 | and we are one so i promise
  L31 gt= 189.26 pred= 181.98 err=  -7.28 conf=0.070 | one day it all
  L32 gt= 193.35 pred= 189.38 err=  -3.97 conf=0.067 | will gone forever will gone forever
  L33 gt= 199.67 pred= 199.68 err=   0.01 conf=0.083 | don't close your eyes
  L34 gt= 203.45 pred= 201.16 err=  -2.29 conf=0.055 | you'll gone forever you'll gone forever
  L35 gt= 209.45 pred= 209.52 err=   0.07 conf=0.070 | there's no more pain
  L36 gt= 213.67 pred= 213.54 err=  -0.13 conf=0.201 | which lasts forever
  L37 gt= 216.13 pred= 216.40 err=   0.27 conf=0.174 | and never severs
  L38 gt= 219.67 pred= 220.10 err=   0.43 conf=0.236 | once in the end
  L39 gt= 223.74 pred= 223.86 err=   0.12 conf=0.251 | the flame was rendered
  L40 gt= 226.15 pred= 226.30 err=   0.15 conf=0.233 | but living embers
  L41 gt= 228.92 pred= 240.48 err=  11.56 conf=0.020 | will burn
Rxbyn_-_Bad_Side
  L00 gt=   8.76 pred=   8.86 err=   0.10 conf=0.414 | one two three
  L01 gt=  10.27 pred=  10.40 err=   0.13 conf=0.513 | see you looking at me with those eyes
  L02 gt=  17.38
```

### [105] Test vocal-activity gating and star penalty on MMS alignment
```
Command running in background with ID: bv4rfqskq. Output is being written to: /private/tmp/claude-501/-Users-sunjaewon-projects-Performance-live-lyrics-tracking/a1551f14-1847-4d65-a2d8-137d02c33533/tasks/bv4rfqskq.output. If it exits while you are still working you will be notified, but it is terminated when you give your final response and no notification can follow that — so do not end your turn to wait for it; if you need its result, wait for it before giving your final response. To check interim output, use Read on that file path.
Session cwd remains /Users/sunjaewon/projects/Performance/live-lyrics-tracking; directory changes made by the backgrounded command do not apply to subsequent commands.
```

### [106] Check licenses of text-processing packages
```
pyopenjtalk-plus 0.4.1.post9 |  | MIT | []
fugashi 1.5.2 |  | MIT AND BSD-3-Clause | []
unidic-lite 1.0.8 |  | None | ['License :: OSI Approved :: MIT License']
g2pk2 0.0.3 | Apache License 2.0 | None | ['License :: OSI Approved :: Apache Software License']
python-mecab-ko 1.3.7 | BSD 3-Clause License | None | ['License :: OSI Approved :: BSD License']
soynlp 0.0.493 |  | None | ['License :: OSI Approved :: GNU General Public License v3 (GPLv3)']
nagisa 0.3.0 | MIT License | None | ['License :: OSI Approved :: MIT License']
uroman 1.3.1.1 | Copyright (C) 2015-2020 Ulf Hermjakob, USC Information Scien | None | ['License :: OSI Approved :: Apache Software License']
pyphen 0.18.1 |  | None | ['License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)', 'License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)']
jamo 0.4.1 | http://www.apache.org/licenses/LICENSE-2.0 | None | ['License :: OSI Approved :: Apache Software License']
cutlet 0.5.2 |  | None | ['License :: OSI Approved :: MIT License']
demucs 4.1.0 | MIT License | None | ['License :: OSI Approved :: MIT License']
audio-separator 0.47.0 | MIT | None | ['License :: OSI Approved :: MIT License']
silero-vad 6.2.3 |  | None | ['License :: OSI Approved :: MIT License']
```

### [114] Wait for gating eval to finish and print
```
HILA_-_Give_Me_the_Sam nogate star_pen 0.0 {'AAE': 0.163, 'med': 0.058, 'PCO03': 0.922, 'n': 322}
HILA_-_Give_Me_the_Sam gate star_pen 0.0 {'AAE': 0.163, 'med': 0.058, 'PCO03': 0.922, 'n': 322}
HILA_-_Give_Me_the_Sam gate star_pen 1.0 {'AAE': 0.163, 'med': 0.058, 'PCO03': 0.922, 'n': 322}
HILA_-_Give_Me_the_Sam gate star_pen 3.0 {'AAE': 0.163, 'med': 0.058, 'PCO03': 0.922, 'n': 322}
Avercage_-_Embers nogate star_pen 0.0 {'AAE': 14.672, 'med': 11.138, 'PCO03': 0.164, 'n': 189}
Avercage_-_Embers gate star_pen 0.0 {'AAE': 1.981, 'med': 0.237, 'PCO03': 0.561, 'n': 189}
Avercage_-_Embers gate star_pen 1.0 {'AAE': 2.137, 'med': 0.252, 'PCO03': 0.545, 'n': 189}
Avercage_-_Embers gate star_pen 3.0 {'AAE': 2.137, 'med': 0.252, 'PCO03': 0.545, 'n': 189}
Rxbyn_-_Bad_Side nogate star_pen 0.0 {'AAE': 5.444, 'med': 0.082, 'PCO03': 0.636, 'n': 440}
Rxbyn_-_Bad_Side gate star_pen 0.0 {'AAE': 2.092, 'med': 0.069, 'PCO03': 0.734, 'n': 440}
Rxbyn_-_Bad_Side gate star_pen 1.0 {'AAE': 2.096, 'med': 0.069, 'PCO03': 0.732, 'n': 440}
Rxbyn_-_Bad_Side gate star_pen 3.0 {'AAE': 2.096, 'med': 0.069, 'PCO03': 0.732, 'n': 440}
Cortez_-_Feel__Strippe nogate star_pen 0.0 {'AAE': 0.166, 'med': 0.063, 'PCO03': 0.932, 'n': 355}
Cortez_-_Feel__Strippe gate star_pen 0.0 {'AAE': 0.166, 'med': 0.063, 'PCO03': 0.932, 'n': 355}
Cortez_-_Feel__Strippe gate star_pen 1.0 {'AAE': 0.167, 'med': 0.063, 'PCO03': 0.932, 'n': 355}
Cortez_-_Feel__Strippe gate star_pen 3.0 {'AAE': 0.167, 'med': 0.063, 'PCO03': 0.932, 'n': 355}

[exited with code 0]
```

### [115] Evaluate gated MMS + Qwen hybrid and agreement-based confidence
```
Command running in background with ID: blxu24rmu. Output is being written to: /private/tmp/claude-501/-Users-sunjaewon-projects-Performance-live-lyrics-tracking/a1551f14-1847-4d65-a2d8-137d02c33533/tasks/blxu24rmu.output. If it exits while you are still working you will be notified, but it is terminated when you give your final response and no notification can follow that — so do not end your turn to wait for it; if you need its result, wait for it before giving your final response. To check interim output, use Read on that file path.
Session cwd remains /Users/sunjaewon/projects/Performance/live-lyrics-tracking; directory changes made by the backgrounded command do not apply to subsequent commands.
```

### [117] Re-run hybrid evaluation
```
Command running in background with ID: bj8zsr0p1. Output is being written to: /private/tmp/claude-501/-Users-sunjaewon-projects-Performance-live-lyrics-tracking/a1551f14-1847-4d65-a2d8-137d02c33533/tasks/bj8zsr0p1.output. If it exits while you are still working you will be notified, but it is terminated when you give your final response and no notification can follow that — so do not end your turn to wait for it; if you need its result, wait for it before giving your final response. To check interim output, use Read on that file path.
Session cwd remains /Users/sunjaewon/projects/Performance/live-lyrics-tracking; directory changes made by the backgrounded command do not apply to subsequent commands.
```

### [118] Wait for hybrid eval to finish
```
The user doesn't want to proceed with this tool use. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). STOP what you are doing and wait for the user to tell you how to proceed.
```
### JamendoLyrics English 4-song eval (beubbwqbj) — word-start errors, seconds
```

== HILA_-_Give_Me_the_Same: 58 lines, 322 words (gt 322)
  MMS_FA global  mix    {'AAE': 0.164, 'med': 0.04, 'PCO03': 0.901, 'n': 322} 32.5s
  MMS_FA global  vocals {'AAE': 0.163, 'med': 0.058, 'PCO03': 0.922, 'n': 322} 33.5s
  Qwen3FA win30  vocals {'AAE': 0.187, 'med': 0.055, 'PCO03': 0.882, 'n': 322} 50.1s
  Qwen3FA whole  vocals {'AAE': 0.234, 'med': 0.075, 'PCO03': 0.795, 'n': 322} 17.2s
  lines with max word err >1s: 4/58; flagged(conf<0.10): 3; caught 2

== Avercage_-_Embers: 42 lines, 189 words (gt 189)
  MMS_FA global  mix    {'AAE': 21.023, 'med': 22.555, 'PCO03': 0.132, 'n': 189} 25.0s
  MMS_FA global  vocals {'AAE': 14.672, 'med': 11.138, 'PCO03': 0.164, 'n': 189} 26.1s
  Qwen3FA win30  vocals {'AAE': 11.729, 'med': 9.894, 'PCO03': 0.132, 'n': 189} 34.5s
  Qwen3FA whole  vocals {'AAE': 23.83, 'med': 3.272, 'PCO03': 0.265, 'n': 189} 22.3s
  lines with max word err >1s: 34/42; flagged(conf<0.10): 27; caught 25

== Rxbyn_-_Bad_Side: 72 lines, 440 words (gt 440)
  MMS_FA global  mix    {'AAE': 4.169, 'med': 0.057, 'PCO03': 0.72, 'n': 440} 22.7s
  MMS_FA global  vocals {'AAE': 5.444, 'med': 0.082, 'PCO03': 0.636, 'n': 440} 25.3s
  Qwen3FA win30  vocals {'AAE': 3.808, 'med': 0.076, 'PCO03': 0.643, 'n': 440} 25.4s
  Qwen3FA whole  vocals {'AAE': 0.469, 'med': 0.068, 'PCO03': 0.777, 'n': 440} 26.2s
  lines with max word err >1s: 23/72; flagged(conf<0.10): 24; caught 18

== Cortez_-_Feel__Stripped_: 42 lines, 355 words (gt 355)
  MMS_FA global  mix    {'AAE': 0.276, 'med': 0.047, 'PCO03': 0.899, 'n': 355} 40.2s
  MMS_FA global  vocals {'AAE': 0.166, 'med': 0.063, 'PCO03': 0.932, 'n': 355} 46.0s
  Qwen3FA win30  vocals {'AAE': 0.07, 'med': 0.043, 'PCO03': 0.977, 'n': 355} 20.4s
  Qwen3FA whole  vocals {'AAE': 0.936, 'med': 0.079, 'PCO03': 0.727, 'n': 355} 32.0s
  lines with max word err >1s: 4/42; flagged(conf<0.10): 3; caught 2

[exited with code 0]
```

### Hybrid eval (bj8zsr0p1, killed partway)
```
HILA_-_Give_Me_the_S MMSgate {'AAE': 0.163, 'med': 0.058, 'PCO03': 0.922, 'n': 322} | Qwin {'AAE': 0.187, 'med': 0.055, 'PCO03': 0.882, 'n': 322} | Qwhole {'AAE': 0.234, 'med': 0.075, 'PCO03': 0.795, 'n': 322} | hybrid {'AAE': 0.14, 'med': 0.056, 'PCO03': 0.938, 'n': 322}
   disagree>0.5s words: 44 of 322 ; of MMS errors>1s: 14 caught by disagreement: 13
Avercage_-_Embers MMSgate {'AAE': 1.981, 'med': 0.237, 'PCO03': 0.561, 'n': 189} | Qwin {'AAE': 1.861, 'med': 0.518, 'PCO03': 0.429, 'n': 189} | Qwhole {'AAE': 23.83, 'med': 3.272, 'PCO03': 0.265, 'n': 189} | hybrid {'AAE': 1.587, 'med': 0.247, 'PCO03': 0.566, 'n': 189}

[killed]
```
