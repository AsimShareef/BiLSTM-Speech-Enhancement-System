# Dataset: VoiceBank-DEMAND

The model trains and is evaluated on the **VoiceBank-DEMAND** corpus
(Valentini-Botinhao et al.), 28-speaker version:

| Split | Pairs | Layout |
|---|---|---|
| train | 11 572 | `speech/clean_trainset_wav/` + `speech/noisy_trainset_wav/` |
| test  | 824    | `speech/clean_testset_wav/`  + `speech/noisy_testset_wav/`  |

Files are matched by identical basename (`p226_001.wav`, ...), 16-bit PCM.
Total size ~2.6 GB, so it is **not** committed (`.gitignore`).

## Get it

Official release (University of Edinburgh DataShare, DOI 10.7488/ds/2117):
<https://datashare.ed.ac.uk/handle/10283/2791>

Download and arrange as:

```
speech/
  clean_trainset_wav/   noisy_trainset_wav/
  clean_testset_wav/    noisy_testset_wav/
```

The original `trainset_28spk_wav` archives may need resampling to 16 kHz; the
28-speaker "wav" packages on DataShare are already 16 kHz.

## For Colab training

```bash
cd <repo>
zip -r voicebank_demand.zip \
    speech/clean_trainset_wav speech/noisy_trainset_wav \
    speech/clean_testset_wav  speech/noisy_testset_wav
```

Upload `voicebank_demand.zip` to `MyDrive/`, then run `notebooks/train_colab.ipynb`.
