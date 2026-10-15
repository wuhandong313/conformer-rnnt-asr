# MFA 单条冒烟

脚本: phoneme_guided_separation/preprocessing/mfa_preprocess.py
默认音频:
  C:\Users\feng\data\speak-seperation-data\librimix-small\wav8k\min\train-100\s1\19-198-0001_2.216_3259-158083-0121_-2.216.wav
转写自动从:
  ...\libri\unzip\LibriSpeech\train-clean-100\19\198\19-198.trans.txt
  → 19-198-0001 THIS LITTLE WORK WAS FINISHED ...

环境:
  Python / 依赖: conda env ``convtasnet``（soundfile, numpy, panphon）
  MFA 二进制: conda env ``aligner`` (montreal-forced-aligner 3.4.1)
模型: english_us_arpa (dictionary + acoustic)
词典: phoneme_guided_separation/preprocessing/arpa_phone2id.json （去重音 ARPABET → ID，K=42）

运行:
  conda activate convtasnet
  python phoneme_guided_separation/preprocessing/mfa_preprocess.py
  # 已有 TextGrid 时:
  python phoneme_guided_separation/preprocessing/mfa_preprocess.py --skip-align --skip-download

成功后看:
  phoneme_guided_separation/preprocessing/_mfa_out/19-198-0001/*.TextGrid
  phoneme_guided_separation/preprocessing/_mfa_out/19-198-0001/phones.json
  phoneme_guided_separation/preprocessing/_mfa_out/19-198-0001/phone_ids.npy
  phoneme_guided_separation/preprocessing/_mfa_out/19-198-0001/artv.npy
  phoneme_guided_separation/preprocessing/arpa_artv.npy
