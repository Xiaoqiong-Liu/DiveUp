<div align="center">

# 🤿 DiveUp: Learning Feature Upsampling from Diverse Vision Foundation Models

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

*A lightweight, VFM-agnostic spatial interpolation framework that preserves intrinsic geometric structures without altering the original semantic space.*

[**Project Page**](https://github.com/Xiaoqiong-Liu/DiveUp) | [**Paper (arXiv)**](https://arxiv.org/abs/xxxx.xxxxx) | [**Weights**](https://huggingface.co/Xiaoqiong-Liu/DiveUp)

</div>

---

> **Abstract:** Current feature upsampling relies on intra-model self-reconstruction, often overfitting to source artifacts. We introduce DiveUp, an encoder-agnostic framework that utilizes multi-VFM relational guidance to break single-model dependency. By employing a universal local center-of-mass (COM) field and a spikiness-aware selection strategy, DiveUp aggregates structural consensus from diverse VFMs. This jointly-trained model achieves state-of-the-art zero-shot upsampling across diverse spaces like SigLIP and DINOv2 without per-model retraining.

<div align="center">
  <img src="teaser.png" alt="DiveUp Teaser" width="90%">
  <p><em>Comparison of upsampled features from different methods. DiveUp is robust against feature noises.</em></p>
</div>


