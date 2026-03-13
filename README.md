<div align="center">

# 🤿 DiveUp: Learning Feature Upsampling from Diverse Vision Foundation Models

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

*A lightweight, VFM-agnostic spatial interpolation framework that preserves intrinsic geometric structures without altering the original semantic space.*

[**Project Page**](https://github.com/Xiaoqiong-Liu/DiveUp) | [**Paper (arXiv)**](https://arxiv.org/abs/xxxx.xxxxx) | [**Weights**](https://huggingface.co/Xiaoqiong-Liu/DiveUp)

</div>

---

> **Abstract:** Existing methods typically rely on high-resolution features from the same foundation model to achieve upsampling via self-reconstruction. However, relying solely on intra-model features forces the upsampler to overfit to the source model's inherent location misalignment and high-norm artifacts. To address this fundamental limitation, we propose DiveUp, a novel framework that breaks away from single-model dependency by introducing multi-VFM relational guidance. Instead of naive feature fusion, DiveUp leverages diverse VFMs as a panel of experts, utilizing their structural consensus to regularize the upsampler's learning process, effectively preventing the propagation of inaccurate spatial structures from the source model. To reconcile the unaligned feature spaces across different VFMs, we propose a universal relational feature representation, formulated as a local center-of-mass (COM) field, that extracts intrinsic geometric structures, enabling seamless cross-model interaction. Furthermore, we introduce a spikiness-aware selection strategy that evaluates the spatial reliability of each VFM, effectively filtering out high-norm artifacts to aggregate guidance from only the most reliable expert at each local region. DiveUp is a unified, encoder-agnostic framework; a jointly-trained model can universally upsample features from diverse VFMs without requiring per-model retraining. Our approach achieves state-of-the-art zero-shot generalization across highly diverse and noisy feature spaces (e.g., SigLIP, DINOv2, PaliGemma2).

<div align="center">
  <img src="teaser.png" alt="DiveUp Teaser" width="90%">
  <p><em>Comparison of upsampled features from different methods. DiveUp is robust against feature noises.</em></p>
</div>


