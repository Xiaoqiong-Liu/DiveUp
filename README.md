<div align="center">

# 🤿 DiveUp: Learning Feature Upsampling from Diverse Vision Foundation Models

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

*A lightweight, VFM-agnostic spatial interpolation framework that preserves intrinsic geometric structures without altering the original semantic space.*

[**Project Page**](https://github.com/Xiaoqiong-Liu/DiveUp) | [**Paper (arXiv)**](https://arxiv.org/abs/xxxx.xxxxx) | [**Weights**](https://huggingface.co/Xiaoqiong-Liu/DiveUp)

</div>

---

> **Abstract:** Recently, feature upsampling has gained increasing attention owing to its effectiveness in enhancing vision foundation models (VFMs) for pixel-level dense prediction tasks. Unlike existing methods that often struggle with high-norm artifacts or require complex decoders, **DiveUp** introduces a consensus location alignment target. By leveraging a cross-scale neighborhood attention mechanism, it fuses high-resolution spatial structures with low-resolution VFM semantics. Our approach achieves state-of-the-art zero-shot generalization across highly diverse and noisy feature spaces (e.g., SigLIP, DINOv2, PaliGemma2).

<div align="center">
  <img src="teaser.png" alt="DiveUp Teaser" width="90%">
  <p><em>Qualitative comparison of Dense Depth Estimation and Semantic Segmentation. DiveUp strictly preserves the original VFM semantics while generating remarkably crisp and accurate object boundaries.</em></p>
</div>


