# KOALA++: Efficient Kalman-Based Optimization of Neural Networks with Gradient-Covariance Products

Abstract, _We propose KOALA++, a scalable Kalman-based optimization algorithm that explicitly models structured gradient uncertainty in neural network training. Unlike second-order methods, which rely on expensive second order gradient calculation, our method directly estimates the parameter covariance matrix by recursively updating compact gradient covariance products. This design improves upon the original KOALA framework that assumed diagonal covariance by implicitly capturing richer uncertainty structure without storing the full covariance matrix and avoiding large matrix inversions. Across diverse tasks, including image classification and language modeling, KOALA++ achieves accuracy on par or better than state-of-the-art first- and second-order optimizers while maintaining the efficiency of first-order methods._ 

<center>
<a href="https://openreview.net/group?id=NeurIPS.cc/2025/Conference/Authors&referrer=%5BHomepage%5D(%2F)" target="_blank">
    <img alt="OpenReview" src="https://img.shields.io/badge/OpenReview-KOALA++-blue?logo=openreview" height="30" />
</a>
<a href="https://arxiv.org/abs/2506.04432" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-KOALA++-red?logo=arxiv" height="30" />
<div>
    <a href="https://sumxiaa.github.io/" target="_blank">Zixuan Xia</a><sup>1,2</sup>,</span>
    <a href="https://araachie.github.io/" target="_blank">Aram Davtyan</a><sup>2</sup>, </span>
    <a href="https://www.cvg.unibe.ch/people/favaro" target="_blank">Paolo Favaro</a><sup>2</sup>,</span>
</div>
<div>
    <sup>1</sup>Work done during Master studies at the University of Bern&emsp;
    <sup>2</sup>Computer Vision Group, University of Bern&emsp;
</div>
</center>
<center>
    <img src="imgs/KOALA++.png" alt="Overview of Project" width="100%" height="280"/>
</center>


> Official implementation of **KOALA++**, a scalable Kalman-based optimization algorithm for neural network training.  
> 📢 The code will be released **after the acceptance of our paper**.

---
## 📰 Latest News
- **[2025-09]** 🎉 Our paper has been **accepted at NeurIPS 2025**!  
- 🔜 Code release is coming soon — stay tuned!

## 📌 Overview
KOALA++ introduces a **Kalman-based optimization framework** that explicitly models structured gradient uncertainty.

✨ Key highlights:
- ⚡ **No Hessian needed** – avoids expensive second-order computations.  
- 🔍 **Compact covariance updates** – recursively updates gradient–covariance products.  
- 🧩 **Beyond diagonals** – captures richer uncertainty without storing full covariance matrices.  
- 🏆 **Strong performance** – matches or surpasses state-of-the-art first- and second-order optimizers.  
- ⚙️ **Efficient** – maintains the scalability of first-order methods.  

---

## 📖 Citation
If you find this work useful, please cite our paper:

```bibtex
@misc{xia2025koalaefficientkalmanbasedoptimization,
      title={KOALA++: Efficient Kalman-Based Optimization of Neural Networks with Gradient-Covariance Products}, 
      author={Zixuan Xia and Aram Davtyan and Paolo Favaro},
      year={2025},
      eprint={2506.04432},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.04432}, 
}

```

---

## ✉️ Contact
For questions or collaboration inquiries, please reach out:  
**Zixuan Xia** — xxiazixuan824@gmail.com · zixuan.xia@students.unibe.ch 

---

⭐️ *We appreciate your interest in KOALA++ and look forward to sharing the code and results with the community.*

