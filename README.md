# KOALA++: Efficient Kalman-Based Optimization of Neural Networks with Gradient-Covariance Products

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

