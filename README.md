# KOALA++: Efficient Kalman-Based Optimization of Neural Networks with Gradient-Covariance Products


This repository will host the official implementation of **KOALA++**, a scalable Kalman-based optimization algorithm for neural network training.

## Project Status
The source code is currently under preparation and will be released **after the acceptance of our paper**.  
Stay tuned for updates!

## Overview
KOALA++ introduces a Kalman-based optimization framework that explicitly models structured gradient uncertainty.  
Unlike traditional second-order methods that rely on costly Hessian computations, KOALA++ maintains efficiency by recursively updating compact gradient–covariance products.  
This approach extends the original KOALA framework, moving beyond diagonal covariance assumptions to capture richer uncertainty structures—without storing full covariance matrices or performing large matrix inversions.  

Across diverse tasks such as image classification and language modeling, KOALA++ achieves accuracy on par with or better than state-of-the-art first- and second-order optimizers, while preserving the efficiency of first-order methods.

## Citation
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

## Contact
For questions or collaboration inquiries, please contact:  
Zixuan Xia – [your email here]

---

We appreciate your interest in KOALA++ and look forward to sharing the code and results with the community.

