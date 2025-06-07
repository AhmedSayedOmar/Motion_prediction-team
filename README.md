# Transformer-MixNet

### Transformer-Based Deep Neural Motion Prediction for Autonomous Racing

This project is a modified version of [MixNet](https://doi.org/10.5281/zenodo.6953977), originally developed by the TUM Autonomous Motorsport team. We extensively reworked their architecture to replace all LSTM components with a Transformer-based approach, aiming to enhance performance in motion prediction for autonomous racing.

Additionally, for overtaking scenarios, we take as inspiration real-time motion prediction and overtaking from the [Predictive Spliner](https://arxiv.org/abs/2209.12689) paper with our Transformer-based model to enable more dynamic and intelligent overtaking behavior.

This repository was developed as part of our graduation project by:
- Myron Tadros  
- Abdelrahman Eissa  
- Ahmed Omar  
- Bola Warsy  
- Omar Abousheishaa  
- Rohin Nair.

## Acknowledgments

We would like to thank the authors of the original [MixNet](https://doi.org/10.5281/zenodo.6953977) paper:

> P. Karle, F. Török, M. Geisslinger, and M. Lienkamp, "MixNet: Physics Constrained Deep Neural Motion Prediction for Autonomous Racing," *IEEE Access*, vol. 11, pp. 85914-85926, 2023. doi: [10.1109/ACCESS.2023.3303841](https://doi.org/10.1109/ACCESS.2023.3303841)

and the authors of the [Predictive Spliner](https://arxiv.org/abs/2209.12689) paper for their valuable contributions, which inspired and supported our work.
