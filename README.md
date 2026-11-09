# Rep2Rep-Shuffle

Official implementation of shuffled repetition-to-repetition learning for noise-adaptive self-supervised MRI denoising.

## Method

For each training example, repetitions are randomly assigned to disjoint, non-empty input and target groups. The remaining repetitions may be unused. The group means form the input and independent target:

```text
x_A = mean(x[i] for i in A)
x_B = mean(x[i] for i in B),  A ∩ B = ∅
loss = mean(abs(f(x_A, sigma_A) - x_B) ** 2)
```

For a shared noise map `sigma` and `k` input repetitions, the input noise map is `sigma_A = sigma / sqrt(k)`. For repetition-specific maps, it is `sqrt(sum(sigma[i] ** 2 for i in A)) / k`. Random disjoint partitions provide multiple independent input-target pairings across training steps. This implementation uses CDLNet2D as the reconstruction network; the shuffled grouping defines the learning method.

## Training

`make_pair` samples the disjoint groups for each call. Supply an iterable of `(repetitions, sigma_map)` batches to `train_epoch`:

```python
import torch
from model import CDLNet2D
from pairing import make_pair
from training import train_epoch

model = CDLNet2D()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss = train_epoch(model, optimizer, batches, make_pair)
```

Repetitions use shape `[N, C, H, W]` with complex values. Noise maps use `[1, 1, H, W]` when shared or `[N, 1, H, W]` when repetition-specific. Data loading and preprocessing are supplied by the application.

## Inference

Pass a complex image batch and its real-valued noise map to `denoise`:

```python
from inference import denoise

estimate = denoise(model, image, sigma_map)
```

`image` has shape `[B, C, H, W]`; `sigma_map` has shape `[B, 1, H, W]` and uses the same intensity scale as the image.

## Citations

```bibtex
@article{luoRep2RepShuffle2026,
  author  = {Luo, Renqing and Janju{\v{s}}evi{\'c}, Nikola and Pei, Haoyang and Wang, Yao and Madelin, Guillaume and Feng, Li},
  title   = {{Rep2Rep-Shuffle}: Shuffled Repetition-to-Repetition Learning for Noise-Adaptive Self-Supervised {MRI} Denoising},
  journal = {Magnetic Resonance in Medicine},
  year    = {2026},
  doi     = {10.1002/mrm.70600}
}

@article{janjusevicRep2Rep2026,
  author  = {Janju{\v{s}}evi{\'c}, Nikola and Chen, Jingjia and Ginocchio, Luke and Bruno, Mary and Huang, Yuhui and Wang, Yao and Chandarana, Hersh and Feng, Li},
  title   = {Self-Supervised Noise Adaptive {MRI} Denoising via Repetition to Repetition ({Rep2Rep}) Learning},
  journal = {Magnetic Resonance in Medicine},
  volume  = {95},
  number  = {3},
  pages   = {1619--1633},
  year    = {2026},
  doi     = {10.1002/mrm.70155}
}

@article{janjusevicCDLNet2022,
  author  = {Janju{\v{s}}evi{\'c}, Nikola and Khalilian-Gourtani, Amirhossein and Wang, Yao},
  title   = {{CDLNet}: Noise-Adaptive Convolutional Dictionary Learning Network for Blind Denoising and Demosaicing},
  journal = {IEEE Open Journal of Signal Processing},
  volume  = {3},
  pages   = {196--211},
  year    = {2022},
  doi     = {10.1109/OJSP.2022.3172842}
}
```

## License

PolyForm Noncommercial 1.0.0. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
