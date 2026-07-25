# GAN design notes

Design decisions and ablations for drift-time synthetic-sample generation.

## Setting and constraints

The generator trains **at drift time** on whatever attack samples the recent buffer holds: between `min_attack_samples = 5` and `buffer_size = 500` rows, typically heavily skewed to one or two subtypes. Constraints that shaped the design:

- **Tiny data**: classic BCE GANs mode-collapse quickly when the discriminator can memorize the real set.
- **Tight time budget**: training happens inline with the stream; epochs and model capacity must stay small (the cost tracker reports exactly what this costs).
- **Tabular features**: ~70 numeric flow features with wildly different scales and many zero-variance columns.

## Phase C: vanilla binary GAN (reference)

The original PoC generator: G(z: 16) → 32 → x, D(x) → 32 → 1 (sigmoid), BCE, Adam lr 0.005, 30 epochs on the buffer, 200 samples per event. Kept as-is (seedable, device-aware) because it is the incumbent the proposal must improve on.

Known weakness: it is subtype-blind. CICIDS2017 subtypes (DoS variants, PortScan, Brute Force, Web, Bot, Infiltration) have structurally different feature distributions; a binary GAN trained on a mixed buffer learns their average, which corresponds to no real attack.

## Phase C_cond: conditional WGAN-GP

- **Conditioning**: subtype label embedded (dim 8), concatenated to z (generator) and x (critic). Class vocabulary grows online as new subtypes appear in buffers; networks rebuild on vocabulary growth (cheap at these sizes).
- **Loss**: Wasserstein + gradient penalty (λ = 10), n_critic = 5, Adam(lr 1e-3, betas 0.5/0.9). Rationale: with 5–500 samples, a BCE discriminator saturates and generator gradients vanish; the Wasserstein critic keeps gradients informative and GP regularizes it without weight clipping.
- **Spectral norm** on all critic layers (default on): additional Lipschitz control that measurably stabilizes small-batch training.
- **Proportional generation**: samples are allocated across subtypes by the buffer mix (largest remainder), so we never inject attacks that are not currently occurring.

## Ablations (config flags, same engine)

| Flag | Values | Question answered |
|---|---|---|
| `cgan_loss` | `wgan_gp` (default) / `bce` | Does the Wasserstein objective matter at buffer scale? |
| `cgan_spectral_norm` | true / false | Contribution of critic Lipschitz control |
| `cgan_label_smoothing` | 0.0–0.3 (BCE variant only) | Cheap alternative to WGAN for stabilizing D |
| `cgan_feature_matching` | true / false | Does matching critic-feature means reduce collapse? |

Each ablation is a config file away (`configs/phase_c_cond.yaml`); results should be compared on the per-subtype quality metrics below **and** on downstream recovery metrics — better MMD that does not move RT/AUD is not a contribution.

## Sample-quality measurement (specified, not used)

The original design called for per-subtype sample-quality metrics — multivariate
energy distance (Székely–Rizzo) and RBF-MMD², both standardized with the *real*
subset's statistics — in a `driftgan.metrics.gan_quality` module.

**These were never used in the reported experiments and the module has been
removed.** The reason is the design's own caveat: with `n_real` sometimes as low
as 5, both metrics are too high-variance to support a conclusion. The verdict on
generative augmentation was instead reached on *downstream* recovery metrics
(RT/AUD/local F1) against budget-matched controls — which is the standard this
document itself sets: "better MMD that does not move RT/AUD is not a
contribution." See [experiment 00](experiments/00-core-run.md).

## Open questions / expected failure modes

- Buffers dominated by one subtype give the conditional model few examples of the others; conditioning cannot invent structure it never saw.
- Zero-variance and heavy-tailed features may need per-feature normalization inside the GAN (currently raw); if generated samples show implausible magnitudes on a few columns, this is the first place to look.
- If C_cond fails to beat C_prime on recovery metrics, the honest conclusion is that at this buffer scale *any* oversampling is what matters — that outcome is reportable and useful.
