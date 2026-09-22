# Directed prior-art and project-duplication search

Scope: mixed-domain TTA, sample/group/cluster-wise adaptation, expert routing, and gradient conflict/alignment. This is a bounded search over the locally archived 2026-09-15 review (40 PDFs/38 distinct works), its source audit, and the project's completed TTA records. It does not establish an exhaustive novelty opinion.

## Findings

| Axis | Directly relevant prior work/evidence | Consequence |
|---|---|---|
| Mixed/dynamic-domain TTA | SAR, *Towards Stable Test-Time Adaptation in Dynamic Wild World*, explicitly targets unreliable samples and changing/mixed test distributions. CoTTA addresses continual shifts with teacher/augmentation and stochastic restoration. ROID-style robust dynamic TTA and the archived RF Stage C random-mixed protocol occupy the same broader problem space. | Batch-composition sensitivity and mixed-stream robustness are established TTA questions, not a new claim by themselves. |
| Sample-wise adaptation/filtering | MEMO adapts on augmented views of a single sample; SAR and PASLE select or treat target samples by reliability/uncertainty; T3A maintains selected pseudo-labeled supports without backbone backprop. | A deployable per-sample sharing/filtering rule would need a concrete improvement beyond confidence/reliability selection and existing support-set methods. |
| Group/cluster-wise adaptation | ARM learns to adapt from unlabeled test groups; T3A support/prototype structure and Proteus GMM pseudo-label fine-tuning already use target structure. The archived Stage A class-clustered versus Stage C random-mixed experiments directly show group composition can change Tent behavior. | “Cluster then run Tent separately” is an obvious composition of known grouping and TTA components and must not be presented as intrinsically novel. |
| Expert/MoE routing | BECoTTA, *Input-dependent Online Blending of Experts for Continual Test-time Adaptation*, directly uses input-dependent expert routing/blending. EcoTTA uses lightweight adaptation modules with a frozen backbone. | Expert routing for adaptation is directly occupied prior-art space; a WF application alone would not establish novelty. |
| Gradient conflict/alignment | General conflict handling includes GradNorm and PCGrad. In this project, P4 initial, P5 terminal and P6 trajectory-consistent gradient-alignment variants were already executed; P6−P4 mean DeltaAG was −0.0697 pp with only 1/3 seeds improved. Static-preserving BN-affine objective routing also failed its fixed pilot. | Gradient similarity/conflict is useful as a diagnostic signal, but reweighting/projection/routing is neither untested locally nor presumptively novel. This Job did not reopen it. |
| Website-fingerprinting adaptation | Proteus combines MMD, entropy/diversity and GMM pseudo-label fine-tuning; OnlineWF performs labeled continual feature-mean updates; NetTTT performs encrypted-traffic TTT; UAF claims unlabeled shifted-target adaptation. | Any later shared-scope method must distinguish its information, isolation and temporal protocol from these works. |

## Project-level duplicate boundary

The archived Stage E evidence gate already records class-clustered positive adaptation versus random-mixed negative adaptation, and closes simple repackaging of batch composition, gradient alignment, routing, update-scale and checkpoint-selection variants. The current Job is narrower: an independent-pool, equal-budget oracle existence diagnostic on DF Day90. Its negative gate result supplies no basis to reopen those method families.

Conclusion: the mechanism neighborhood is crowded and contains direct group, sample, expert-routing and gradient-conflict analogues. No novelty claim is made, and “oracle grouping” here is only a diagnostic upper bound.
