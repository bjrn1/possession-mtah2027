# Possession labelling from automatic tracking

More Than A Hack 2027, Initial Challenge. Team Bucuresti.

All numbers below were measured on external matches. No parameter, threshold or model was trained or tuned on the provided match, its ground truth or its video, and the submitted file comes from one frozen configuration.

## 1. Approach

The system labels each of the 199,319 frames with H, A or D in five stages.

1. Ball cleaning. Frames where the ball moves faster than 40 m/s between observations are treated as missing, and gaps are filled by linear interpolation. The threshold is a physical limit and was not fitted.
2. Per-frame features, 216 in total, in the frame of reference where the home team attacks to the right in each half. The direction is detected from the mean x position of the two teams in each half.
   - 191 context features: ball state (visibility, distance to the last observation, speed, acceleration, direction change, position), team shape (tracked players, centroid, spread, mean speed, share of slow players, share in own half, line heights, centroid speed along the attack direction), ball-player relations (nearest and second nearest player of each team, players within 1.5 m and 3 m, relative speed), restart configurations (players outside the pitch, distance to corners and centre, players in the boxes) and temporal context (centred means over 0.2, 1, 3 and 10 s, future minus past differences over 1 and 3 s).
   - 25 control and gap features computed on the raw ball, not on the filled one: time since and until the ball was last and next seen, the length of the current gap, where and how fast the ball was last and next seen and which team was nearest to it then; a control detector (a player within 1.5 m moving with the ball, relative speed under 2 m/s, for at least 0.4 s), deflections (a turn above 30 degrees at speed next to a player, without control after it) and contacts (a velocity change above 3 m/s next to a player), each summarised as the team of the last and next event and the time since and until it; soft control shares over 1, 4 and 10 s. These features carry evidence across the gaps where the ball is not tracked.
3. Four probability models, each giving the probability that the ball is dead and the probability that the away team has possession.
   - Two gradient boosting models (150 iterations, 31 leaves, fixed seeds). They share the dead-ball classifier; the team classifier of the first is trained on E2 labels, the second on E1 labels (section 3). Team classifiers weight every frame by the inverse of the length of its possession, so each possession counts once, as in the ranking metric.
   - Two temporal convolutional networks over the feature sequence at 5 frames per second (6 residual blocks with dilations 1 to 32, 64 channels, receptive field of about 50 s, three classes H, A, D), one trained on E2 and one on E1 labels. Missing values enter as zeros plus a missing indicator. Training uses PyTorch with a fixed seed and budget; the frozen weights are exported to NumPy, and the run computes the network in NumPy, so the submission needs no deep learning library.
   - The four outputs are averaged in logit space with equal weights fixed before any scoring. Models trained on E1 change possession more often, models trained on E2 less often.
4. Exact decoding. A Viterbi decoder over the states H, A, D with minimum durations (12 frames for a possession, 50 frames for a stoppage) and transition penalties (H to A and A to H: 16; into D: 8; out of D: 16), selected with leave-one-match-out nested selection on external data.
5. Kick-off rule. The second half is graded from before its kick-off (FAQ.md), while every external half starts at its kick-off. A graded half that does not open inside a kick-off line-up is set to D until the end of the last line-up found in its first 3 minutes (at least 7 tracked players per team, 90 percent of them in their own half, walking pace, held for 2 s), and only if the stretch before it is dead on average for the model or at walking pace. No frame number is used; on the external matches the rule changes no frame.

The decoding is global because the ranking metric (spellF1) matches whole possessions by temporal overlap: a possession broken in two scores at most one match and one false positive. Minimum durations prevent this fragmentation, which a fixed smoothing window cannot do for possessions of different lengths.

Credit: the control, deflection and contact detector, the use of two label sets, the temporal network members, the kick-off rule and the idea of training on copies with a moving and a parked dead ball follow ideas published by team AndaAnda in its public repository (https://github.com/JulienErbland/AndaAnda, MIT licence). The implementation here is our own and was tested on our external benchmark before adoption.

## 2. Assets used and dropped

- Tactical ATD (positions, no identities): used. The only match data read by the pipeline.
- Frame-to-ATD alignment (`video_frame = frame_id - 1`): used. Checked structurally.
- Published half boundaries (DATA.md): used. Only to separate the two halves; labels never depend on other frame numbers.
- Video, broadcast video, broadcast ATD: dropped. No external counterpart; video was not used to check the output.
- Smart Tagging: dropped. No external counterpart, so its weight could not be tuned on external data.
- Homography: dropped. Not needed: the ATD is already in pitch coordinates.
- Public 5-minute ground truth: not used. No score computed on it.
- Event data, reports or statistics about the match: not used.

Tried and dropped on external data: Kalman-RTS smoothing of the ball (diverges on long gaps), a dedicated boundary detector inside the decoder (0.664 against 0.663), dense sampling around transitions, a second gradient boosting stage over windows of the first-stage probabilities (0.676 against 0.683), temperature scaling of the team probability (changes below 0.002), a proposal and verification step for short possessions (a classifier on segment features keeps or removes short possessions proposed with a lower penalty; never better than the tuned decoder alone), and averaging each model with its prediction on the team-swapped input (about +0.002 on the first variants, below the threshold fixed in advance).

## 3. Training data

- IDSSE
  - Source: Bassek et al., Scientific Data 12, 195 (2025), https://huggingface.co/datasets/pysport/idsse-data
  - Labelled by: DFL (optical tracking, ball status and possession fields, human-collected events)
  - Volume: 7 matches of Bundesliga and 2. Bundesliga 2022-23 (J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9), two halves each, 25 fps
  - Licence: CC BY 4.0
- Synchronised IDSSE events
  - Source: Kim et al., PathCRF, https://github.com/hyunsungkim-ds/pathcrf
  - Labelled by: DFL events, aligned to frames by the PathCRF authors
  - Volume: same 7 matches
  - Licence: MPL 2.0

Two label definitions are built from them. E1 is the provider's native ball status and possession field. E2 changes possession only at a controlled action of the opposing team, taken from the synchronised events; it is the definition closest to the challenge text and the one used to select the decoder. Both are used to train ensemble members, and both are reported. None of the data comes from the provided game.

Degradation. The ATD of the competition is noisier than the IDSSE tracking, so every external match is degraded in 5 random variants (35 in total) following the published data description: ball dropouts in bursts, slot fragmentation, identity swaps, duplicates, shuffled slot order, speed recomputed from slots.

Randomization of the ball. In the clean DFL tracking the dead ball does not behave like a ball seen by a camera. Over the 686 stoppages of at least 3 s in the 7 matches, the ball is always present, stands still for the last 2 s before the restart, jumps faster than 40 m/s at 90 percent of restarts (it is moved to the restart spot), and has a player within 1.5 m on only 6 percent of dead frames. A camera-based tracker sees the real ball, which at a stoppage is carried, held or out of view. Every degraded variant therefore gets one more copy with randomized ball behaviour, drawn from wide ranges and not fitted to the provided match: ball smoothing over 0 to 25 frames, the dead ball kept, carried by the nearest player, removed in 30 to 90 percent of stoppages or a mix of these, and up to 70 percent of the live-play ball gaps refilled from the clean ball. Gradient boosting uses one frame per second of every original and randomized variant plus copies with the teams swapped; the temporal networks use every original and randomized variant and their swapped copies at 5 frames per second.

## 4. Possession rules implemented

- A possession is a maximal run of one team with the ball alive; a dead ball separates possessions.
- Possession changes only when the opposing team controls the ball. The decoder holds the last team through loose balls, contested balls and touches without control, because a change needs a sustained signal and a minimum duration of 12 frames.
- Dead-ball ownership is not labelled: the answer is D.
- Restarts have a clear possessor: after D the models decide the team from the frames that follow the restart.
- A half graded before its kick-off is D until the kick-off (section 1, stage 5).
- No frame is labelled X. Frames without tracking inside a half take the last label of that half.

## 5. Missing and noisy tracking

According to the published data description (DATA.md), the ball is missing on 41 percent of the graded frames. The cleaning step fills short gaps by interpolation; for long gaps the models rely on player features and on the gap features of stage 2, which say where the ball was last seen, who was near it and who touched it last and next. The temporal networks see about 25 s on each side of a frame. Training on degraded and randomized matches lets the models learn how far to trust each signal.

## 6. Results on external matches

Every model is trained without the evaluated match, and the decoder parameters are selected without it. All scores come from the official scorer.

Seven held-out matches, 35 degraded variants:

- official baseline, best window: spellF1 (E2) 0.238, macroF1 (E2) 0.383, spellF1 (E1) 0.199
- one gradient boosting pair: spellF1 (E2) 0.682, macroF1 (E2) 0.915, spellF1 (E1) 0.571
- four-member ensemble, trained without randomization: spellF1 (E2) 0.700, macroF1 (E2) 0.923, spellF1 (E1) 0.607

Training the ensemble with randomization takes about twice as long, so its held-out models were fitted for three matches (J03WMX, J03WOY, J03WR9). On these three matches, 15 degraded variants:

- one gradient boosting pair: spellF1 (E2) 0.682, spellF1 (E1) 0.565, macroF1 0.920
- four-member ensemble without randomization: spellF1 (E2) 0.699, spellF1 (E1) 0.599, macroF1 0.927
- four-member ensemble with randomization (submitted): spellF1 (E2) 0.702, spellF1 (E1) 0.591, macroF1 0.928

Stress tests on the first degraded variant of the same three matches, spellF1 (E2), without randomization then with it: dead ball carried by a player 0.545 and 0.709; ball smoothed over 13 frames 0.678 and 0.708; ball missing in 80 percent of stoppages and more often visible in live play 0.685 and 0.735; all of these combined 0.579 and 0.712; 10 percent of player tracks hidden, a perturbation not used in training, 0.704 and 0.704. On the first degraded variant of all seven matches, the ensemble without randomization falls from 0.705 to 0.553 when the dead ball is carried.

The randomized training lowers spellF1 (E1) on the original variants by 0.008, beyond the 0.005 tolerance fixed in advance for this comparison. It was adopted anyway: spellF1 (E2) and macroF1 do not drop, the difference is of the size of the noise on three matches, and without it the ensemble depends on a dead-ball behaviour that a camera-based tracker does not produce.

Share of reference possessions (E2) matched at IoU 0.5 by length, one gradient boosting pair then the four-member ensemble without randomization, seven matches: under 2 s 11 and 17 percent, 2-3 s 36 and 42, 3-5 s 63 and 64, 5-10 s 78 and 78, over 10 s 91 and 92.

## 7. Limitations

- Possessions under 3 s, about a quarter of all, are still matched poorly. In about two thirds of the missed ones the models lean towards the new team at some moment, but the evidence is too weak or too brief to pay for two changes; in almost half of them the ball is untracked for most of the possession.
- The challenge definition of control cannot be validated on the provided match without using its ground truth, so models are trained on both E1 and E2 and both are reported.
- The degradation and the randomization approximate the ATD; they were not fitted to it, so absolute scores on the provided match may differ from the external ones.
- The kick-off rule could not be tested on external data, where every half starts at its kick-off; it is tested on synthetic tracking and changes nothing on the external matches.
- With more time: training on a second tracking provider, boundary regression in the temporal network, and reconstruction of player identities from the anonymous slots.

## 8. Reproduction

Python 3.10, `pip install -r requirements.lock`, then `python -m possession.run --data-dir <challenge-data> --out <team_name>.csv` with `OMP_NUM_THREADS=1` and `PYTHONHASHSEED=0`. Model files are in `models`; their SHA-256 hashes in `config/frozen.json` are checked at run time. Member probabilities and their average are rounded to 1e-4 before decoding to absorb numerical differences between platforms. Retraining the temporal networks also needs PyTorch (`requirements-train.txt`).

SHA-256 of the submitted file: 1a5a105b8278b175e519e6e5000772214e78000c177196f55d4b04c2d1bf8ef8
