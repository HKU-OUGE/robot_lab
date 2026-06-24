# Failure Modes And Lessons

## Terrain Curves Can Mislead

Do not optimize only for `terrain_levels`.

Observed failure:

- some runs showed better terrain curves but awful play;
- robot could look like it was progressing statistically while flat walking became crouched, twisted, or unnatural.

Rule:

```text
Every important training decision needs both curves and play/video.
```

## Avoid Overediting Terrain Curriculum

Frequent terrain-level logic changes made comparisons confusing. Terrain curriculum should be changed only when the reset distribution or task definition is clearly wrong.

For highstep, useful terrain changes included:

- adding inverse/low-approach cases;
- creating situations where the robot actually climbs from low side to high side;
- avoiding only training from high platform downward.

But terrain-level scoring itself should not be repeatedly rewritten just to make the curve look better.

## Video During Training Can Crash

Training with `--video` or frequent Isaac/Replicator video capture has caused heavy system stalls. Prefer:

- long train without video;
- separate `play.py` videos;
- short diagnostic videos only.

## Highstep Reward Direction

A useful highstep posture should resemble:

- front legs reach and land on platform;
- rear legs remain active and push;
- rear box joints extend during push;
- body advances over the obstacle.

Rewards that only make the front legs touch the top are not enough. They can create a stall where the robot cannot bring the rear body forward.

## Hip Abduction Is Not For Highstep

Sidestep used hip abduction to prevent lateral roll. Highstep needs forward climbing freedom. Hip abduction and feet-width constraints can damage highstep ability.

Rule:

```text
Do not import sidestep lateral-stability objectives into highstep unless explicitly testing a side-step-highstep hybrid.
```

## Student Distillation Is Not A Dumping Ground

Student should not be used to fix an unstable teacher. A weak or inconsistent teacher produces weaker students.

For no-prior student:

- preserve locomotion first;
- do not aggressively force action-prior behavior at the expense of raw gait;
- avoid casually opening PPO or changing trainable actor scope without a clear reason.

## Student OOD Commands Can Fail Before Teacher Does

Observed on 2026-06-18 highstep student:

- teacher could often handle keyboard backward walking even though it was not explicitly trained for negative `lin_vel_x`;
- student `model_69998.pt` reared up and triggered `bad_orientation` much more easily when commanded backward on flat ground;
- global WandB averages did not expose this strongly because backward play is a narrow out-of-distribution scenario.

Important lesson:

```text
If play/deployment will use commands outside the training command range, student distillation must include those commands even if teacher generalizes to them.
```

Current concrete case:

- original highstep command range: `lin_vel_x=(0.20, 0.85)`;
- student-only augmentation: `lin_vel_x=(-0.30, 0.85)`;
- with final command multiplier `0.75`, this gives mild backward coverage around `-0.225 m/s`.

Do not assume continuing the same student run will fix OOD backward stability. Add the distribution first; retrain teacher only if the student cannot inherit a stable fallback.

## Backup Discipline

Many files in this repo are intentionally dirty and experimental. Never revert user changes. Never overwrite without backup.

Before modifying code:

```text
1. Identify relevant latest log suffix.
2. Copy original file to *_<suffix>.py.
3. Make the smallest necessary edit.
4. Record the intended run/checkpoint.
```
