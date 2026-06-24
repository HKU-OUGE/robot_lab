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
- student-only augmentation was tried first: `lin_vel_x=(-0.30, 0.85)`;
- later teacher-side training also moved to `lin_vel_x=(-0.30, 0.85)`;
- with final command multiplier `0.75`, this gives mild backward coverage around `-0.225 m/s`.

Do not assume continuing the same student run will fix OOD backward stability. Add the distribution first; retrain teacher only if the student cannot inherit a stable fallback.

Follow-up lesson from 2026-06-18/19:

```text
The student-first fix was insufficient. If the teacher's good behavior is only a fragile/OOD fallback, the blind student may not preserve it. Put important deployment commands into teacher training first, then distill.
```

Current highstep command range is now teacher-side:

```text
lin_vel_x=(-0.30, 0.85)
```

## Box Joint Default Is A Separate Objective

Backward-stability fixes made the robot safer but produced abnormal box-joint positions on flat ground and normal motion. That is not acceptable: box joints should visibly help on high platforms, but should not drift into odd extension/compression during flat standing or normal walking.

Current fix:

```text
highstep_box_default_position_penalty()
```

This penalty:

- computes box-joint deviation from default;
- builds a highstep gate from front/rear foot geometry and terrain height;
- multiplies it by a forward-command gate;
- uses strong `hold_scale` outside active highstep and low `highstep_scale` during active forward highstep.

Lesson:

```text
Do not solve climbing by letting box joints drift everywhere. Box joints need a task gate: default on flat/non-forward, released on forward highstep.
```

## Resume Gate Rule

Fresh highstep training may keep a command terrain gate so the robot is not asked to track the full command range too early. For resume/refine runs from a mature checkpoint, that gate can block useful refinement.

Current behavior in `scripts/rsl_rl/base/train.py`:

```text
if highstep teacher and --resume:
    command_levels.params["terrain_gate_level"] = 0.0
```

Do not forget this distinction:

- fresh run: keep the gate;
- mature resume/refine: relax the gate.

## Terrain Levels Are Not Platform Height

The user was understandably frustrated because play showed the robot climbing a 0.35 m platform while `terrain_levels` stayed around 2.7-2.8. The correct interpretation is:

```text
terrain_levels = mean terrain-row index of all training envs after curriculum updates
```

It is not:

```text
maximum platform height climbed
```

Why this matters:

- highstep training mixes boxes, pits, stairs, inverse stairs, slopes, roughness, standing, backward and lateral commands;
- box terrain is only part of the distribution;
- play can intentionally select a high platform, while training metrics average many other scenes;
- bodyflat curves near 5.x are from a different task, terrain mix, command range, and curriculum function.

Rule:

```text
Use terrain_levels as a distribution/progress signal, not as proof of highstep ability. For highstep ability, use play/eval success on selected platform heights plus stability and box-joint sanity metrics.
```

## Averaged Rear-Leg Rewards Can Hide Branch Collapse

Observed on 2026-06-23 highstep teacher refinement:

- `model_97300` could climb and is useful as a comparison checkpoint, but after checking lineage it should not be the primary parent for the new reward fixes.
- `model_97300` was produced by resuming from `2026-06-22_12-41-28/model_95398.pt` for about 1900 updates under the older reward mix.
- If the reason for a new fix is "the old reward mix may have guided the policy into a bad branch", prefer the cleaner checkpoint before that interval unless video evidence overwhelmingly proves otherwise.
- `model_105397` still had a successful right-rear-first branch, confirmed by a later supplemental video, but the dominant behavior too often became the weaker left-rear-first branch.
- When the left rear foot climbed first, the other rear foot could remain trapped below the step edge; the robot might keep pushing without completing the climb.
- Successful climbs often depended on the rear foot already on the platform acting as a support/drive leg to lift the body and bring the second rear foot up.

Curve trap:

```text
rear clearance / rear drive / box push rewards can rise while the actual climb gets less reliable.
```

Reason:

```text
If rear-foot clearance is partly paid by max/single-foot success, and stuck-foot penalty is averaged over both rear feet, the reward can miss the "one foot up, one foot stuck" failure.
```

Rule:

```text
For highstep, inspect RL-first and RR-first cases separately.
Use worst-rear-foot or second-rear-foot logic for stuck/clearance terms.
Before selecting a checkpoint, check whether it already includes training under the reward logic being fixed.
Do not continue from a later checkpoint just because its scalar reward is slightly higher if video shows branch collapse.
```

## Backup Discipline

Many files in this repo are intentionally dirty and experimental. Never revert user changes. Never overwrite without backup.

Before modifying code:

```text
1. Identify relevant latest log suffix.
2. Copy original file to *_<suffix>.py.
3. Make the smallest necessary edit.
4. Record the intended run/checkpoint.
```
