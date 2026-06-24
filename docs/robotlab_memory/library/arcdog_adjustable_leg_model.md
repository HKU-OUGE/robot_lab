# Arcdog Adjustable Leg Model Notes

## Active Robot Asset

Config file:

```text
source/robot_lab/robot_lab/assets/arclab.py
```

Current USD path:

```text
source/robot_lab/data/Robots/Arclab/Arcdog_adjustable_leg_short_limit/arcdog_adjustable_leg_short_limit.usd
```

## Joint Order Used In Training

Current highstep/bodyflat action order:

```text
FL_hip_joint
FR_hip_joint
RL_hip_joint
RR_hip_joint
FL_thigh_joint
FR_thigh_joint
RL_thigh_joint
RR_thigh_joint
FL_calf_joint
FR_calf_joint
RL_calf_joint
RR_calf_joint
FL_box_joint
FR_box_joint
RL_box_joint
RR_box_joint
```

## Initial Joint Positions

Current default initial pose in `arclab.py`:

```text
hip:   0.0
thigh: 0.7
calf: -1.3
box:   0.03
```

Reason for box initial value:

```text
Current box_joint range is 0.00 to 0.06, so 0.03 is the middle.
This keeps action output symmetric around the default.
```

## Box Joint Direction

Permanent convention:

```text
Smaller box_joint value -> longer leg extension.
Larger box_joint value  -> shorter leg.
```

This direction caused repeated mistakes before. Always verify it in play output and action mapping logs before trusting new rewards or priors.

Current highstep action prior range:

```text
min_box_target = 0.000
max_box_target = 0.060
```

## Action Scaling

Current highstep action scaling:

```text
box joints: 0.02
hip/thigh/calf: 0.1
raw action clip: -60 to 60
```

The physical/soft joint limits still matter; raw action clip is not the real safe range.

## Actuator Settings

From current `arclab.py`:

```text
hip:
  effort_limit = 35.0
  velocity_limit = 45.0
  stiffness = 45.0
  damping = 1.5

thigh:
  effort_limit = 35.0
  velocity_limit = 45.0
  stiffness = 50.0
  damping = 1.5

calf:
  effort_limit = 80.0
  velocity_limit = 45.0
  stiffness = 60.0
  damping = 2.0

box:
  effort_limit = 1000.0
  velocity_limit = 0.13
  stiffness = 8000.0
  damping = 100.0
  armature = 0.4
```

## Deployment Relevance

Simulation should reveal box limit problems instead of hiding them. Deployment-side clamp may be useful as safety protection, but training should still avoid relying on impossible box motion.

