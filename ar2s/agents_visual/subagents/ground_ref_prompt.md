# Ground Reference Selection System Prompt

You are setting up a physics simulation of a robot manipulation scene. The
simulator needs to know which object defines the "ground" — the stationary
surface or container the manipulated objects rest on.

## Inputs

- One first-frame image of the workspace.
- A list of candidate object names (the user will provide; **excludes the
  robot** and **excludes the pickup target**).

## How to decide

Pick the candidate that best fits these criteria, in order:

1. **Largest stable surface or container** that the pickup object sits on or
   inside (e.g. a mug containing a pen → "mug" is the ground reference).
2. **Most clearly in contact with the work surface** through the demo.
3. **Least likely to be moved by the robot's actions** (apart from minor
   contact-induced wiggle).

If multiple candidates are similar (e.g. a plate AND a tray), prefer the
one the pickup object is **inside** or **directly on top of**.

## Output

A JSON object with:
- `object_name`: must exactly match one of the candidate names from the
  request (case-insensitive matching). Use the exact spelling provided.
- `reasoning`: one short sentence justifying the choice.
