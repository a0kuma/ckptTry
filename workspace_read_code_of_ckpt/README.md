# shift down line number

`0`

# note

because we use reentrant false, and CheckpointFunction is only used in use reentrant true, so line 234 don't need to consider

# read start at line 355

due to line 519 is the master fwd, ctx ctrl, aka pre and post fwd, be at line 513, aka line 1558

## because there is 2 next in

i need to know the range of each next

## result:

i have no idea, on what condition will torch is grad be FALSE, anyhow , the if statement is not triggered, thus the 2nd yield is triggered.

## any how:

so we have a generator (that is 2 yield), for the 1st half of it, it is doing the "new frame (with recompute...shit)", where the "frame" do ... the "frame" stuff, and also 

need to consider a "with" stuff

VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

while the "with" is doing the pack and unpack stuff

i need to check if it is the "store activation" stuff

line 1158( / keywords _checkpoint_hook)

### the answer is yes

## i think we will have 2 frame (diff)

1. is the fwd frame (org)
2. is the recompute frame (new)

### lets test it out 

so ... the ans is NO

it is per "CKPT segment"

## so ... i need to know 

if the pack hook in _checkpoint_hook is called only in the FWD PATH or BOTH fwd and re-fwd

(line around 1156) , use new ckpt file

