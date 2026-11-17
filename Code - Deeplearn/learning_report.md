# What I Learned Building the MNIST Digit Classifier

Personal notes from working through `train.py` / `train2.py` / `test.py` — a
2-layer neural net written from scratch with numpy, trained on the MNIST
digit dataset (`train.csv`, `test.csv`).

## 1. Data cleaning is a guard rail, not learning

Before any training happens, pixel values need to be valid (0-255). The
cleaning step checks each row for out-of-range or missing pixels:

- if a row is mostly bad (>5% of its pixels), drop the row entirely
- otherwise just clip the bad pixels back into [0,255]

This is pure data hygiene. It has nothing to do with the model "learning" —
it happens before the model even sees the data. Mixed this up at first with
the model's internal filtering (ReLU), but they solve different problems:
guard rail = "is this input trustworthy", ReLU = "is this neuron's signal
worth passing on".

## 2. Weight and bias

- **Weight** = how much each input (pixel) matters to a neuron. Big weight =
  that pixel strongly influences whether the neuron fires. Near-zero weight
  = the neuron ignores it. Training is mostly about reshaping these.
- **Bias** = a flat offset added to the neuron's score before the
  activation judges it. It's not tied to any one input — it just shifts how
  easy or hard it is for the neuron to fire overall. Without bias every
  neuron is forced to trigger exactly at `W·X = 0`; bias lets each neuron
  pick its own trigger point.

Mental model that stuck: a neuron is an **adaptable logic gate**. A normal
logic gate (AND/OR) has a fixed, hand-built truth table. A neuron's "truth
table" (its weight + bias) starts random and gets reshaped automatically by
training until it fires on whatever pattern is actually useful.

## 3. Activation functions = the judge

- **ReLU** (hidden layer): `max(0, z)`. If the weighted score is positive,
  pass it on; if not, block it (output 0). This is also *why* a multi-layer
  network can learn curves instead of just a straight line — stacking pure
  linear layers collapses into one linear layer, and ReLU's kink is what
  breaks that collapse.
- **Softmax** (output layer): not a filter like ReLU, just squashes the
  final scores into a probability distribution (sums to 1) so it can be
  compared against the true label (`one_hot`).

Activation isn't just "passing a value along" — without it the whole
network, no matter how many layers, could only ever represent one giant
linear function.

## 4. Forward prop = the assessment pass

`Z = W·X + b` then activation, layer by layer, until a prediction (`A2`)
comes out. This is the full "look at the input and make a guess" step.

## 5. Backprop = measuring the slope, not applying it

Backprop walks the computation graph `X -> Z1 -> A1 -> Z2 -> A2 -> loss`
*backward*, using the chain rule to compute the gradient (slope) of the loss
with respect to every weight and bias. It answers "which direction is
downhill" — it does **not** move anything yet.

## 6. Gradient descent = actually applying the slope

`W -= alpha * dW` is the real learning step: nudge every weight/bias a
little in the downhill direction found by backprop. `alpha` (the learning
rate) controls how big that nudge is.

So one full training loop = forward (assess) -> backprop (measure slope) ->
gradient descent (apply the nudge), repeated every iteration/epoch.

## 7. Learning rate is a speed/stability dial

- Too low: slow, steady climb, rarely unstable, but wastes time.
- Good: fast climb early (big steps help when far from the optimum), then
  naturally needs smaller steps near the optimum.
- Too high: overshoots — accuracy oscillates or even diverges.

Watched this directly: `LEARNING_RATE = 0.03` plateaued around 93%;
`0.1` reached ~95% in the same number of epochs — 0.03 just hadn't
converged yet, it wasn't "the safe choice". There's no formula for the
right value, just iterate: if it stalls, raise it; if it wobbles, lower it.
(Possible next step: learning-rate decay — start high, shrink partway
through training — to get the fast-early benefit without the late-stage
stall.)

## 8. Batch size — splitting rows, not pixels

Easy thing to mix up: `BATCH_SIZE` splits the data by **sample (image)**,
not by feature (pixel). Every image always keeps all 784 pixels — batching
only decides how many *whole images* are grouped into one gradient step.

Each epoch: shuffle all samples, slice them into consecutive chunks of
`batch_size`, and take one gradient step per chunk (not a "swap", a single
pass through the shuffled data). Smaller batches = noisier but more
frequent updates per epoch; bigger batches = smoother but fewer updates.
There's no single "correct" batch size — 128 made sense here because the
model is tiny and there's no GPU memory pressure forcing a smaller size
(which is where numbers like 16 usually come from).

Switching from full-batch (one gradient step over all ~42,000 rows per
iteration) to mini-batch (328 steps per epoch at batch size 128) converged
dramatically faster: ~89% accuracy by epoch 20, vs. barely moving with
full-batch gradient descent.

## 9. Training accuracy isn't proof the model is good

Reading accuracy off the same data the model trained on can't detect
overfitting — the model was literally optimized to be good at predicting
exactly that data. To actually know whether it generalizes, you need a
**held-out validation split** — rows the model never saw during training.

- train accuracy high, val accuracy much lower -> overfitting
- both low -> underfitting
- both close and both reasonably high -> looks healthy

Implemented this with `train_test_split(..., test_size=0.1, stratify=y)`:
train on 90%, check on the held-out 10%.

## 10. train.csv vs test.csv aren't interchangeable

`test.csv` (Kaggle-style) has **no label column** — there's no way to score
accuracy against it locally, only generate predictions. It's not a
diagnostic tool; it's the final step, used only after the model already
looks trustworthy via the train/val check. Workflow that makes sense:

1. train the model
2. check real generalization on a held-out val split (fast, free, lets you
   iterate on learning rate / batch size / epochs)
3. only once that looks solid, run on `test.csv` and write out predictions
   (`submission.csv` in `ImageId,Label` format to match
   `sample_submission.csv`)

## Open questions / things to try next

- Learning-rate decay schedule instead of one fixed value for the whole run
- Try smaller batch sizes (16/32) and compare convergence speed/stability
- Increase hidden layer size if accuracy plateaus — 10 hidden neurons might
  be capping how much the net can represent (capacity ceiling, not a
  learning-rate problem)
- Persist the trained weights (`np.savez`) instead of retraining from
  scratch every run
