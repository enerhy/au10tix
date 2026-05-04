# Multi-Task Loss Functions

This project trains one backbone network with two classification heads:

- `gender`: 2 classes
- `race`: 5 classes

Each image produces two predictions, so training needs two task losses:

```python
loss_gender = CrossEntropyLoss(gender_logits, gender_label)
loss_race = CrossEntropyLoss(race_logits, race_label)
```

The important question is how to combine those two numbers into one final loss for backpropagation.

That final value is logged as `Combined Loss` during training:

```text
Train | Combined Loss: ... | Gender Loss: ... | Race Loss: ...
```

`Gender Loss` and `Race Loss` are the raw task cross-entropy losses. `Combined Loss` is the value actually used for `loss.backward()`.

## Why Multi-Task Loss Needs Care

If this were a single-task model, the loss would simply be one cross-entropy value. With two heads, we need one scalar objective, for example:

```text
combined_loss = gender_loss + race_loss
```

That simple sum is a good baseline, but it assumes both tasks should influence training equally. In practice, one task can be easier, noisier, more imbalanced, or simply have a different loss scale.

For example:

- Gender may become easy quickly and produce a very small loss.
- Race may remain harder because it has more classes.
- Race labels may be noisier or less visually separable.
- Class imbalance can make some classes underrepresented.

If one task dominates the gradients, the shared backbone may learn features that help that task more than the other. The loss-combination strategy decides how strongly each head pushes on the shared model.

## Class Weights vs Task Weights

There are two different kinds of weighting in this project.

### Class Weights

Class weights are used inside each individual cross-entropy loss.

They answer this question:

```text
Within this task, should rare classes count more?
```

For example, if one race class appears much less often than others, class weighting increases the penalty when the model gets that rare class wrong.

In the config:

```yaml
loss:
  class_weights: True
```

When `class_weights` is `True`, the training script computes weights from the training split only:

```python
weight = total_samples / (num_classes * samples_in_class)
```

These weights are passed into `nn.CrossEntropyLoss`.

### Task Weights

Task weights combine the gender task and race task.

They answer this question:

```text
Between gender and race, which task should influence the shared model more?
```

Task weights are only used by `FixedWeightedMultiTaskCELoss`.

In the config:

```yaml
loss:
  type: FixedWeightedMultiTaskCELoss
  class_weights: True
  task_weights:
    gender: 1.0
    race: 1.5
```

This means race loss contributes 1.5 times as much as gender loss to the final combined loss.

## Selecting a Loss Function

The loss is selected in `classification-config.yaml`:

```yaml
loss:
  type: OriginalUncertaintyMultiTaskLoss
  class_weights: True
  task_weights:
    gender: 1.0
    race: 1.0
```

Supported values:

```yaml
OriginalUncertaintyMultiTaskLoss
PaperUncertaintyMultiTaskCELoss
SummedMultiTaskCELoss
FixedWeightedMultiTaskCELoss
```

There is also a legacy alias:

```yaml
WeightedCrossEntropy
```

`WeightedCrossEntropy` is kept for old configs. It means:

```text
OriginalUncertaintyMultiTaskLoss + class_weights=True
```

## 1. SummedMultiTaskCELoss

This is the simplest multi-task loss.

Formula:

```text
combined_loss = gender_loss + race_loss
```

Implementation:

```python
return loss_gender + loss_race, loss_gender, loss_race
```

Why use it:

- It is simple and easy to reason about.
- The combined loss is always non-negative because cross-entropy is non-negative.
- It is a strong baseline.
- It is useful when you want to debug the model without dynamic task weighting.

When to use it:

- First baseline experiment.
- When negative combined losses are confusing or undesirable.
- When you want validation loss to be easier to interpret.
- When both tasks seem similarly important and similarly scaled.

What to watch:

- If one task has consistently larger loss, it may dominate training.
- Equal loss contribution does not always mean equal task quality.

Example config:

```yaml
loss:
  type: SummedMultiTaskCELoss
  class_weights: True
```

## 2. FixedWeightedMultiTaskCELoss

This is a manually weighted version of the summed loss.

Formula:

```text
combined_loss = gender_task_weight * gender_loss
              + race_task_weight * race_loss
```

Example:

```yaml
loss:
  type: FixedWeightedMultiTaskCELoss
  class_weights: True
  task_weights:
    gender: 1.0
    race: 1.5
```

This produces:

```text
combined_loss = 1.0 * gender_loss + 1.5 * race_loss
```

Why use it:

- It gives direct manual control over task importance.
- It is still easy to understand.
- It keeps the combined loss non-negative as long as task weights are non-negative.
- It can help if one task matters more for the project goal.

When to use it:

- Race performance is more important than gender performance.
- Gender learns much faster and you want race to have more influence.
- You want a controlled experiment with known task weights.
- You do not want learnable uncertainty parameters.

What to watch:

- The weights are manual hyperparameters.
- Bad task weights can hurt one task.
- You may need several runs to find useful values.

Recommended starting values:

```yaml
task_weights:
  gender: 1.0
  race: 1.0
```

If race needs more emphasis:

```yaml
task_weights:
  gender: 1.0
  race: 1.25
```

or:

```yaml
task_weights:
  gender: 1.0
  race: 1.5
```

## 3. OriginalUncertaintyMultiTaskLoss

This is the original loss wrapper that was already implemented in the project.

Class name in the code:

```python
MultiTaskLossWrapper
```

Config name:

```yaml
OriginalUncertaintyMultiTaskLoss
```

This loss is inspired by uncertainty-based multi-task learning from Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics".

The idea is that the model learns how much to trust each task during training. Instead of manually choosing task weights, the loss has trainable parameters:

```python
self.log_var_gender = nn.Parameter(torch.zeros((1,), requires_grad=True))
self.log_var_race = nn.Parameter(torch.zeros((1,), requires_grad=True))
```

These parameters are optimized together with the model.

Formula per task:

```text
task_loss_weighted = exp(-log_var) * task_loss + log_var
```

Combined:

```text
combined_loss =
    exp(-log_var_gender) * gender_loss + log_var_gender
  + exp(-log_var_race) * race_loss + log_var_race
```

### Why This Works

The term:

```text
exp(-log_var) * task_loss
```

acts like a learned task weight.

If `log_var` increases:

```text
exp(-log_var)
```

gets smaller, so that task is down-weighted.

This is useful when a task is harder, noisier, or temporarily producing much larger gradients.

But without another term, the model could cheat by making `log_var` very large. That would shrink the weighted task loss toward zero.

So the loss also adds:

```text
+ log_var
```

This penalizes making uncertainty too large. The model has to balance:

- down-weighting a difficult task
- not increasing uncertainty too much

### Why Combined Loss Can Become Negative

This loss can produce a negative `Combined Loss`. That is not automatically a bug.

For one task:

```text
weighted_loss = exp(-s) * L + s
```

where:

- `s = log_var`
- `L = raw cross-entropy loss`

The best value for `s` is near:

```text
s = log(L)
```

At that point, the weighted term is approximately:

```text
1 + log(L)
```

If the raw cross-entropy loss is less than about `0.3679`, then:

```text
1 + log(L) < 0
```

So when the model becomes very confident and the raw task losses get small, the combined uncertainty loss can go below zero.

This does not mean cross-entropy became negative. The raw `Gender Loss` and `Race Loss` should still be non-negative.

It means the combined objective includes learned log-variance terms.

Why use it:

- It automatically learns task weighting.
- It can reduce the need for manual task-weight tuning.
- It can help when one task is noisier or harder than the other.

When to use it:

- You want dynamic task balancing.
- You are comfortable interpreting `Combined Loss` as an optimization objective, not as plain CE.
- You care more about validation metrics than the absolute combined-loss value.

What to watch:

- `Combined Loss` can be negative.
- The loss is less intuitive than a summed CE loss.
- Check raw task losses and metrics, not only combined loss.

Example config:

```yaml
loss:
  type: OriginalUncertaintyMultiTaskLoss
  class_weights: True
```

## 4. PaperUncertaintyMultiTaskCELoss

This is a closer version of the uncertainty loss described in the Kendall et al. paper for classification-style losses.

Formula per task:

```text
task_loss_weighted = exp(-log_var) * task_loss + 0.5 * log_var
```

Combined:

```text
combined_loss =
    exp(-log_var_gender) * gender_loss + 0.5 * log_var_gender
  + exp(-log_var_race) * race_loss + 0.5 * log_var_race
```

The difference from `OriginalUncertaintyMultiTaskLoss` is the regularization term:

```text
Original:
+ log_var

Paper-style:
+ 0.5 * log_var
```

### Why The 0.5 Factor Exists

The paper uses uncertainty parameters based on variance. If:

```text
s = log(sigma^2)
```

then:

```text
log(sigma) = 0.5 * log(sigma^2) = 0.5 * s
```

That is why this implementation uses:

```python
0.5 * self.log_var_gender
0.5 * self.log_var_race
```

instead of:

```python
self.log_var_gender
self.log_var_race
```

Why use it:

- It is closer to the paper's uncertainty-loss form.
- It still learns task weights automatically.
- It gives you a cleaner comparison against the original implementation.

When to use it:

- You want the Kendall-style uncertainty objective.
- You want dynamic task balancing but prefer the paper-aligned regularization term.
- You are comparing uncertainty weighting methods.

What to watch:

- It can also produce negative combined loss.
- The learned weighting behavior may differ from the original project version.
- It is still not as directly interpretable as plain summed CE.

Example config:

```yaml
loss:
  type: PaperUncertaintyMultiTaskCELoss
  class_weights: True
```

## Which Loss Should I Choose?

Start with `SummedMultiTaskCELoss` if you want the cleanest baseline.

Use `FixedWeightedMultiTaskCELoss` if you want direct control over task importance.

Use `OriginalUncertaintyMultiTaskLoss` if you want to preserve the behavior of the original project.

Use `PaperUncertaintyMultiTaskCELoss` if you want uncertainty weighting that is closer to the Kendall et al. formulation.

## Recommended Experiment Order

For a practical comparison, run these in order:

### 1. Summed baseline

```yaml
loss:
  type: SummedMultiTaskCELoss
  class_weights: True
```

This gives a simple, stable reference.

### 2. Fixed weighted race emphasis

```yaml
loss:
  type: FixedWeightedMultiTaskCELoss
  class_weights: True
  task_weights:
    gender: 1.0
    race: 1.25
```

Use this if race macro F1 is lagging behind.

### 3. Paper uncertainty

```yaml
loss:
  type: PaperUncertaintyMultiTaskCELoss
  class_weights: True
```

Use this to test automatic task balancing.

### 4. Original uncertainty

```yaml
loss:
  type: OriginalUncertaintyMultiTaskLoss
  class_weights: True
```

Use this to compare against the original project behavior.

## How To Compare Runs

Do not compare only `Combined Loss` across different loss types.

Different loss functions produce combined losses with different meanings and scales. For example:

- `SummedMultiTaskCELoss` is a plain CE sum.
- `FixedWeightedMultiTaskCELoss` depends on manual task weights.
- Uncertainty losses include learned `log_var` terms and can become negative.

Instead, compare:

- validation gender accuracy
- validation race accuracy
- validation gender macro F1
- validation race macro F1
- raw `Gender Loss`
- raw `Race Loss`
- test metrics from the best validation checkpoint

Macro F1 is especially important when classes are imbalanced, because it gives each class more equal influence than plain accuracy.

## Practical Notes

`class_weights: True` is usually helpful when class distributions are imbalanced.

`task_weights` only affects `FixedWeightedMultiTaskCELoss`.

Negative `Combined Loss` is expected for the uncertainty losses, but not for `SummedMultiTaskCELoss` or `FixedWeightedMultiTaskCELoss` with non-negative task weights.

When resuming from a checkpoint, use the same loss type unless you intentionally want to start a new experiment. The checkpoint stores the loss wrapper state, and uncertainty losses have trainable `log_var` parameters.
