# Basic character predicting model/LLM

Repo:
https://github.com/DenilsonDesigns/CharLLM

Components We’ll Implement

1.  One-hot encoding: random weight init
2.  Forward pass:
    - Input -> Hidden (ReLU or tanh)
    - Hidden - Output (softmax)
3.  Loss: cross-entropy
4.  Backward pass: gradient descent
5.  Training loop
6.  Sampling (text generation)

## Step 1: Data prep & hot-encoding

We convert each character into a one-hot vector — a list where only one index is "on".

From a string like `"hello world"`, we generate training pairs:

- `'h'` → `'e'`
- `'e'` → `'l'`
- `'l'` → `'l'`
- etc.

Each pair becomes:

```
{
  input:  [1, 0, 0, 0, ...], // 'h'
  target: [0, 1, 0, 0, ...]  // 'e'
}
```

## Step 2: (Forward pass) Build a Basic Neural Net

We build a small neural network with two weight matrices:

W1: maps input → hidden neurons

W2: maps hidden → output logits

The output goes through:

ReLU (non-linear activation)

Softmax (to convert scores into probabilities)

This currently gives us essentially a random prediction on the first pass:

```
[0.125, 0.125, 0.125, 0.125, ...] // → flat random guess
```

## Step 3: Loss function (cross-entropy)

Cross-entropy gives us a penalty when the model predicts the wrong character.

The more confident and wrong it is, the higher the loss.

## Step 4: Back-propagation of loss function

This is the process of applying our loss backward through the neural network and figuring out how much the weights that were previously applied contributed to the incorrect result.

Applying 100 runs of this backward propagation training, we can start with a basically flat probability of a random guess:

```
// Before training:
[0.125, 0.125, 0.125, 0.125, ...] // → flat

// After 100 training steps on 'h' → 'e':
[0.003, 0.977, 0.003, 0.003, ...] // → confident in 'e'
```

This is after training the model to target `e` after character `h`
