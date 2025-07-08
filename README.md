## Hosted at:

https://github.com/DenilsonDesigns/CharLLM

Notes:

Step 1: Data prep & hot-encoding

Questions:

- what is hot-encoding and why do we data prep the way we do?
- what is a corpus?
- "one-hot vector"?
-

Step 2: Build a Basic Neural Net

Questions:

- Basically everything we implemented I need to dig further into:

* Input (one-hot vector of char)
  → Hidden layer (tiny # of neurons, e.g. 16)
  → Output layer (softmax over vocab chars)

  Components We’ll Implement

  1.  random weight init
  2.  Forward pass:
      - Input -> Hidden (ReLU or tanh)
      - Hidden - Output (softmax)
  3.  Loss: cross-entropy
  4.  Backward pass: gradient descent
  5.  Training loop
  6.  Sampling (text generation)

Step 3: Loss and Gradient
