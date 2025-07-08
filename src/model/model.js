/**
 * Input (one-hot vector of char)
    → Hidden layer (tiny # of neurons, e.g. 16)
    → Output layer (softmax over vocab chars)

     Components We’ll Implement
     1. random weight init
     2. Forward pass:
        - Input -> Hidden (ReLU or tanh)
        - Hidden - Output (softmax)
     3. Loss: cross-entropy
     4. Backward pass: gradient descent
     5. Training loop
     6. Sampling (text generation)
 */

const { vocabSize } = require("../data/data"); // 8 for 'hello world'

const hiddenSize = 16; // tiny brain neurons

// Weight matrices (input → hidden, hidden → output)
const W1 = new Array(vocabSize)
  .fill()
  .map(() => new Array(hiddenSize).fill().map(() => Math.random() * 0.01));
const W2 = new Array(hiddenSize)
  .fill()
  .map(() => new Array(vocabSize).fill().map(() => Math.random() * 0.01));

// Activation: ReLU
function relu(vec) {
  return vec.map((x) => Math.max(0, x));
}

// Softmax for output layer
function softmax(vec) {
  const max = Math.max(...vec); // avoid overflow
  const exps = vec.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

// Matrix multiply: vec • W
function matMulVec(W, vec) {
  const out = new Array(W[0].length).fill(0);
  for (let i = 0; i < W[0].length; i++) {
    for (let j = 0; j < W.length; j++) {
      out[i] += vec[j] * W[j][i];
    }
  }
  return out;
}

// Forward pass: input vector → prediction
function forward(inputVec) {
  const hiddenRaw = matMulVec(W1, inputVec); // input → hidden
  const hiddenAct = relu(hiddenRaw); // ReLU
  const outputRaw = matMulVec(W2, hiddenAct); // hidden → output
  const outputProb = softmax(outputRaw); // Softmax
  return {
    input: inputVec,
    hiddenRaw,
    hiddenAct,
    outputRaw,
    outputProb,
  };
}

module.exports = {
  W1,
  W2,
  forward,
};
