// Step 2: Forward pass
const { vocabSize } = require("../data/data"); // 8 for 'hello world'

const neurons = 16; // tiny brain neurons

const W1 = new Array(vocabSize)
  .fill()
  .map(() => new Array(neurons).fill().map(() => Math.random() * 0.01));

const W2 = new Array(neurons)
  .fill()
  .map(() => new Array(vocabSize).fill().map(() => Math.random() * 0.01));

// Activation: ReLU
function relu(vec) {
  return vec.map((x) => Math.max(0, x));
}

// Softmax for output layer
// give probabilities summing to 1 for each vector entry.
function softmax(vec) {
  const max = Math.max(...vec);
  const exps = vec.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

function matMulVec(W, vec) {
  const out = new Array(W[0].length).fill(0); // [0, ... *16] for W1, [0, ... * 8] for W2

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

// Train step/backwards propagation
function trainStep(inputVec, targetVec, learningRate = 0.1) {
  // 1. Forward pass
  const { hiddenRaw, hiddenAct, outputProb } = forward(inputVec);

  // 2. Compute loss
  const targetIndex = targetVec.findIndex((v) => v === 1);
  const loss = -Math.log(outputProb[targetIndex] + 1e-9);

  // 3. Backward pass

  // --- Step 3a: dOutput = dLoss/dLogits ---
  const dOutput = outputProb.slice();
  dOutput[targetIndex] -= 1; // softmax + cross-entropy combined

  // --- Step 3b: dW2 = outer product of dOutput and hiddenAct ---
  const dW2 = W2.map((row, i) => row.map((_, j) => dOutput[j] * hiddenAct[i]));

  // --- Step 3c: dHidden = backprop through W2 ---
  const dHidden = new Array(hiddenAct.length).fill(0);
  for (let i = 0; i < hiddenAct.length; i++) {
    for (let j = 0; j < dOutput.length; j++) {
      dHidden[i] += dOutput[j] * W2[i][j];
    }
    // ReLU derivative
    if (hiddenRaw[i] <= 0) dHidden[i] = 0;
  }

  // --- Step 3d: dW1 = outer product of dHidden and inputVec ---
  const dW1 = W1.map((row, i) => row.map((_, j) => dHidden[j] * inputVec[i]));

  // 4. Gradient descent update
  for (let i = 0; i < W2.length; i++) {
    for (let j = 0; j < W2[0].length; j++) {
      W2[i][j] -= learningRate * dW2[i][j];
    }
  }

  for (let i = 0; i < W1.length; i++) {
    for (let j = 0; j < W1[0].length; j++) {
      W1[i][j] -= learningRate * dW1[i][j];
    }
  }

  return loss;
}

module.exports = {
  W1,
  W2,
  forward,
  trainStep,
};
