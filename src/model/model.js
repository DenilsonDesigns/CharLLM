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

/**
 * Example for W1/W2:
  [
    0.0047343596328174664, 
    0.008812500613511312,  
    0.003732669088724674,  
    0.0007149477878344191, 
    0.0024370031972661944, 
    0.00529274185800737,   
    0.007882645655596432,  
    0.0008945588410191196  
  ]
 */

// Activation: ReLU
// In this example, the ReLU does nothing, as we are feeding it:
/**
 * [
  0.0019168301980879289,
  0.009483825439606344,
  0.0005918440827694772,
  0.003973657602150138,
  0.009110943691961849,
  0.008436279722900435,
  0.005940415444028961,
  0.0037332654766470364,
  0.0029963106438972644,
  0.006940712493531432,
  0.006015048496892506,
  0.00992179374061352,
  0.0020291559022959363,
  0.005772072513619597,
  0.00695049890257039,
  0.0033261841118561565
]
  and all these numbers are > 0, so it effectively just returns it to us. 
  in more complex models/training tho, inputs can be negative (not sure how/why)
  need to investigate this more later. 
 */
function relu(vec) {
  return vec.map((x) => Math.max(0, x));
}

// Softmax for output layer
// give probabilities summing to 1 for each vector entry.
function softmax(vec) {
  const max = Math.max(...vec); // avoid overflow
  const exps = vec.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

// Matrix multiply: vec • W
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

module.exports = {
  W1,
  W2,
  forward,
};
