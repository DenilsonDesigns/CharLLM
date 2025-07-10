const { forward } = require("../model/model");
const { oneHotEncode, indexToChar, charToIndex } = require("../data/data");

// Choose index by sampling from probability distribution
function sampleFromProbabilities(probabilities) {
  const r = Math.random();
  let accum = 0;
  for (let i = 0; i < probabilities.length; i++) {
    accum += probabilities[i];
    if (r < accum) return i;
  }
  return probabilities.length - 1; // fallback
}

function generateText(startChar, length = 20) {
  let currentChar = startChar;
  let output = currentChar;

  for (let i = 0; i < length; i++) {
    const inputVec = oneHotEncode(currentChar);
    const { outputProb } = forward(inputVec);
    const nextIndex = sampleFromProbabilities(outputProb);
    const nextChar = indexToChar[nextIndex];

    output += nextChar;
    currentChar = nextChar;
  }

  return output;
}

module.exports = { generateText };
