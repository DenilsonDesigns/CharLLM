const corpus = "hello world";

// Step 1: Build Vocabulary
const uniqueChars = [...new Set(corpus.split(""))];
const vocabSize = uniqueChars.length;

const charToIndex = {};
const indexToChar = {};

uniqueChars.forEach((char, idx) => {
  charToIndex[char] = idx;
  indexToChar[idx] = char;
});

// Step 2: One-hot encode a character
function oneHotEncode(char) {
  const vector = new Array(vocabSize).fill(0);
  const index = charToIndex[char];
  vector[index] = 1;
  return vector;
}

// Step 3: Decode a one-hot vector to a character
function oneHotDecode(vector) {
  const index = vector.findIndex((v) => v === 1);
  return indexToChar[index];
}

// Step 4: Build input/output training pairs
const trainingData = [];

for (let i = 0; i < corpus.length - 1; i++) {
  const inputChar = corpus[i];
  const targetChar = corpus[i + 1];
  trainingData.push({
    input: oneHotEncode(inputChar),
    target: oneHotEncode(targetChar),
  });
}

// Exports
module.exports = {
  corpus,
  uniqueChars,
  vocabSize,
  charToIndex,
  indexToChar,
  oneHotEncode,
  oneHotDecode,
  trainingData,
};
