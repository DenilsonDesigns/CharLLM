// Step 1: data prep & hot-encoding.

const corpus = "hello world";

const uniqueChars = [...new Set(corpus.split(""))];
const vocabSize = uniqueChars.length;

const charToIndex = {};
const indexToChar = {};

uniqueChars.forEach((char, idx) => {
  charToIndex[char] = idx;
  indexToChar[idx] = char;
});

function oneHotEncode(char) {
  const vector = new Array(vocabSize).fill(0);
  const index = charToIndex[char];
  vector[index] = 1;
  return vector;
}

function oneHotDecode(vector) {
  const index = vector.findIndex((v) => v === 1);
  return indexToChar[index];
}

const trainingData = [];

for (let i = 0; i < corpus.length - 1; i++) {
  const inputChar = corpus[i];
  const targetChar = corpus[i + 1];
  trainingData.push({
    input: oneHotEncode(inputChar),
    target: oneHotEncode(targetChar),
  });
}

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
