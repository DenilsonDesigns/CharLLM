const {
  corpus,
  uniqueChars,
  vocabSize,
  charToIndex,
  indexToChar,
  oneHotEncode,
  oneHotDecode,
  trainingData,
} = require("../data/data");

console.log("📄 Corpus:");
console.log(corpus);

console.log("\n🔠 Unique Characters:");
console.log(uniqueChars);

console.log("\n🔢 Vocabulary Size:");
console.log(vocabSize);

console.log("\n🔁 Character to Index Map:");
console.log(charToIndex);

console.log("\n🔁 Index to Character Map:");
console.log(indexToChar);

console.log("\n🔁 Training data:");
console.log(trainingData);

console.log("\n🧪 Test One-Hot Encoding for 'e':");
const encodedH = oneHotEncode("e");
console.log(encodedH);

console.log("\n🔍 Decoding back to character:");
const decoded = oneHotDecode(encodedH);
console.log(decoded);

console.log("\n📚 Training Data Samples (first 5):");
for (let i = 0; i < 5; i++) {
  const { input, target } = trainingData[i];
  const inputChar = oneHotDecode(input);
  const targetChar = oneHotDecode(target);
  console.log(`Pair ${i + 1}: '${inputChar}' → '${targetChar}'`);
}
