const { trainingData, oneHotDecode } = require("../data/data");
const { train, evaluate } = require("../trainer/trainer");
const { generateText } = require("../sampler/sampler");

console.log("\n🔎 BEFORE TRAINING");
evaluate(trainingData, oneHotDecode);

console.log("\n🏋️ FULL TRAINING");
train(trainingData, { epochs: 100, learningRate: 0.1 });

console.log("\n✅ AFTER TRAINING");
evaluate(trainingData, oneHotDecode);

console.log("\n🧾 SAMPLE OUTPUT");

const seedChar = "h";
const generated = generateText(seedChar, 20);

console.log(`Starting from '${seedChar}':`);
console.log(generated);
