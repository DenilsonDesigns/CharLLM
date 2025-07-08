const { oneHotEncode, oneHotDecode } = require("../data/data");
const { forward, W1, W2 } = require("../model/model");

const inputVec = oneHotEncode("h");
const result = forward(inputVec);

console.log('\n🧠 W1: "', W1);
console.log('\n🧠 W2: "', W2);

console.log('\n🧠 Forward Pass for "h"');

console.log("\n🧠 hiddenRaw: ", result.hiddenRaw);
console.log("\n🧠 hiddenAct: ", result.hiddenAct);
console.log("\n🧠 outputRaw: ", result.outputRaw);
console.log("\nOutput Probabilities:", result.outputProb);

console.log(
  "\nPredicted Char:",
  oneHotDecode(
    result.outputProb.map((p) => (p === Math.max(...result.outputProb) ? 1 : 0))
  )
);
