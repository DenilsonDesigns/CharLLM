const { oneHotEncode, oneHotDecode } = require("../data/data");
const { forward, W1, W2, trainStep } = require("../model/model");
const { crossEntropyLoss } = require("../utils/loss");

const inputVec = oneHotEncode("h");
const targetVec = oneHotEncode("e");
const result = forward(inputVec);
const loss = crossEntropyLoss(result.outputProb, targetVec);

console.log('\n🧠 W1: "', W1);
console.log('\n🧠 W2: "', W2);

console.log('\n🧠 Forward Pass for "h"');

// console.log("\n🧠 hiddenRaw: ", result.hiddenRaw);
// console.log("\n🧠 hiddenAct: ", result.hiddenAct);
// console.log("\n🧠 outputRaw: ", result.outputRaw);
console.log("\nOutput Probabilities:", result.outputProb);

console.log(
  "\nPredicted Char:",
  oneHotDecode(
    result.outputProb.map((p) => (p === Math.max(...result.outputProb) ? 1 : 0))
  )
);

console.log("\n🔻 Cross-Entropy Loss:", loss.toFixed(6)); // 2.07 when predicting "w" instead of "e" (high)

// 🧪 Training loop
console.log("\n🏋️ TRAINING...");

for (let step = 0; step < 100; step++) {
  const loss = trainStep(inputVec, targetVec, 0.1);
  if (step % 10 === 0) {
    console.log(`Step ${step}: Loss = ${loss.toFixed(6)}`);
  }
}

// 🔍 After training
console.log("\n✅ AFTER TRAINING");
const resultAfter = forward(inputVec);

console.log("\nOutput Probabilities:", resultAfter.outputProb);
const lossAfter = crossEntropyLoss(resultAfter.outputProb, targetVec);

console.log(
  "Predicted Char:",
  oneHotDecode(
    resultAfter.outputProb.map((p) =>
      p === Math.max(...resultAfter.outputProb) ? 1 : 0
    )
  )
);
console.log("Loss:", lossAfter.toFixed(6));
