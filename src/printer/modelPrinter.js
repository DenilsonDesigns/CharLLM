const { oneHotEncode, oneHotDecode } = require("../data/data");
const { forward } = require("../model/model");

const inputVec = oneHotEncode("h");
const result = forward(inputVec);

console.log('\n🧠 Forward Pass for "h"');
console.log("Output Probabilities:", result.outputProb);
console.log(
  "Predicted Char:",
  oneHotDecode(
    result.outputProb.map((p) => (p === Math.max(...result.outputProb) ? 1 : 0))
  )
);
