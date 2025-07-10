const { forward, trainStep } = require("../model/model");
const { crossEntropyLoss } = require("../utils/loss");

function evaluate(trainingData, decodeFn) {
  trainingData.forEach(({ input, target }, i) => {
    const { outputProb } = forward(input);
    const predicted = decodeFn(
      outputProb.map((p) => (p === Math.max(...outputProb) ? 1 : 0))
    );
    const targetChar = decodeFn(target);
    const loss = crossEntropyLoss(outputProb, target);
    console.log(
      `Sample ${i}: Predicted '${predicted}' | Target '${targetChar}' | Loss: ${loss.toFixed(
        4
      )}`
    );
  });
}

function train(trainingData, { epochs = 100, learningRate = 0.1 }) {
  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    for (const { input, target } of trainingData) {
      totalLoss += trainStep(input, target, learningRate);
    }
    if (epoch % 10 === 0) {
      const avgLoss = totalLoss / trainingData.length;
      console.log(`Epoch ${epoch}: Avg Loss = ${avgLoss.toFixed(4)}`);
    }
  }
}

module.exports = { train, evaluate };
