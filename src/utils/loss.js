function crossEntropyLoss(probs, targetVec) {
  const epsilon = 1e-9; // to avoid log(0)
  // Find the index where targetVec is 1 (one-hot)
  const targetIndex = targetVec.findIndex((v) => v === 1);
  return -Math.log(probs[targetIndex] + epsilon);
}

module.exports = {
  crossEntropyLoss,
};
