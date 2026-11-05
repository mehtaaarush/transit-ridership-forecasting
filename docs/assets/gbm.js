/* gbm.js - evaluates the exported gradient-boosting model (docs/data/model.json) in the
   browser, and in Node for the parity test (tests/test_export.py). */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.GBM = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* Sum of tree outputs on the log-ratio scale. Inputs are rounded to float32 because
     scikit-learn compares float32 features against float64 thresholds. */
  function rawPredict(spec, x) {
    const xs = x.map(Math.fround);
    let out = spec.init;
    for (const tree of spec.trees) {
      let node = 0;
      while (tree.left[node] !== -1) {
        node = xs[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
      }
      out += spec.learning_rate * tree.value[node];
    }
    return out;
  }

  /* Riders for one target day: level at the forecast origin x exp(model), with the
     horizon's calibrated 80% interval. */
  function forecast(spec, x, h) {
    const pred = spec.level * Math.exp(rawPredict(spec, x));
    const q = spec.interval_log_quantiles[String(h)];
    return { pred: pred, lo: pred * Math.exp(q[0]), hi: pred * Math.exp(q[1]) };
  }

  return { rawPredict: rawPredict, forecast: forecast };
});
