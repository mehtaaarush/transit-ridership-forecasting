import json
import shutil
import subprocess

import numpy as np
import pytest

from ridership import config
from ridership.export import export_gbm, predict_exported
from ridership.features import MODEL_FEATURES, make_supervised, model_matrix
from ridership.models import GradientBoostingForecaster
from tests.conftest import make_daily

GBM_JS = config.ROOT / "docs" / "assets" / "gbm.js"


@pytest.fixture(scope="module")
def fitted():
    daily = make_daily(n_days=300, snow_effect=0.25, seed=3)
    rows = make_supervised(daily, daily.index[60], daily.index[-15])
    model = GradientBoostingForecaster(n_estimators=60).fit(rows)
    X = model_matrix(rows)
    return model, export_gbm(model.model, MODEL_FEATURES), X


def test_exported_trees_reproduce_scikit_learn(fitted):
    model, spec, X = fitted
    np.testing.assert_allclose(predict_exported(spec, X), model.model.predict(X), rtol=0, atol=1e-9)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_browser_model_matches_python(fitted, tmp_path):
    model, spec, X = fitted
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    (tmp_path / "x.json").write_text(json.dumps(X.tolist()))
    script = (
        "const fs = require('fs'); const gbm = require(process.argv[1]);"
        "const spec = JSON.parse(fs.readFileSync(process.argv[2]));"
        "const X = JSON.parse(fs.readFileSync(process.argv[3]));"
        "process.stdout.write(JSON.stringify(X.map(x => gbm.rawPredict(spec, x))));"
    )
    out = subprocess.run(
        ["node", "-e", script, str(GBM_JS), str(tmp_path / "spec.json"), str(tmp_path / "x.json")],
        check=True, capture_output=True, text=True,
    )
    np.testing.assert_allclose(json.loads(out.stdout), model.model.predict(X), rtol=0, atol=1e-9)
