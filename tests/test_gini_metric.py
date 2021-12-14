
from featurediscovery.fitter.fit_metrics import Gini_Metric
import cupy as cp



def test_gini():


    y = cp.array([1,1,0,0,1,1,0,0])

    y_hat_good = cp.array([1,1,0,0,1,1,0,0])
    y_had_bad = cp.array([0,1,0,1,0,1,0,1])
    y_hat_ok = cp.array([1,1,1,0,1,1,0,1])

    gini = Gini_Metric()

    gini_good = gini.score_fit_quality(y, y_hat_good)
    gini_bad = gini.score_fit_quality(y, y_had_bad)
    gini_ok = gini.score_fit_quality(y, y_hat_ok)

    assert gini_good == 0.5
    assert gini_bad == 0.0


