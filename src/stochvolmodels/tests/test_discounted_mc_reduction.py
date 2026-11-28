"""Ordinary paired Monte Carlo reduction across trade and batch boundaries."""

import numpy as np
import pytest
from stochvolmodels.valuation import reduce_discounted_mc


def test_merge_and_pair():
    x = np.array([84.48, 74.88, 103.68, 103.68])
    ids = np.array([7, 3, 7, 3])
    kw = dict(stream_id="s", group_size=2, keep_groups=True)
    estimate = reduce_discounted_mc(x, ids, **kw)
    chunks = [reduce_discounted_mc(x[ids == i], ids[ids == i], **kw) for i in (7, 3)]
    merged = chunks[0].merge(chunks[1])
    assert merged.value == pytest.approx(91.68)
    assert merged.standard_error == pytest.approx(2.4)
    assert merged.paired_difference(estimate).standard_error == 0.0
    with pytest.raises(ValueError, match="overlap"):
        merged.merge(chunks[0])
    with pytest.raises(ValueError, match="incomplete"):
        reduce_discounted_mc(x[:-1], ids[:-1], **kw)
    with pytest.raises(ValueError, match="stream"):
        estimate.paired_difference(
            reduce_discounted_mc(x, ids, group_size=2, stream_id="other", keep_groups=True)
        )
