import numpy as np
from scipy.sparse import csr_matrix
from chatflypt.brain import simulate


def test_passive_voltage_recording_preserves_all_engine_outputs():
    graph = csr_matrix(np.array([[0., -12., 4.], [0., 0., 0.], [0., 0., 0.]]))
    times = np.array([10, 50, 90])
    tape = (np.array([0], np.int32),
            np.r_[0, np.cumsum(np.bincount(times, minlength=200))],
            np.zeros(3, np.int32), 200)
    original = simulate(graph.indptr, graph.indices, graph.data, *tape, True)
    state = np.zeros((4, 3))
    observed = simulate(graph.indptr, graph.indices, graph.data, *tape, True, state)
    for a, b in zip(original, observed):
        np.testing.assert_array_equal(a, b)
    assert state[1, 1] > 0 and state[3, 1] > 0
    assert state[0, 2] > 0 and state[2, 2] > 0
    assert observed[0][1] == observed[0][2] == 0


def test_blank_has_no_spikes_or_subthreshold_response():
    graph = csr_matrix(np.array([[0., -100.], [100., 0.]]))
    state = np.zeros((4, 2))
    result = simulate(graph.indptr, graph.indices, graph.data,
                      np.array([], np.int32), np.zeros(101, np.int64),
                      np.array([], np.int32), 100, True, state)
    assert not state.any() and not result[0].any()


def test_decoder_requires_measurable_internal_signal_even_with_positive_intercept(tmp_path):
    from chatflypt.state_readout import StateDecoder
    path=tmp_path/'state.npz'
    np.savez(path,feature_indices=[3],indices=[1],weights=[[10.]],intercept=[2.],
        thresholds=[.5],qualified=[True],labels=['probe:left'],names=['左侧候选'],catalogue_signature='test')
    decoder=StateDecoder('FlyWire v783',path)
    counts=np.zeros(2);state=np.zeros((4,2))
    state[0,1]=1e-7;state[2,1]=1e-8
    assert decoder.decode(counts,state)['words']==[]
    counts[0]=10000;state[:,0]=10000
    assert decoder.decode(counts,state)['words']==[]
    state[0,1]=.1;state[2,1]=.01
    assert decoder.decode(counts,state)['words']==['probe:left']
def test_direct_probe_fit_preserves_the_regularized_linear_objective():
    from scipy import sparse
    from sklearn.linear_model import Ridge
    from chatflypt.state_readout import fit_probe
    rng = np.random.default_rng(4282)
    x = sparse.csr_matrix(rng.normal(size=(80, 250)))
    y = rng.normal(size=(80, 6))
    direct = fit_probe(x, y)
    iterative = Ridge(alpha=10, solver='lsqr', tol=1e-10).fit(x, y)
    np.testing.assert_allclose(direct.predict(x), iterative.predict(x), atol=1e-8, rtol=1e-8)


def test_validation_threshold_honors_reporting_precision_before_recall():
    from chatflypt.state_readout import choose_thresholds
    # A low cutoff maximizes F1 but would report a wrong stimulus. A qualified
    # cutoff preserves both isolated positives while declining weak mixtures.
    scores = np.array([.60, .55, .45, .44, .43, .42, .48, .10])[:, None]
    truth = np.array([1, 1, 1, 1, 1, 1, 0, 0])[:, None]
    isolated = np.array([True, True, False, False, False, False, True, True])
    cutoff = choose_thresholds(scores, truth, isolated, np.ones(8, dtype=bool))[0]
    predicted = scores[:, 0] >= cutoff
    assert predicted[:2].all() and not predicted[6:].any()
    assert .48 < cutoff <= .55


def test_validation_cannot_manufacture_separation_for_identical_scores():
    from chatflypt.state_readout import choose_thresholds
    scores = np.array([[.6], [.6]])
    truth = np.array([[1], [0]])
    cutoff = choose_thresholds(scores, truth, np.ones(2, dtype=bool), np.ones(2, dtype=bool))[0]
    predicted = scores[:, 0] >= cutoff
    assert predicted[0] == predicted[1]
