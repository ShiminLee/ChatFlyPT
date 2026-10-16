"""Re-run the published sensory sentence through the active chat renderer.

No adviser, external API, navigation policy or synthetic outward sound is used.
Fresh trials have fixed new seeds; archived trials are separately re-decoded.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits

from chatflypt.brain import Brain
from chatflypt.experiment import Decoder
from chatflypt.expression import expression_report
from chatflypt.language import verbalize
from chatflypt.paths import ROOT, ARTIFACTS, REPORTS, dump, sha256
from chatflypt.stimuli import FIRST_SENTENCE, groups_for


def main():
    output=ROOT/'reports/neural-language-route'
    raw_dir=ROOT/'data/derived/neural-language-route'
    raw_dir.mkdir(parents=True,exist_ok=True)
    brain=Brain()
    decoder=Decoder()
    manifest=json.loads((ARTIFACTS/'stimuli.json').read_text(encoding='utf8'))
    internal=np.asarray(manifest['readout_indices'])
    groups=groups_for(manifest,FIRST_SENTENCE)
    driven=np.asarray([brain.lookup[str(n)] for ids,_ in groups for n in ids])
    assert not np.intersect1d(decoder.indices,driven).size
    assert np.isin(decoder.indices,internal).all()

    def translate(counts,oid):
        # The decoder sees only neural counts; no source labels or user message.
        decoded=decoder.decode(counts)
        expr=expression_report({'id':oid,'dataset':'FlyWire v783','decoded':decoded,
            'reaction':{'non_sensory_spikes':int(counts[internal].sum())}})
        return {'decoded_words':decoded['words'],'expression':expr,
            'english_sensory_text':verbalize(decoded['words'],ask=False),
            'matches_example_words':set(decoded['words'])==set(FIRST_SENTENCE)}

    archived=[]
    old=json.loads((REPORTS/'first-sentence.json').read_text(encoding='utf8'))
    for row in old['trials']:
        path=ROOT/row['raw_spikes']
        with np.load(path,allow_pickle=False) as raw:
            record=translate(raw['counts'],'archived-'+str(row['seed']))
        record.update(seed=row['seed'],raw_sha256_matches=sha256(path)==row['raw_sha256'])
        archived.append(record)

    brain.run([],1,0)  # Compile once, before parallel independent trials.
    def fresh(seed):
        trial=brain.run(groups,300,seed,record_spikes=True)
        path=raw_dir/f'{seed}.npz'
        np.savez_compressed(path,counts=trial.counts,spike_indices=trial.spike_ids,
                            spike_steps=trial.spike_steps,seed=seed,duration_ms=300)
        with np.load(path,allow_pickle=False) as saved:
            np.testing.assert_array_equal(np.bincount(saved['spike_indices'],minlength=len(brain.ids)),saved['counts'])
            record=translate(saved['counts'],'fresh-'+str(seed))
        record.update(seed=seed,raw_record=str(path.relative_to(ROOT)),raw_sha256=sha256(path))
        return record

    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=2) as pool:
        fresh_trials=list(pool.map(fresh,range(2026092400,2026092420)))
    blank=brain.run([],300,2026092420,record_spikes=True)
    silence=translate(blank.counts,'fresh-silence')
    assert not silence['expression']['messages']
    success=sum(t['matches_example_words'] and bool(t['expression']['text']) for t in fresh_trials)
    report={'scope':'Independent reconstruction of finite neural readout -> natural language',
        'exact_author_replication':False,'llm_called':False,'external_action_control':False,
        'body_physics_tested_here':False,'continuous_feedback_calibrated_here':False,
        'decoder_sha256':sha256(ARTIFACTS/'decoder.npz'),
        'expression_source_sha256':sha256(ROOT/'chatflypt/expression.py'),
        'direct_input_and_sensory_features_excluded':True,
        'fresh_successes':success,'fresh_total':len(fresh_trials),
        'archived_successes':sum(t['matches_example_words'] and bool(t['expression']['text']) for t in archived),
        'all_archived_hashes_match':all(t['raw_sha256_matches'] for t in archived),
        'fixed_question_suffix_used':False,'silence':silence,
        'fresh_trials':fresh_trials,'archived_trials':archived}
    dump(output/'acceptance.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('fresh_trials','archived_trials','silence')},ensure_ascii=False),flush=True)
    print(fresh_trials[0]['expression']['text'],flush=True)
    assert success==len(fresh_trials)


if __name__=='__main__':
    main()
