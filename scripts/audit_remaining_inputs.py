"""Export evidence and exact missing bridges, without inventing neural inputs.

Optional research-only readers: rdata==1.1.0, xarray==2026.7.0.
Neither is imported by the application or neural inference service.
"""
import json
from pathlib import Path
import sys
import warnings
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from chatflypt.paths import dump,sha256
from chatflypt.sensory_catalogue import build_catalogue


def main():
    import rdata
    warnings.filterwarnings('ignore',category=UserWarning,module='rdata')
    folder=ROOT/'research/retinotopy'
    data=rdata.read_rda(folder/'data_eyemap.RData')
    pairs=np.asarray(data['eyemap'],dtype=int)
    native=np.asarray(data['med_xyz']);direction=np.asarray(data['ucl_rot_sm'])
    if pairs.shape!=(778,2) or native.shape!=(778,3) or direction.shape!=(778,3):
        raise ValueError('Unexpected publication coordinate tables')
    if not np.isfinite(direction).all() or not np.isfinite(native).all():raise ValueError('Non-finite coordinates')
    # These are author row indices, NOT connectome neuron IDs. Export explicitly
    # named columns so this table cannot be mistaken for an executable ID map.
    import pandas as pd
    table=pd.DataFrame(np.c_[pairs,native,direction],columns=[
        'author_mi1_row_1based','author_lens_row_1based','native_x','native_y','native_z',
        'eye_direction_x','eye_direction_y','eye_direction_z'])
    table.iloc[:,:2]=table.iloc[:,:2].astype(int)
    table.to_csv(folder/'published_eye_coordinates.csv',index=False)
    pr=rdata.read_rda(folder/'data_neu_PR.RData')['PR']
    photo_id=str(pr['NeuronName'][0])
    catalogue=build_catalogue(save=False)
    missing={k:{'label':v['label'],'limitations':v.get('limitations',[]),
                'evidence':v.get('evidence',[]),'reason':v.get('reason',v.get('status'))}
             for k,v in catalogue['channels'].items() if v['status']=='unmapped' and not v.get('advanced',False)}
    with np.load(ROOT/'research/outward-signals/flies.npz',allow_pickle=False) as song:
        song_record={'sample_rate_hz':int(song['samplerate']),
            'duration_s':len(song['recording'])/int(song['samplerate']),
            'pulse_annotations':len(song['pulsetimes']),'sine_annotations':len(song['sinetimes']),
            'finite_waveform':bool(np.isfinite(song['recording']).all()),
            'annotation_scope':'pulse/sine acoustic event classes, not sentence meanings',
            'sha256':sha256(ROOT/'research/outward-signals/flies.npz')}
    from chatflypt.acoustic_readout import acoustic_capability
    song_record['acoustic_classifier']=acoustic_capability()
    report={'date':'2026-09-24','full_free_chat_complete':False,
        'persistent_session':'implemented_and_physics_tested; simulation pauses between requests; explicit 60 s recording limit',
        'vision':{'publication':'https://www.nature.com/articles/s41586-025-09276-5',
            'source_manifest':'research/retinotopy/sources.json','published_pairs':len(pairs),
            'coordinate_export':'research/retinotopy/published_eye_coordinates.csv',
            'example_photoreceptor_namespace':'CATMAID skeleton ID','example_photoreceptor_id':photo_id,
            'executable_neuron_registration':False,
            'missing':['verified CATMAID-to-current-FlyWire/MaleCNS cell identity mapping',
                       'left/right camera geometry to published eye-frame registration',
                       'photoreceptor transfer model and independent spatial-stimulus validation']},
        'outward_sound':song_record,
        'outward_generation':{'measured_acoustic_pressure':False,'measured_chemical_release':False,
            'calibrated_substrate_vibration':False,'semantic_labels_for_open_conversation':False,
            'source':'current body worker contains joint kinematics, not these measured output modalities'},
        'unmapped_named_input_channels':missing,
        'interoception':{'metabolic_and_endocrine_state_model':False,
            'note':'Existing LIF connectivity does not implement receptor chemistry, metabolism, or peptide dynamics.'},
        'not_equivalent':['signal classification is not sentence translation',
                          'anatomical cell IDs are not calibrated sensory transduction',
                          'persistent dynamic state is not learned language or plasticity']}
    columns=folder/'column-candidate-audit.json'
    if columns.exists():
        recovered=json.loads(columns.read_text(encoding='utf8'))
        report['vision']['local_column_candidates']={k:recovered[k] for k in
            ('photoreceptors','statuses','candidate_counts_by_side','unique_candidate_columns','used_in_live_simulation')}
        report['vision']['local_column_candidate_file']='research/retinotopy/column-candidate-audit.json'
    dump(ROOT/'reports/persistent-session/remaining-capabilities.json',report)
    print(json.dumps({'eye_pairs':len(pairs),'catmaid_example':photo_id,'sound':song_record,
                      'unmapped_named_inputs':list(missing)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
