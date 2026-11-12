import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml

root=Path(__file__).resolve().parents[3]
base=root/'artifacts/evaluations/phone69_460h_stride_sweep_local_cpu_20261111'
components=('encoder_weights','attention_products','projections','fusion_mlp','detector_dots','fusion_weighted_sum','predictor','joiner')
protocol=dict(device='cpu',dtype='float32',batch=1,beam=20,threads=4,interop_threads=1,data_workers=0,openmp_max_active_levels=1,warmup_utterances=10,subset='test-clean')
def read(path):
    return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(data)
    return h.hexdigest()
def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))
def distance(a,b):
    while a and b and a[0]==b[0]:
        a=a[1:];b=b[1:]
    while a and b and a[-1]==b[-1]:
        a=a[:-1];b=b[:-1]
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(previous[j]+1,current[j-1]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

summary=read(base/'summary.json')
controller=read(base/'controller_state.json')
assert summary['complete'] and controller['status']=='complete'
assert sorted(map(int,summary['models']))==sorted(controller['completed_models'])==[1,2,8]
assert (base/'EVALUATION_COMPLETE').is_file()
manifest0=read(base/'stride1_proj128/manifest.json')
refs=rows(Path(manifest0['reference_csv']))
assert len(refs)==2620 and len({r['uid'] for r in refs})==2620
checks=[]
for stride in [1,2,8]:
    folder=base/('stride%d_proj128'%stride)
    rp=folder/'attempt_01/result.json'; cp=folder/'attempt_01/utterances.csv'; mp=folder/'manifest.json'
    result=read(rp); manifest=read(mp); records=rows(cp)
    assert result==summary['models'][str(stride)]
    assert result['verified_full_corpus'] and result['protocol']==manifest['protocol']==protocol
    assert sha(mp)==result['manifest_sha256'] and sha(cp)==result['csv_sha256']
    assert result['model']==manifest['model']
    entry=manifest['model']
    for key in ['checkpoint','config','training_config']:
        assert sha(entry[key])==entry[key+'_sha256'],key
    cfg=yaml.safe_load(Path(entry['config']).read_text())
    training=yaml.safe_load(Path(entry['training_config']).read_text())
    assert cfg['model']==training['model'] and cfg['model']['time_reduction_stride']==stride
    assert cfg['decoding']['beam_width']==20 and cfg['data']['num_workers']==0
    assert sha(cfg['paths']['global_stats_path'])==manifest['global_stats_sha256']
    assert sha(cfg['tokenizer']['vocab_path'])==manifest['vocab_sha256']
    assert sha(manifest['reference_csv'])==manifest['reference_csv_sha256']
    for relative,digest in manifest['code_sha256'].items():
        assert sha(Path(manifest['source'])/relative)==digest,relative
    assert sha(root/'scripts/evaluation/eval_phone69_stride_checkpoint.py')==manifest['evaluator_sha256']
    for key in ['audio','chapter_labels','reference_csv_sha256','reference_manifest_sha256','global_stats_sha256','vocab_sha256']:
        assert manifest[key]==manifest0[key],key
    assert [r['uid'] for r in records]==[r['uid'] for r in refs]==[r['uid'] for r in manifest['audio']]
    subset=Path(cfg['paths']['dataset_path'])/'LibriSpeech/test-clean'
    if stride==1:
        for item in manifest['audio']:
            uid=item['uid']; speaker,chapter,_=uid.split('-')
            assert sha(subset/speaker/chapter/(uid+'.flac'))==item['sha256']
        for name,digest in manifest['chapter_labels'].items():
            speaker,chapter,_=name.split('-',2)
            assert sha(subset/speaker/chapter/name)==digest
    for index,(row,ref,audio) in enumerate(zip(records,refs,manifest['audio'])):
        assert int(row['idx'])==index and row['ref']==ref['ref']
        phones=row['ref'].split(); hyp=row['hyp'].split()
        assert len(phones)==int(row['reference_phones'])==int(ref['reference_phones'])
        assert distance(phones,hyp)==int(row['phone_errors'])
        assert math.isclose(float(row['PER']),int(row['phone_errors'])/len(phones),rel_tol=1e-12)
        assert float(row['audio_seconds'])==audio['samples']/audio['sample_rate']
        frames=audio['samples']//cfg['data']['hop_length']+1
        steps=(frames+stride-1)//stride; width=128*stride
        assert json.loads(row['layer_frames'])==[steps]*10
        expected=80*128*frames+10*(steps*(4*width*512+7*width**2+31*width)+2*width*steps**2)+256*width*steps
        assert sum(int(row[k]) for k in components[:-2])==expected
        assert int(row['predictor'])==int(row['predictor_tokens'])*294912
        assert int(row['joiner'])==int(row['joiner_rows'])*256*70
    counts={k:sum(int(row[k]) for row in records) for k in components}
    phones=sum(int(row['reference_phones']) for row in records)
    errors=sum(int(row['phone_errors']) for row in records)
    duration=math.fsum(float(row['audio_seconds']) for row in records)
    assert len(records)==result['utterances']==2620 and phones==result['reference_phones']==189023
    assert errors==result['phone_errors'] and duration==result['audio_seconds']==19452.480625
    assert counts==result['components_macs'] and sum(counts.values())==result['total_major_macs']
    assert errors/phones==result['corpus_PER']
    assert sum(counts.values())/duration/1e9==result['gmac_per_audio_second']
    assert sum(counts[k] for k in components[:-2])/duration/1e9==result['encoder_compression_gmac_per_audio_second']
    assert counts['predictor']/duration/1e9==result['predictor_gmac_per_audio_second']
    assert counts['joiner']/duration/1e9==result['joiner_gmac_per_audio_second']
    checks.append(dict(name=entry['name'],result_sha256=sha(rp),csv_sha256=sha(cp),manifest_sha256=sha(mp),
        checkpoint_sha256=entry['checkpoint_sha256'],utterances=len(records),reference_phones=phones,
        phone_errors=errors,corpus_PER=errors/phones,gmac_per_audio_second=result['gmac_per_audio_second'],
        audio_seconds=duration,full_reference_order_verified=True,independent_edit_distances_verified=True,
        independent_csv_totals_verified=True,per_utterance_encoder_formula_verified=True,
        per_utterance_decoder_matrix_counts_verified=True,inputs_and_frozen_source_hashes_verified=True))
    print(json.dumps({'verified':entry['name'],'utterances':len(records),'phone_errors':errors}),flush=True)
value=dict(schema=1,verified_at_client_utc=datetime.now(timezone.utc).isoformat(),valid=True,
    scope=[r['name'] for r in checks],models=checks,completion_marker_sha256=sha(base/'EVALUATION_COMPLETE'),
    summary_sha256=sha(base/'summary.json'),controller_status=controller['status'],
    controller_cached_pending_strides=controller.get('pending_strides'),
    note='The completed controller retains cached pending_strides=[8]; status, completed_models, all full results and EVALUATION_COMPLETE agree on completion.')
(base/'verification_complete.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'valid':True,'models':len(checks),'verification':str(base/'verification_complete.json')}))
