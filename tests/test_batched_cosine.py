"""Reference equivalence tests; on CUDA additionally exercises device math."""
import unittest
import numpy as np
import torch
from scipy.signal import find_peaks
from conformer_rnnt.compression.natural import NaturalPeakCompression
from conformer_rnnt.compression.batched_cosine import frame_mask_from_cosine_distances, batched_natural_cosine_fusion


def scipy_mask(row, length, width):
    result=np.zeros(width,dtype=np.bool_)
    if length == 0:return result
    if length == 1:result[0]=True;return result
    c=row[:length-1]
    if len(c)<3:indices=np.array([0])
    else:
        peaks,_=find_peaks(c);valleys,_=find_peaks(-c)
        indices=np.unique(np.concatenate([peaks+1,valleys]));indices=indices[indices<length]
        if len(indices)==0:indices=np.array([0])
    result[indices]=True
    return result


class BatchedCosineTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_scipy_extrema_plateaus_boundaries_and_nan(self):
        rng=np.random.default_rng(123)
        seqs=[[],[0],[0,1],[1,0,1],[0,2,2,0],[0,2,2,2,0],
              [2,1,1,2],[0,1,0,1,0],[0,0,0,0],[1,np.nan,1,2,1],
              [np.nan,np.nan,1,1,2,1],[-np.inf,1,1,-np.inf],[0,-0.,1,-0.]]
        for i in range(200):seqs.append(rng.integers(-3,4,size=rng.integers(0,48)).astype(float).tolist())
        width=max(map(len,seqs));d=torch.zeros((len(seqs),width),dtype=torch.float32);lens=[]
        for i,seq in enumerate(seqs):d[i,:len(seq)]=torch.tensor(seq);lens.append(len(seq)+1)
        # Include empty feature sequence and a length-one sequence.
        lens[0]=0;lens[1]=1
        expected=np.stack([scipy_mask(d[i].numpy(),n,width+1) for i,n in enumerate(lens)])
        actual=frame_mask_from_cosine_distances(d,torch.tensor(lens)).numpy()
        np.testing.assert_array_equal(actual,expected)
    def test_reference_output_indices_and_gradients(self):
        torch.manual_seed(718)
        for mode,window in [('five_frames',5),('nine_frames',9)]:
            for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                ref=NaturalPeakCompression(detector={'type':'cosine_diff'},mixer={'type':'learned'},feature_dim=8,window_size=window,weight_predictor_mode=mode).to(device)
                ref.record_indices=True
                x=torch.randn(11,49,8,device=device,requires_grad=True);lengths=torch.tensor([0,1,2,3,4,5,12,20,49,35,44],device=device,dtype=torch.int32)
                x2=x.detach().clone().requires_grad_(True)
                result,ln=ref(x,lengths);loss=result.square().sum();loss.backward();gx=x.grad.detach().clone();gp=[p.grad.detach().clone() for p in ref.mixer.parameters()]
                ref.zero_grad(set_to_none=True)
                actual,ln2,mask=batched_natural_cosine_fusion(x2,lengths,ref.mixer)
                actual.square().sum().backward()
                torch.testing.assert_close(ln2,ln,rtol=0,atol=0)
                self.assertEqual(ref.last_indices['retained'],[torch.where(mask[i])[0].tolist() for i in range(len(lengths))])
                torch.testing.assert_close(actual,result,rtol=2e-5,atol=2e-6)
                torch.testing.assert_close(x2.grad,gx,rtol=3e-5,atol=3e-6)
                for p,g in zip(ref.mixer.parameters(),gp):torch.testing.assert_close(p.grad,g,rtol=3e-5,atol=3e-6)
    def test_constant_near_tie_random_and_padding(self):
        torch.manual_seed(14)
        ref=NaturalPeakCompression(detector={'type':'cosine_diff'},mixer={'type':'learned'},feature_dim=8,window_size=9,weight_predictor_mode='nine_frames');ref.record_indices=True
        for k in range(20):
            x=torch.randn(16,100,8);lengths=torch.randint(0,101,(16,),dtype=torch.int32)
            if k%2==0:x[0]=1
            for i,n in enumerate(lengths.tolist()):x[i,n:]=99.0
            y,l=ref(x,lengths);z,l2,m=batched_natural_cosine_fusion(x,lengths,ref.mixer)
            self.assertEqual(ref.last_indices['retained'],[torch.where(row)[0].tolist() for row in m]);torch.testing.assert_close(l,l2,rtol=0,atol=0);torch.testing.assert_close(y,z,rtol=2e-5,atol=2e-6)
    def test_all_empty_and_unsupported(self):
        ref=NaturalPeakCompression(detector={'type':'cosine_diff'},mixer={'type':'learned'},feature_dim=3,window_size=5,weight_predictor_mode='five_frames')
        x=torch.randn(2,7,3);lens=torch.tensor([0,0],dtype=torch.int32)
        y,l,m=batched_natural_cosine_fusion(x,lens,ref.mixer);self.assertEqual(tuple(y.shape),(2,0,3));self.assertFalse(m.any())
        with self.assertRaises(ValueError):batched_natural_cosine_fusion(x,torch.tensor([0,8]),ref.mixer)
    def test_factory_configuration_checkpoint_and_rnnt(self):
        import copy, tempfile
        from pathlib import Path
        import json
        import torchaudio
        from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
        torch.manual_seed(91)
        with tempfile.TemporaryDirectory() as directory:
            stats=Path(directory)/'stats.json';stats.write_text(json.dumps({'mean':[0.]*8,'invstddev':[1.]*8}))
            cfg={'paths':{'dataset_path':directory,'global_stats_path':str(stats)},'tokenizer':{'type':'phone69'},'model':{
                'sequence_mode':'offline','input_dim':8,'encoding_dim':16,'num_symbols':69,
                'time_reduction_input_dim':8,'time_reduction_stride':1,'num_heads':2,'ffn_dim':32,
                'num_layers_block1':2,'num_layers_block2':1,'depthwise_conv_kernel_size':3,'dropout':0.,
                'compression':{'detector':{'type':'cosine_diff'},'mixer':{'type':'learned'},'weight_predictor_mode':'nine_frames','hanning_window_size':9},
                'predictor':{'symbol_embedding_dim':8,'num_lstm_layers':1,'lstm_hidden_dim':8,'lstm_dropout':0.}},'data':{'n_mels':8}}
            original=LibriSpeechRNNTModule.from_config(cfg).eval();fast_cfg=copy.deepcopy(cfg);fast_cfg['model']['compression']['execution_backend']='batched'
            fast=LibriSpeechRNNTModule.from_config(fast_cfg).eval();self.assertEqual(fast.model.transcriber.compression.execution_backend,'batched')
            fast.load_state_dict(original.state_dict(),strict=True)
            checkpoint={};original.on_save_checkpoint(checkpoint);fast.on_load_checkpoint(checkpoint)
            self.assertEqual(original.offline_native_definition,fast.offline_native_definition)
            x=torch.randn(2,23,8);lengths=torch.tensor([23,17],dtype=torch.int32);targets=torch.tensor([[1,2,3],[4,5,69]],dtype=torch.int32);tl=torch.tensor([3,2],dtype=torch.int32)
            inputs=torch.cat((targets.new_full((2,1),69),targets),dim=1)
            outputs=[];losses=[]
            for module in [original,fast]:
                out,ol,_,_=module.model(x,lengths,inputs,tl+1)
                loss=torchaudio.transforms.RNNTLoss(blank=69,reduction='sum',clamp=1.0)(out,targets,ol,tl)
                loss.backward();outputs.append((out.detach(),ol));losses.append(loss.detach())
            torch.testing.assert_close(outputs[0][1],outputs[1][1],rtol=0,atol=0)
            torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=3e-5,atol=3e-6)
            torch.testing.assert_close(losses[0],losses[1],rtol=3e-5,atol=3e-6)
            for a,b in zip(original.model.parameters(),fast.model.parameters()):
                if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=5e-4,atol=1e-5)
            wrong=copy.deepcopy(fast_cfg);wrong['model']['compression']['detector']={'type':'raw'}
            with self.assertRaises(ValueError):LibriSpeechRNNTModule.from_config(wrong)


if __name__=='__main__':unittest.main()
