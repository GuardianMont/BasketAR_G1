import csv, json, sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts import extract_detections
from scripts.extract_tracking_features import (
    FEATURE_NAMES, _FEATURE_INDEX, _build_dense_features, _gate_ball,
    _window_features, extract_tracking_features,
)
from data.temporal_sampling import build_temporal_sampler

def test_cache_candidates_are_unflipped_and_shared_downstream(tmp_path, monkeypatch):
    class Frame:
        pts=0
        time_base=1/10
        def to_ndarray(self,format): return np.zeros((20,40,3),np.uint8)
    class Stream:
        frames=1
        average_rate=10
    class Container:
        streams=SimpleNamespace(video=[Stream()])
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def decode(self,video=0): return iter([Frame()])
    monkeypatch.setattr(extract_detections.av,"open",lambda *_a,**_k:Container())
    class Boxes:
        xyxy=torch.tensor([[4.,4.,12.,12.],[20.,3.,36.,8.]])
        conf=torch.tensor([.9,.8]); cls=torch.tensor([0.,1.])
        def __len__(self): return 2
    class YOLO:
        def __init__(self,*a): pass
        def predict(self,source,**kwargs): return [SimpleNamespace(boxes=Boxes()) for _ in source]
    monkeypatch.setitem(sys.modules,"ultralytics",SimpleNamespace(YOLO=YOLO))
    root=tmp_path/"data"; root.mkdir(); (root/"weights.pt").write_bytes(b"fake")
    (root/"manifest.csv").write_text("split,clip_id,path\ntrain,c1,v.mp4\n",encoding="utf-8")
    args=Namespace(dataset_root=str(root),manifest=None,split="train",weights=str(root/"weights.pt"),imgsz=64,
        ball_class_id=0,rim_class_id=1,min_confidence=.1,topk_ball=5,topk_rim=2,
        output_root=str(root/"cache"),device="cpu",batch_size=1,limit=None,overwrite=False)
    extract_detections.extract_detections(args)
    cache_path=next((root/"cache").glob("*/*/c1.npz"))
    with np.load(cache_path) as cache:
        original=cache["candidates"].copy()
    assert original.shape==(1,2,5,5)
    det=original[:, :, 0, :]
    unflipped=_build_dense_features(det,np.array([0.]),1.5,False)
    flipped=_build_dense_features(det,np.array([0.]),1.5,True)
    assert unflipped[0,_FEATURE_INDEX["ball_x_center_norm"]]==pytest.approx(.2)
    assert flipped[0,_FEATURE_INDEX["ball_x_center_norm"]]==pytest.approx(.8)
    with np.load(cache_path) as cache:
        np.testing.assert_array_equal(cache["candidates"],original)

def test_sampler_projection_lengths_match_reference():
    dense=np.zeros((20,len(FEATURE_NAMES)),np.float32); times=np.arange(20)/10
    for config,expected in [({"name":"uniform","num_frames":5},5),
                            ({"name":"edge_dense","num_frames":7,"edge_strength":1.0},7)]:
        sampler=build_temporal_sampler(config); idx=sampler.sample_indices(len(dense),fps=10)
        projected=_window_features(dense,times,idx)
        assert len(projected)==expected==len(idx)

def test_flip_mirrors_x_and_changes_vx_sign():
    det=np.zeros((2,2,5),np.float32)
    det[:,0]=[[.2,.4,.1,.1,.9],[.4,.4,.1,.1,.9]]
    times=np.array([0.,1.])
    normal=_build_dense_features(det,times,1.5,False)
    flipped=_build_dense_features(det,times,1.5,True)
    assert flipped[0,_FEATURE_INDEX["ball_x_center_norm"]]==pytest.approx(.8)
    assert normal[1,_FEATURE_INDEX["ball_vx_norm_per_s"]]==pytest.approx(.2)
    assert flipped[1,_FEATURE_INDEX["ball_vx_norm_per_s"]]==pytest.approx(-.2)

def test_window_aggregation_handles_missing_detections():
    dense=np.zeros((5,len(FEATURE_NAMES)),np.float32)
    dense[0,_FEATURE_INDEX["ball_visible"]]=dense[0,_FEATURE_INDEX["ball_seen_before"]]=1
    dense[0,_FEATURE_INDEX["ball_x_center_norm"]]=.1
    dense[2,_FEATURE_INDEX["ball_visible"]]=1; dense[2,_FEATURE_INDEX["ball_x_center_norm"]]=.5
    dense[2,_FEATURE_INDEX["ball_rim_valid"]]=1; dense[2,_FEATURE_INDEX["ball_rim_distance_in_rim_units"]]=2
    dense[2,_FEATURE_INDEX["ball_crossed_rim_down"]]=1
    times=np.arange(5,dtype=float); out=_window_features(dense,times,[2,4])
    assert out[0,_FEATURE_INDEX["ball_visible_fraction_window"]]==pytest.approx(2/3)
    assert out[0,_FEATURE_INDEX["ball_last_x_center_norm_window"]]==pytest.approx(.5)
    assert out[0,_FEATURE_INDEX["ball_window_displacement_x_per_s"]]==pytest.approx(.2)
    assert out[0,_FEATURE_INDEX["ball_time_since_detection_s"]]==pytest.approx(0)
    assert out[0,_FEATURE_INDEX["ball_crossing_window_or"]]==1
    assert out[1,_FEATURE_INDEX["ball_seen_before"]]==1
    assert out[1,_FEATURE_INDEX["ball_time_since_detection_s"]]==pytest.approx(2)

def test_gate_rejects_distant_false_positive_and_chooses_plausible_candidate():
    candidates=np.zeros((3,2,5),np.float32)
    candidates[0,0]=[.1,.2,.02,.02,.7]
    candidates[1,0]=[.9,.8,.02,.02,.99] # falso positivo più confidente
    candidates[1,1]=[.11,.2,.02,.02,.8] # candidato plausibile
    candidates[2,0]=[.95,.9,.02,.02,.99] # nessun candidato plausibile
    accepted,rejected=_gate_ball(candidates,np.array([0.,1.,2.]),np.tile([100,100],(3,1)),.1,3.)
    assert accepted[1,4]==pytest.approx(.8)
    assert accepted[2,4]==0
    assert rejected==1

def test_assert_t_mismatch_against_reference(tmp_path):
    root=tmp_path; (root/"video.mp4").touch()
    (root/"manifest.csv").write_text("split,clip_id,path,label\ntrain,c1,video.mp4,x\n",encoding="utf-8")
    ref=root/"reference.pt"
    torch.save({"metadata":{"split":"train","experiment_id":"exp","experiment_config":{
        "sampler":{"name":"uniform","num_frames":2},"extra":{"transform":"none"}}},
        "samples":[{"id":"c1","data":torch.ones(3,4)}]},ref)
    cache_dir=root/"detections"/"det"/"train"; cache_dir.mkdir(parents=True)
    metadata={"detector_id":"det"}
    np.savez(cache_dir/"c1.npz",candidates=np.zeros((4,2,5,5),np.float32),
        timestamps_sec=np.arange(4,dtype=float),source_wh=np.tile([10,10],(4,1)),
        fps=np.float64(1),num_decoded_frames=np.int64(4),stream_frames=np.int64(4),
        metadata_json=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError,match="T diverso"):
        extract_tracking_features(Namespace(dataset_root=str(root),manifest=None,split="train",
            feature_reference=str(ref),expected_experiment_id="exp",detector_id="det",detections_root=None,
            confidence_threshold=.1,max_speed=1.,max_gap_s=2.,rim_expansion=1.5,output_dir=str(root/"out"),
            limit=None,overwrite=False,save_previews=False,show_rejected=False))


def test_track_mode_routes_detector_cache_and_gate_params(tmp_path, monkeypatch):
    from argparse import Namespace
    from data.experiment_config import ExperimentConfig
    from scripts import extract_dataset_features
    output=tmp_path/"features"; output.mkdir()
    experiment=ExperimentConfig(sampler={"name":"uniform","num_frames":2},backbone="x3d_l",
        extra={"transform":"none"})
    reference=output/experiment.feature_filename("train"); reference.touch()
    captured=[]
    monkeypatch.setattr("scripts.extract_tracking_features.extract_tracking_features",
                        lambda args:captured.append(args))
    args=Namespace(feature_reference=None,output_dir=output,split="train",dataset_root="dataset",
        manifest=None,track_detector_id="det123",track_detections_root=tmp_path/"detections",
        track_confidence_threshold=.3,track_max_speed=2.,track_max_gap_s=.5,track_rim_expansion=1.5,
        track_output_dir=None,track_limit=4,track_overwrite=False,track_save_previews=True,
        track_show_rejected=True)
    extract_dataset_features._extract_tracking_only(args,experiment)
    assert len(captured)==1
    assert captured[0].feature_reference==str(reference)
    assert captured[0].detector_id=="det123"
    assert captured[0].max_speed==2.
    assert captured[0].max_gap_s==.5
    assert captured[0].show_rejected is True

