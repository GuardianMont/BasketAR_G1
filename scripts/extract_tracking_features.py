"""Costruisce feature ball/rim dalla cache e le allinea al riferimento X3D."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch

from data.temporal_sampling import build_temporal_sampler


FEATURE_NAMES = (
    "ball_visible",
    "ball_confidence",
    "ball_x_center_norm",
    "ball_y_center_norm",
    "ball_width_norm",
    "ball_height_norm",
    "rim_visible",
    "rim_confidence",
    "rim_x_center_norm",
    "rim_y_center_norm",
    "rim_width_norm",
    "rim_height_norm",
    "ball_velocity_valid",
    "ball_vx_norm_per_s",
    "ball_vy_norm_per_s",
    "ball_speed_norm_per_s",
    "ball_acceleration_valid",
    "ball_ax_norm_per_s2",
    "ball_ay_norm_per_s2",
    "ball_acceleration_norm_per_s2",
    "ball_rim_valid",
    "ball_rim_dx_in_rim_widths",
    "ball_rim_dy_in_rim_heights",
    "ball_rim_distance_in_rim_units",
    "ball_above_rim",
    "ball_below_rim",
    "ball_center_inside_expanded_rim_box",
    "ball_rim_motion_valid",
    "ball_rim_approach_speed_per_s",
    "rim_crossing_valid",
    "ball_crossed_rim_down",
    "ball_visible_fraction_window", "ball_last_x_center_norm_window", "ball_last_y_center_norm_window",
    "ball_window_displacement_x_per_s", "ball_window_displacement_y_per_s", "ball_window_motion_valid",
    "ball_window_max_speed_norm_per_s", "ball_rim_min_distance_window", "ball_rim_window_valid",
    "ball_crossing_window_or", "ball_time_since_detection_s", "ball_seen_before",
    "ball_window_position_valid", "ball_window_speed_valid",
)
FEATURE_SCHEMA_VERSION = 2
FEATURE_GROUPS = {"detection": list(FEATURE_NAMES[:12]), "kinematics": list(FEATURE_NAMES[12:20]),
                  "rim_relation": list(FEATURE_NAMES[20:31]), "window_aggregates": list(FEATURE_NAMES[31:])}

_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary_path, path)


def _load_reference(path: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"File feature di riferimento non trovato: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("Il file di riferimento deve contenere metadata valido.")
    metadata = payload["metadata"]
    if not isinstance(metadata.get("experiment_id"), str) or not metadata["experiment_id"]:
        raise ValueError("Il file di riferimento non contiene metadata['experiment_id'] valido.")
    if metadata.get("split") != split:
        raise ValueError(
            f"Lo split nel riferimento è {metadata.get('split')!r}, richiesto {split!r}."
        )
    experiment_config = metadata.get("experiment_config")
    if not isinstance(experiment_config, dict):
        raise ValueError("Manca metadata['experiment_config'] nel riferimento.")
    sampler_config = experiment_config.get("sampler")
    if not isinstance(sampler_config, dict):
        raise ValueError("Manca la configurazione sampler nell'esperimento X3D.")
    # Fail early for unknown or malformed sampler configurations.
    build_temporal_sampler(sampler_config)

    extra = experiment_config.get("extra")
    if not isinstance(extra, dict) or "transform" not in extra:
        raise ValueError(
            "Il riferimento deve registrare extra.transform per verificare "
            "l'allineamento temporale e l'eventuale flip."
        )
    transform = extra["transform"]
    if transform not in {"none", "horizontal_flip"}:
        raise ValueError(
            f"Transform {transform!r} non supportato per l'allineamento del tracking. "
            "RandomTemporalShift non salva gli indici scelti per clip."
        )

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Il riferimento X3D non contiene samples validi.")
    reference_samples = []
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("id") is None:
            raise ValueError("Il riferimento X3D contiene un sample senza ID valido.")
        data = sample.get("data")
        if (
            not isinstance(data, torch.Tensor)
            or data.ndim != 2
            or data.shape[0] <= 0
            or data.shape[1] <= 0
        ):
            raise ValueError(f"Feature non valide nel sample {sample.get('id')!r}.")
        reference_samples.append(
            {"id": str(sample["id"]), "num_steps": int(data.shape[0])}
        )
    ids = [sample["id"] for sample in reference_samples]
    if len(ids) != len(samples) or len(ids) != len(set(ids)):
        raise ValueError("Gli ID nel file di riferimento sono mancanti o duplicati.")

    alignment = {
        "experiment_id": metadata.get("experiment_id"),
        "experiment_config": experiment_config,
        "sampler_config": sampler_config,
        "transform": transform,
        "reference_sample_ids": ids,
    }
    del payload, samples
    gc.collect()
    return alignment, reference_samples


def _load_manifest(
    manifest_path: Path,
    dataset_root: Path,
    split: str,
    reference_ids: list[str],
) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"split", "clip_id", "path", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Colonne mancanti nel manifest: {sorted(missing)}")
        rows = [row for row in reader if row["split"] == split]

    rows_by_id = {row["clip_id"]: row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError(f"clip_id duplicati nello split {split!r} del manifest.")

    missing_ids = [clip_id for clip_id in reference_ids if clip_id not in rows_by_id]
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} clip_id del riferimento non sono nello split "
            f"{split!r} del manifest; primi ID: {missing_ids[:5]}"
        )

    selected = []
    for clip_id in reference_ids:
        row = dict(rows_by_id[clip_id])
        video_path = (dataset_root / row["path"]).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video mancante per {clip_id}: {video_path}")
        row["_absolute_path"] = str(video_path)
        selected.append(row)
    return selected


def _build_dense_features(
    detections: np.ndarray,
    timestamps: np.ndarray,
    rim_expansion: float,
    horizontal_flip: bool,
) -> np.ndarray:
    # detections: (T, ball/rim, [cx, cy, w, h, conf]); absent detections are all zero.
    observations = detections.copy()
    visible = observations[:, :, 4] > 0.0
    if horizontal_flip:
        observations[:, :, 0] = np.where(
            visible, 1.0 - observations[:, :, 0], observations[:, :, 0]
        )

    length = len(detections)
    features = np.zeros((length, len(FEATURE_NAMES)), dtype=np.float32)

    for object_index, prefix in enumerate(("ball", "rim")):
        is_visible = visible[:, object_index]
        observations[:, object_index, :4][~is_visible] = 0.0
        columns = {
            "visible": is_visible.astype(np.float32),
            "confidence": observations[:, object_index, 4],
            "x_center_norm": observations[:, object_index, 0],
            "y_center_norm": observations[:, object_index, 1],
            "width_norm": observations[:, object_index, 2],
            "height_norm": observations[:, object_index, 3],
        }
        for column, values in columns.items():
            features[:, _FEATURE_INDEX[f"{prefix}_{column}"]] = values

    ball_visible = visible[:, 0]
    rim_visible = visible[:, 1]
    ball_x = observations[:, 0, 0]
    ball_y = observations[:, 0, 1]
    rim_x = observations[:, 1, 0]
    rim_y = observations[:, 1, 1]
    rim_w = observations[:, 1, 2]
    rim_h = observations[:, 1, 3]

    velocity_valid = np.zeros(length, dtype=bool)
    relative_valid = ball_visible & rim_visible & (rim_w > 0) & (rim_h > 0)
    relative_dx = np.zeros(length, dtype=np.float64)
    relative_dy = np.zeros(length, dtype=np.float64)

    for index in range(length):
        if index > 0 and ball_visible[index] and ball_visible[index - 1]:
            dt = timestamps[index] - timestamps[index - 1]
            if dt > 0:
                vx = (ball_x[index] - ball_x[index - 1]) / dt
                vy = (ball_y[index] - ball_y[index - 1]) / dt
                features[index, _FEATURE_INDEX["ball_velocity_valid"]] = 1.0
                features[index, _FEATURE_INDEX["ball_vx_norm_per_s"]] = vx
                features[index, _FEATURE_INDEX["ball_vy_norm_per_s"]] = vy
                features[index, _FEATURE_INDEX["ball_speed_norm_per_s"]] = math.hypot(vx, vy)
                velocity_valid[index] = True

        if relative_valid[index]:
            relative_dx[index] = (ball_x[index] - rim_x[index]) / rim_w[index]
            relative_dy[index] = (ball_y[index] - rim_y[index]) / rim_h[index]
            features[index, _FEATURE_INDEX["ball_rim_valid"]] = 1.0
            features[index, _FEATURE_INDEX["ball_rim_dx_in_rim_widths"]] = relative_dx[index]
            features[index, _FEATURE_INDEX["ball_rim_dy_in_rim_heights"]] = relative_dy[index]
            features[index, _FEATURE_INDEX["ball_rim_distance_in_rim_units"]] = math.hypot(
                relative_dx[index], relative_dy[index]
            )
            features[index, _FEATURE_INDEX["ball_above_rim"]] = float(relative_dy[index] < 0)
            features[index, _FEATURE_INDEX["ball_below_rim"]] = float(relative_dy[index] > 0)
            features[index, _FEATURE_INDEX["ball_center_inside_expanded_rim_box"]] = float(
                abs(relative_dx[index]) <= rim_expansion / 2
                and abs(relative_dy[index]) <= rim_expansion / 2
            )

    for index in range(2, length):
        if velocity_valid[index] and velocity_valid[index - 1]:
            dt = (timestamps[index] - timestamps[index - 2]) / 2.0
            if dt > 0:
                vx_now = features[index, _FEATURE_INDEX["ball_vx_norm_per_s"]]
                vy_now = features[index, _FEATURE_INDEX["ball_vy_norm_per_s"]]
                vx_prev = features[index - 1, _FEATURE_INDEX["ball_vx_norm_per_s"]]
                vy_prev = features[index - 1, _FEATURE_INDEX["ball_vy_norm_per_s"]]
                ax = (vx_now - vx_prev) / dt
                ay = (vy_now - vy_prev) / dt
                features[index, _FEATURE_INDEX["ball_acceleration_valid"]] = 1.0
                features[index, _FEATURE_INDEX["ball_ax_norm_per_s2"]] = ax
                features[index, _FEATURE_INDEX["ball_ay_norm_per_s2"]] = ay
                features[index, _FEATURE_INDEX["ball_acceleration_norm_per_s2"]] = math.hypot(ax, ay)

    for index in range(1, length):
        if relative_valid[index] and relative_valid[index - 1]:
            dt = timestamps[index] - timestamps[index - 1]
            if dt > 0:
                features[index, _FEATURE_INDEX["ball_rim_motion_valid"]] = 1.0
                prev_distance = math.hypot(relative_dx[index - 1], relative_dy[index - 1])
                curr_distance = math.hypot(relative_dx[index], relative_dy[index])
                features[index, _FEATURE_INDEX["ball_rim_approach_speed_per_s"]] = (
                    prev_distance - curr_distance
                ) / dt
                features[index, _FEATURE_INDEX["rim_crossing_valid"]] = 1.0
                if (
                    relative_dy[index - 1] < 0 <= relative_dy[index]
                    and abs(relative_dx[index]) <= rim_expansion / 2
                ):
                    features[index, _FEATURE_INDEX["ball_crossed_rim_down"]] = 1.0

    return features


def _resample_features(
    dense_features: np.ndarray,
    sampled_indices: list[int],
) -> np.ndarray:
    sampled = dense_features[sampled_indices].copy()
    event_index = _FEATURE_INDEX["ball_crossed_rim_down"]
    valid_index = _FEATURE_INDEX["rim_crossing_valid"]
    previous = -1
    for output_index, frame_index in enumerate(sampled_indices):
        start = previous + 1
        # Assign dense events to the next X3D step; include the unsampled tail
        # in the last step so a crossing after the final sample is not dropped.
        end = len(dense_features) - 1 if output_index == len(sampled_indices) - 1 else frame_index
        if start <= end:
            interval = dense_features[start : end + 1]
            sampled[output_index, event_index] = float(np.any(interval[:, event_index] > 0))
            sampled[output_index, valid_index] = float(np.any(interval[:, valid_index] > 0))
        previous = frame_index
    return sampled


def _save_clip(
    path: Path,
    dense_features: np.ndarray,
    sampled_features: np.ndarray,
    dense_timestamps: np.ndarray,
    sampled_indices: list[int],
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            features=sampled_features.astype(np.float32),
            frame_indices=np.asarray(sampled_indices, dtype=np.int64),
            timestamps_sec=dense_timestamps[sampled_indices].astype(np.float64),
            dense_features=dense_features.astype(np.float32),
            dense_frame_indices=np.arange(len(dense_features), dtype=np.int64),
            dense_timestamps_sec=dense_timestamps.astype(np.float64),
        )
    os.replace(temporary_path, path)



def _candidate_rows(cache, slot, threshold):
    rows=cache["candidates"][:,slot].copy()
    rows[rows[:,:,4] < threshold] = 0
    return rows

def _gate_ball(candidates,times,sizes,max_speed,max_gap_s):
    accepted=np.zeros((len(candidates),5),np.float32); rejected=0; last=None; last_t=None
    for i,rows in enumerate(candidates):
        rows=rows[rows[:,4]>0]
        if not len(rows): continue
        t=float(times[i])
        if last_t is not None and t-last_t>max_gap_s: last=last_t=None
        if last is None:
            chosen=rows[int(np.argmax(rows[:,4]))]
            accepted[i]=chosen; last=(float(chosen[0]),float(chosen[1])); last_t=t; continue
        dt=t-last_t
        plausible=[]
        if dt>0:
            for candidate in rows:
                d=math.hypot((float(candidate[0])-last[0])*int(sizes[i,0])/int(sizes[i,1]),float(candidate[1])-last[1])
                if d<=max_speed*dt: plausible.append(candidate)
        if plausible:
            chosen=max(plausible,key=lambda item:float(item[4]))
            accepted[i]=chosen; last=(float(chosen[0]),float(chosen[1])); last_t=t
        else: rejected+=1
    return accepted,rejected

def _window_features(dense,times,indices):
    result=_resample_features(dense,indices)
    for oi,idx in enumerate(indices):
        start=0 if oi==0 else indices[oi-1]+1; end=len(dense)-1 if oi==len(indices)-1 else idx
        win=dense[start:end+1]
        if not len(win): continue
        vi=_FEATURE_INDEX["ball_visible"]; seen=np.flatnonzero(win[:,vi]>0)
        result[oi,_FEATURE_INDEX["ball_visible_fraction_window"]]=float(np.mean(win[:,vi]>0))
        result[oi,_FEATURE_INDEX["ball_crossing_window_or"]]=float(np.any(win[:,_FEATURE_INDEX["ball_crossed_rim_down"]]>0))
        history=np.flatnonzero(dense[:end+1,vi]>0)
        if len(history):
            last_seen=int(history[-1]); result[oi,_FEATURE_INDEX["ball_seen_before"]]=1
            result[oi,_FEATURE_INDEX["ball_time_since_detection_s"]]=max(0.,float(times[end]-times[last_seen]))
        if len(seen):
            result[oi,_FEATURE_INDEX["ball_window_position_valid"]]=1
            last=start+int(seen[-1])
            result[oi,_FEATURE_INDEX["ball_last_x_center_norm_window"]]=dense[last,_FEATURE_INDEX["ball_x_center_norm"]]
            result[oi,_FEATURE_INDEX["ball_last_y_center_norm_window"]]=dense[last,_FEATURE_INDEX["ball_y_center_norm"]]
            if len(seen)>1:
                first=start+int(seen[0]); dt=float(times[last]-times[first])
                if dt>0:
                    result[oi,_FEATURE_INDEX["ball_window_displacement_x_per_s"]]=(dense[last,_FEATURE_INDEX["ball_x_center_norm"]]-dense[first,_FEATURE_INDEX["ball_x_center_norm"]])/dt
                    result[oi,_FEATURE_INDEX["ball_window_displacement_y_per_s"]]=(dense[last,_FEATURE_INDEX["ball_y_center_norm"]]-dense[first,_FEATURE_INDEX["ball_y_center_norm"]])/dt
                    result[oi,_FEATURE_INDEX["ball_window_motion_valid"]]=1
        v=win[win[:,_FEATURE_INDEX["ball_velocity_valid"]]>0,_FEATURE_INDEX["ball_speed_norm_per_s"]]
        if len(v):
            result[oi,_FEATURE_INDEX["ball_window_max_speed_norm_per_s"]]=float(np.max(v))
            result[oi,_FEATURE_INDEX["ball_window_speed_valid"]]=1
        d=win[win[:,_FEATURE_INDEX["ball_rim_valid"]]>0,_FEATURE_INDEX["ball_rim_distance_in_rim_units"]]
        if len(d): result[oi,_FEATURE_INDEX["ball_rim_min_distance_window"]]=float(np.min(d)); result[oi,_FEATURE_INDEX["ball_rim_window_valid"]]=1
    return result

def _hash_tracker(detector_id,threshold,max_speed,max_gap_s,rim_expansion):
    cfg={"detector_id":detector_id,"threshold":threshold,"max_speed":max_speed,"max_gap_s":max_gap_s,
         "rim_expansion":rim_expansion,"feature_schema":FEATURE_SCHEMA_VERSION}
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:16]

def _write_tracking_preview(video_path,output_path,accepted,timestamps,fps,rejected=None,rim=None):
    import cv2
    if not fps or fps<=0:
        d=np.diff(timestamps); d=d[d>0]; fps=1/float(np.median(d)) if len(d) else 25.
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with av.open(str(video_path)) as container:
        writer=None
        try:
            for i,frame in enumerate(container.decode(video=0)):
                image=frame.to_ndarray(format="bgr24"); h,w=image.shape[:2]
                if writer is None:
                    writer=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*"mp4v"),float(fps),(w,h))
                    if not writer.isOpened(): raise RuntimeError(f"Impossibile creare anteprima: {output_path}")
                for arr,color,label in ((accepted,(0,255,0),"PALLA ACCETTATA"),(rim,(0,165,255),"FERRO"),(rejected,(128,128,128),"SCARTATA")):
                    if arr is None or i>=len(arr): continue
                    cx,cy,bw,bh,conf=arr[i]
                    if conf<=0: continue
                    x1,y1,x2,y2=int((cx-bw/2)*w),int((cy-bh/2)*h),int((cx+bw/2)*w),int((cy+bh/2)*h)
                    cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
                    cv2.putText(image,f"{label} {conf:.2f}",(x1,max(18,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
                writer.write(image)
        finally:
            if writer is not None: writer.release()

def extract_tracking_features(args):
    root=Path(args.dataset_root).resolve(); alignment,samples=_load_reference(Path(args.feature_reference).resolve(),args.split)
    expected=getattr(args,"expected_experiment_id",None)
    if expected and alignment["experiment_id"]!=expected: raise ValueError("experiment_id X3D non corrispondente")
    if args.limit is not None:
        if args.limit<=0: raise ValueError("limit deve essere > 0")
        samples=samples[:args.limit]; alignment["reference_sample_ids"]=alignment["reference_sample_ids"][:args.limit]
    manifest=Path(args.manifest).resolve() if args.manifest else root/"manifest.csv"
    rows=_load_manifest(manifest,root,args.split,alignment["reference_sample_ids"])
    cache_dir=Path(args.detections_root or root/"detections")/args.detector_id/args.split
    outdir=Path(args.output_dir or root/"tracking"); outdir.mkdir(parents=True,exist_ok=True)
    tid=_hash_tracker(args.detector_id,args.confidence_threshold,args.max_speed,args.max_gap_s,args.rim_expansion)
    outpath=outdir/f"{args.split}__{alignment['experiment_id']}__trk-{tid}.pt"
    sampler=build_temporal_sampler(alignment["sampler_config"]); outputs=[]
    diag={"schema_version":2,"detector_id":args.detector_id,"tracker_id":tid,"clips":{}}
    for n,(row,ref) in enumerate(zip(rows,samples),1):
        if row["clip_id"]!=ref["id"]: raise ValueError("Ordine/ID reference non coerente col manifest")
        path=cache_dir/f"{ref['id']}.npz"
        if not path.is_file(): raise FileNotFoundError(f"Cache detection mancante: {path}")
        c=np.load(path,allow_pickle=False); cm=json.loads(str(c["metadata_json"].item()))
        if cm.get("detector_id")!=args.detector_id: raise ValueError("detector_id cache errato")
        times=c["timestamps_sec"].astype(np.float64); sizes=c["source_wh"]; cand=c["candidates"]
        if len(cand)!=int(c["num_decoded_frames"]): raise ValueError("Conteggio cache incoerente")
        if not np.all(np.isfinite(times)) or np.any(np.diff(times)<0): raise ValueError(f"Timestamp cache non validi: {ref['id']}")
        ballraw=_candidate_rows(c,0,args.confidence_threshold); rimraw=_candidate_rows(c,1,args.confidence_threshold)
        ball,ngate=_gate_ball(ballraw,times,sizes,args.max_speed,args.max_gap_s)
        ballraw_best=np.zeros_like(ball)
        for bi, candidates in enumerate(ballraw):
            valid=candidates[candidates[:,4]>0]
            if len(valid): ballraw_best[bi]=valid[np.argmax(valid[:,4])]
        rejected=np.where(((ballraw_best[:,4]>0)&(ball[:,4]==0))[:,None],ballraw_best,0)
        rim=np.zeros((len(rimraw),5),np.float32)
        for ri,candidates in enumerate(rimraw):
            valid=candidates[candidates[:,4]>0]
            if len(valid): rim[ri]=valid[np.argmax(valid[:,4])]
        dense=_build_dense_features(np.stack((ball,rim),axis=1),times,args.rim_expansion,alignment["transform"]=="horizontal_flip")
        fps=float(c["fps"]); count=int(c["stream_frames"]) or len(dense)
        indices=sampler.sample_indices(count,fps=fps if fps>0 else None)
        if any(i<0 or i>=len(dense) for i in indices): raise ValueError(f"Indici sampler non decodificati: {ref['id']}")
        if len(indices)!=ref["num_steps"]: raise ValueError(f"T diverso dal reference per {ref['id']}: {len(indices)} != {ref['num_steps']}")
        sampled=_window_features(dense,times,indices)
        assert sampled.shape[0]==ref["num_steps"],f"T diverso dal reference per {ref['id']}"
        outputs.append({"id":ref["id"],"data":torch.from_numpy(sampled.astype(np.float32))})
        count_cand=int(np.count_nonzero(np.any(ballraw[:,:,4]>0,axis=1)))
        diag["clips"][ref["id"]]={"ball_visibility_ratio":float(np.mean(ball[:,4]>0)),"rim_visibility_ratio":float(np.mean(rim[:,4]>0)),
            "gate_rejected_fraction":ngate/max(1,count_cand),"crossings":int(np.count_nonzero(dense[:,_FEATURE_INDEX["ball_crossed_rim_down"]]))}
        if args.save_previews: _write_tracking_preview(Path(row["_absolute_path"]),outdir/"previews"/f"{ref['id']}.mp4",ball,times,fps,rejected if args.show_rejected else None,rim)
        print(f"[{n}/{len(rows)}] {ref['id']}: {len(dense)} frame densi -> {len(sampled)} passi X3D")
    metadata={"split":args.split,"experiment_id":alignment["experiment_id"],"experiment_config":alignment["experiment_config"],
        "detector_id":args.detector_id,"tracker_id":tid,"feature_schema_version":2,"feature_names":list(FEATURE_NAMES),
        "feature_groups":FEATURE_GROUPS,"missing_detection_policy":"Assenze a zero con maschere; nessun carry-forward o interpolazione."}
    if outpath.exists() and not args.overwrite: raise FileExistsError(f"Sidecar già presente: {outpath}")
    tmp=outpath.with_suffix(".pt.tmp"); torch.save({"metadata":metadata,"samples":outputs},tmp); os.replace(tmp,outpath)
    _write_json_atomic(outpath.with_suffix(".diagnostics.json"),diag); print(f"Sidecar scritto: {outpath}")

def build_parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--dataset-root",default="dataset_hd"); p.add_argument("--manifest")
    p.add_argument("--split",choices=("train","val","test"),required=True); p.add_argument("--feature-reference",required=True)
    p.add_argument("--expected-experiment-id"); p.add_argument("--detector-id",required=True); p.add_argument("--detections-root")
    p.add_argument("--confidence-threshold",type=float,required=True); p.add_argument("--max-speed",type=float,required=True)
    p.add_argument("--max-gap-s",type=float,required=True); p.add_argument("--rim-expansion",type=float,default=1.5)
    p.add_argument("--output-dir"); p.add_argument("--limit",type=int); p.add_argument("--overwrite",action="store_true")
    p.add_argument("--save-previews",action="store_true"); p.add_argument("--show-rejected",action="store_true"); return p
def main():
    a=build_parser().parse_args()
    if not 0<=a.confidence_threshold<=1: raise ValueError("confidence-threshold deve essere in [0,1]")
    if a.max_speed<=0 or a.max_gap_s<=0: raise ValueError("max-speed e max-gap-s devono essere > 0")
    if a.rim_expansion<1: raise ValueError("rim-expansion deve essere >= 1")
    extract_tracking_features(a)

if __name__ == "__main__":
    main()
