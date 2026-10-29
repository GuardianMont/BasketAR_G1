"""Estrae candidati YOLO per clip, indipendenti dagli esperimenti X3D."""
from __future__ import annotations
import argparse, csv, hashlib, json, os
from pathlib import Path
import av
import numpy as np
CACHE_SCHEMA_VERSION = 1

def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def detector_identity(weights_hash, imgsz, ball_class_id, rim_class_id, min_confidence, topk_ball, topk_rim):
    payload = {"schema_version": CACHE_SCHEMA_VERSION, "weights_sha256": weights_hash, "imgsz": imgsz,
               "ball_class_id": ball_class_id, "rim_class_id": rim_class_id,
               "min_confidence": min_confidence, "topk_ball": topk_ball, "topk_rim": topk_rim}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

def _manifest(path, root, split):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "clip_id", "path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest: mancano le colonne {sorted(required - set(reader.fieldnames or []))}")
        rows = [row for row in reader if row["split"] == split]
    if len({row["clip_id"] for row in rows}) != len(rows):
        raise ValueError(f"clip_id duplicati nello split {split}")
    return [(row, (Path(root) / row["path"]).resolve()) for row in rows]

def _infer(pending, model, args, records, timestamps, pts, sizes, fps):
    results = model.predict(source=[item[0] for item in pending], imgsz=args.imgsz,
                            conf=args.min_confidence, device=args.device, verbose=False)
    for item, result in zip(pending, results):
        image, frame_pts, time_base, width, height = item
        candidates = []
        boxes = result.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy() if boxes is not None and len(boxes) else np.zeros((0,4))
        conf = boxes.conf.detach().cpu().numpy() if boxes is not None and len(boxes) else np.zeros(0)
        classes = boxes.cls.detach().cpu().numpy().astype(np.int64) if boxes is not None and len(boxes) else np.zeros(0, dtype=np.int64)
        for class_id, topk in ((args.ball_class_id, args.topk_ball), (args.rim_class_id, args.topk_rim)):
            selected = np.flatnonzero((classes == class_id) & (conf >= args.min_confidence))
            selected = selected[np.argsort(conf[selected])[::-1][:topk]]
            rows = np.zeros((max(args.topk_ball, args.topk_rim), 5), np.float32)
            for slot, idx in enumerate(selected):
                x1,y1,x2,y2 = xyxy[idx]
                rows[slot] = ((x1+x2)/(2*width),(y1+y2)/(2*height),max(0.,x2-x1)/width,max(0.,y2-y1)/height,conf[idx])
            candidates.append(rows)
        records.append(np.stack(candidates))
        pts.append(-1 if frame_pts is None else int(frame_pts))
        timestamps.append(float(frame_pts*time_base) if frame_pts is not None and time_base is not None
                          else (len(timestamps)/fps if fps > 0 else float(len(timestamps))))
        sizes.append((width,height))

def _extract_clip(video_path, model, args):
    records,timestamps,pts,sizes=[],[],[],[]
    with av.open(str(video_path)) as container:
        if not container.streams.video: raise ValueError(f"Nessuno stream video in {video_path}")
        stream=container.streams.video[0]; reported=int(stream.frames or 0)
        fps=float(stream.average_rate) if stream.average_rate else 0.
        pending=[]
        for frame in container.decode(video=0):
            image=frame.to_ndarray(format="bgr24"); h,w=image.shape[:2]
            pending.append((image,frame.pts,frame.time_base,w,h))
            if len(pending)>=args.batch_size:
                _infer(pending,model,args,records,timestamps,pts,sizes,fps); pending.clear()
        if pending: _infer(pending,model,args,records,timestamps,pts,sizes,fps)
    if not records: raise ValueError(f"Nessun frame decodificabile: {video_path}")
    return np.stack(records),np.asarray(timestamps,np.float64),np.asarray(pts,np.int64),np.asarray(sizes,np.int32),fps,len(records),reported

def extract_detections(args):
    from ultralytics import YOLO
    root=Path(args.dataset_root).resolve(); manifest=Path(args.manifest).resolve() if args.manifest else root/"manifest.csv"
    weights=Path(args.weights).resolve()
    if not weights.is_file(): raise FileNotFoundError(f"Checkpoint detector non trovato: {weights}")
    if args.imgsz<=0 or args.batch_size<=0 or args.topk_ball<=0 or args.topk_rim<=0: raise ValueError("imgsz, batch-size e topk devono essere > 0")
    if not 0<=args.min_confidence<=1: raise ValueError("min-confidence deve essere in [0, 1]")
    if args.ball_class_id==args.rim_class_id: raise ValueError("I class ID ball e rim devono essere diversi")
    whash=_sha256_file(weights); did=detector_identity(whash,args.imgsz,args.ball_class_id,args.rim_class_id,args.min_confidence,args.topk_ball,args.topk_rim)
    output=Path(args.output_root or root/"detections")/did/args.split
    rows=_manifest(manifest,root,args.split)
    if args.limit is not None:
        if args.limit<=0: raise ValueError("limit deve essere > 0")
        rows=rows[:args.limit]
    model=YOLO(str(weights)); output.mkdir(parents=True,exist_ok=True)
    print(f"detector_id={did} | cache={output}")
    metadata={"schema_version":CACHE_SCHEMA_VERSION,"detector_id":did,"weights_sha256":whash,"imgsz":args.imgsz,
              "ball_class_id":args.ball_class_id,"rim_class_id":args.rim_class_id,"min_confidence":args.min_confidence,
              "topk_ball":args.topk_ball,"topk_rim":args.topk_rim}
    for n,(row,video) in enumerate(rows,1):
        dest=output/f"{row['clip_id']}.npz"
        if dest.exists() and not args.overwrite:
            print(f"[{n}/{len(rows)}] {row['clip_id']}: cache presente, salto"); continue
        cand,times,pts,sizes,fps,decoded,reported=_extract_clip(video,model,args)
        temp=dest.with_suffix(".npz.tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream,candidates=cand,timestamps_sec=times,pts=pts,source_wh=sizes,
                fps=np.float64(fps),num_decoded_frames=np.int64(decoded),stream_frames=np.int64(reported),
                metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
        os.replace(temp,dest); print(f"[{n}/{len(rows)}] {row['clip_id']}: {decoded} frame salvati")

def build_parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--dataset-root",default="dataset_hd"); p.add_argument("--manifest")
    p.add_argument("--split",choices=("train","val","test"),required=True); p.add_argument("--weights",required=True)
    p.add_argument("--imgsz",type=int,required=True); p.add_argument("--ball-class-id",type=int,required=True); p.add_argument("--rim-class-id",type=int,required=True)
    p.add_argument("--min-confidence",type=float,required=True); p.add_argument("--topk-ball",type=int,default=5); p.add_argument("--topk-rim",type=int,default=2)
    p.add_argument("--output-root"); p.add_argument("--device"); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--limit",type=int); p.add_argument("--overwrite",action="store_true")
    return p
if __name__=="__main__": extract_detections(build_parser().parse_args())
