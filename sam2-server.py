from __future__ import annotations

import gc
import glob
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Ensure sam2-main is importable when running from repo root.
_ROOT = Path(__file__).resolve().parent
_SAM2_MAIN = _ROOT / "sam2-main"
if _SAM2_MAIN.exists() and str(_SAM2_MAIN) not in sys.path:
    sys.path.insert(0, str(_SAM2_MAIN))

from sam2.build_sam import build_sam2_video_predictor

class InitVideoRequest(BaseModel):
    video_path: str


class TrackRequest(BaseModel):
    video_path: str
    start_frame_idx: int
    obj_id: int
    bbox_xyxy: List[float]  # [x1,y1,x2,y2] in pixel coords
    forward_frames: int
    frame_paths: List[str]  # ordered list of absolute frame paths (length <= 150)


class TrackResponse(BaseModel):
    result_file_path: str
    processing_time: float
    frame_count: int


class HealthResponse(BaseModel):
    ok: bool
    device: str
    model_cfg: str
    checkpoint_path: str

class _ServiceState:
    def __init__(
        self,
        *,
        model_cfg: str,
        checkpoint_path: str,
        device: str | None,
    ):
        if device is None or not str(device).strip():
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.model_cfg = str(model_cfg)
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(str(device))

        if self.device.type == "cuda":
            try:
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

        if self.device.type == "cuda":
            self.autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            self.autocast_ctx = torch.no_grad()

        # Build predictor once; per-request we will init_state on a tiny clip dir.
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint_path, device=self.device)

        # Single-flight lock (GPU model, avoid concurrent runs)
        import threading

        self.lock = threading.Lock()


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    if mask is None:
        return None
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    coords = np.where(mask)
    if len(coords[0]) == 0 or len(coords[1]) == 0:
        return None
    y1, y2 = int(coords[0].min()), int(coords[0].max())
    x1, x2 = int(coords[1].min()), int(coords[1].max())
    # +1 to make width/height inclusive in pixel space
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    return [int(x1), int(y1), int(w), int(h)]


def _looks_normalized_xyxy(b: list[float]) -> bool:
    # Heuristic: all coords in [0, ~1] (allow small epsilon), and x2>=x1,y2>=y1.
    try:
        x1, y1, x2, y2 = [float(v) for v in b]
    except Exception:
        return False
    if x2 < x1 or y2 < y1:
        return False
    m = max(abs(x1), abs(y1), abs(x2), abs(y2))
    return m <= 1.5

# 新增：只处理JSON键名的转换函数，保持bounding box的整数类型
# 此函数只将整数类型的键转换为字符串，不改变值的类型
# 因为JSON的键不能是整数，但值可以保持原始类型

def convert_keys_for_json(obj):
    if isinstance(obj, dict):
        return {str(key): convert_keys_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_for_json(item) for item in obj]
    else:
        # 保持原始值的类型，不做任何转换
        return obj

# 新增：缓存管理函数
def manage_cache(tmp_dir: str, max_files: int | None = None) -> None:
    """
    管理临时文件夹中的文件，删除超过最大数量的旧文件
    
    参数:
        tmp_dir: 临时文件夹路径
        max_files: 最大文件数量限制，如果不指定则使用配置中的值
    """
    if max_files is None:
        max_files = MAX_CACHE_FILES
    # 确保临时文件夹存在
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 获取文件夹中的所有JSON文件
    json_files = glob.glob(os.path.join(tmp_dir, "*.json"))
    
    # 如果文件数量超过限制，删除最早创建的文件
    if len(json_files) >= max_files:
        # 按照文件创建时间排序
        json_files.sort(key=os.path.getctime)
        
        # 计算需要删除的文件数量
        files_to_delete = json_files[:len(json_files) - max_files + 1]
        
        # 删除文件
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"已删除旧缓存文件: {file_path}")
            except Exception as e:
                print(f"删除文件失败 {file_path}: {str(e)}")

# 创建FastAPI应用
app = FastAPI(title="SAM2视频目标跟踪服务")

# 全局变量用于存储跟踪器实例
STATE: _ServiceState | None = None

# 配置文件路径和配置
_SETTINGS_PATH = _ROOT / "jh4player_settings.json"

def _load_settings() -> dict:
    """从配置文件加载设置"""
    if _SETTINGS_PATH.exists():
        try:
            return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

_CONFIG = _load_settings()
TMP_DIR = _CONFIG.get("sam2_tmp_dir", "./tmp/sam2_tracking_results")
MAX_CACHE_FILES = int(_CONFIG.get("sam2_max_cache_files", 200))


def _safe_int(v: Any, *, name: str) -> int:
    try:
        return int(v)
    except Exception:
        raise HTTPException(status_code=400, detail=f"invalid {name}")


def _link_or_copy(src: str, dst: str) -> None:
    # Prefer hardlink (fast, no extra disk); fallback to copy.
    try:
        os.link(src, dst)
        return
    except Exception:
        pass
    shutil.copy2(src, dst)


def _make_clip_dir(frame_paths: list[str]) -> Path:
    clip_id = uuid.uuid4().hex
    clip_dir = Path(TMP_DIR) / f"clip_{clip_id}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    # Use numeric filenames to guarantee sort order in sam2 loader.
    for i, src in enumerate(frame_paths):
        src_p = Path(src)
        if not src_p.exists():
            raise HTTPException(status_code=400, detail=f"frame not found: {src}")
        ext = src_p.suffix or ".jpg"
        dst = clip_dir / f"{i:05d}{ext}"
        _link_or_copy(str(src_p), str(dst))
    return clip_dir

@app.on_event("startup")
async def startup_event():
    global STATE
    os.makedirs(TMP_DIR, exist_ok=True)

    default_model_cfg = _SAM2_MAIN / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_s.yaml"
    default_checkpoint = _SAM2_MAIN / "checkpoints" / "sam2.1_hiera_small.pt"

    # 优先从配置文件读取，其次从环境变量读取，最后使用默认值
    model_cfg = _CONFIG.get("sam2_model_cfg") or os.environ.get("SAM2_MODEL_CFG") or str(default_model_cfg)
    checkpoint_path = _CONFIG.get("sam2_checkpoint") or os.environ.get("SAM2_CHECKPOINT") or str(default_checkpoint)
    device = _CONFIG.get("sam2_device") or os.environ.get("SAM2_DEVICE")

    # Optional: reset compilation caches for a clean run (best-effort).
    try:
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
            torch.compiler.reset()
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "reset"):
            torch._dynamo.reset()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    print("服务启动中，加载 SAM2 模型...")
    STATE = _ServiceState(model_cfg=model_cfg, checkpoint_path=checkpoint_path, device=device)
    print(f"SAM2 模型已加载，device={STATE.device}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if STATE is None:
        raise HTTPException(status_code=503, detail="model not ready")
    return HealthResponse(ok=True, device=str(STATE.device), model_cfg=STATE.model_cfg, checkpoint_path=STATE.checkpoint_path)


@app.post("/init_video")
async def init_video(req: InitVideoRequest):
    # Intentionally lightweight: validate that the base frame dir exists.
    if STATE is None:
        raise HTTPException(status_code=503, detail="model not ready")
    p = Path(req.video_path)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"video_path not found: {req.video_path}")
    if not p.is_dir():
        # We only support image sequence directories in this project.
        raise HTTPException(status_code=400, detail=f"video_path must be a directory: {req.video_path}")
    return {"ok": True}
    
@app.post("/track", response_model=TrackResponse)
async def track_target(request: TrackRequest):
    """
    跟踪指定目标
    
    参数:
        request: 包含跟踪参数的请求体
    
    返回:
        包含结果文件路径和处理信息的响应
    """
    try:
        if STATE is None:
            raise HTTPException(status_code=503, detail="model not ready")

        # 记录请求处理开始时间
        start_time = time.time()

        start_frame_idx = _safe_int(request.start_frame_idx, name="start_frame_idx")
        obj_id = _safe_int(request.obj_id, name="obj_id")
        forward_frames = _safe_int(request.forward_frames, name="forward_frames")
        forward_frames = max(1, min(150, forward_frames))

        frame_paths = list(request.frame_paths or [])
        if not frame_paths:
            raise HTTPException(status_code=400, detail="frame_paths is empty")
        if len(frame_paths) != forward_frames:
            # Trust explicit frame_paths list; bound it to <=150.
            frame_paths = frame_paths[:forward_frames]
            forward_frames = len(frame_paths)
        if len(frame_paths) > 150:
            frame_paths = frame_paths[:150]
            forward_frames = len(frame_paths)

        bbox_xyxy = list(request.bbox_xyxy or [])
        if len(bbox_xyxy) != 4:
            raise HTTPException(status_code=400, detail="bbox_xyxy must have 4 numbers")

        print(f"接收到跟踪请求: start_frame_idx={start_frame_idx}, frames={forward_frames}, obj_id={obj_id}")
        print(f"  bbox_xyxy={bbox_xyxy}")
        print(f"  frame_paths count={len(frame_paths)}, first={frame_paths[0] if frame_paths else 'N/A'}")

        with STATE.lock:
            clip_dir: Path | None = None
            debug: dict[str, Any] = {
                "start_frame_idx": int(start_frame_idx),
                "forward_frames": int(forward_frames),
                "obj_id": int(obj_id),
                "bbox_xyxy_in": [float(x) for x in bbox_xyxy],
            }
            try:
                clip_dir = _make_clip_dir(frame_paths)
                clip_files = sorted(clip_dir.glob("*"))
                print(f"  clip_dir={clip_dir}, files_count={len(clip_files)}")

                # Run SAM2 on the small clip only.
                with STATE.autocast_ctx:
                    inference_state = STATE.predictor.init_state(
                        video_path=str(clip_dir),
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=False,
                        async_loading_frames=False,
                    )

                    video_W = inference_state.get("video_width") or 0
                    video_H = inference_state.get("video_height") or 0
                    num_frames = inference_state.get("num_frames") or 0
                    debug["video_width"] = int(video_W)
                    debug["video_height"] = int(video_H)
                    debug["num_frames"] = int(num_frames)
                    print(f"  video loaded: {video_W}x{video_H}, num_frames={num_frames}")

                    # SAM2 expects pixel coords by default (normalize_coords=True).
                    # If caller passes normalized coords, we must set normalize_coords=False.
                    normalize_coords = True
                    if _looks_normalized_xyxy(bbox_xyxy):
                        normalize_coords = False
                    debug["normalize_coords"] = bool(normalize_coords)
                    print(f"  normalize_coords={normalize_coords}")

                    # Add initial bbox prompt on local frame 0.
                    _, first_obj_ids, first_masks = STATE.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        box=np.array(bbox_xyxy, dtype=np.float32),
                        normalize_coords=bool(normalize_coords),
                    )
                    # 检查第一帧分割结果
                    if first_masks is not None and len(first_masks) > 0:
                        first_m = (first_masks[0] > 0.0).cpu().numpy()
                        first_area = int(np.count_nonzero(first_m))
                        first_bb = _bbox_from_mask(first_m)
                        print(f"  第一帧分割: mask_area={first_area}, bbox={first_bb}")
                        debug["first_frame_mask_area"] = first_area
                        debug["first_frame_bbox"] = first_bb
                    else:
                        print(f"  第一帧分割: 无 mask 输出!")
                        debug["first_frame_mask_area"] = 0
                        debug["first_frame_bbox"] = None

                    # Track forward over the clip. NOTE: predictor counts inclusive range.
                    max_to_track = max(0, forward_frames - 1)
                    results: dict[int, dict[int, dict[str, Any]]] = {}
                    debug_frames: dict[str, Any] = {}
                    print(f"  开始传播跟踪: max_to_track={max_to_track}")
                    frame_count_iter = 0
                    for local_idx, obj_ids, masks in STATE.predictor.propagate_in_video(
                        inference_state=inference_state,
                        start_frame_idx=0,
                        max_frame_num_to_track=max_to_track,
                        reverse=False,
                    ):
                        frame_count_iter += 1
                        global_idx = int(start_frame_idx) + int(local_idx)
                        results.setdefault(global_idx, {})
                        # masks: [num_obj, H, W]
                        for i, oid in enumerate(list(obj_ids)):
                            m = (masks[i] > 0.0).cpu().numpy()
                            try:
                                area = int(np.count_nonzero(m))
                            except Exception:
                                area = 0
                            bb = _bbox_from_mask(m)
                            if bb is None:
                                debug_frames[str(global_idx)] = {"obj_id": int(oid), "mask_area": area, "bbox": None}
                                continue
                            results[global_idx][int(oid)] = {"bbox": bb}
                            debug_frames[str(global_idx)] = {"obj_id": int(oid), "mask_area": area, "bbox": bb}
                    
                    print(f"  传播完成: 迭代了 {frame_count_iter} 帧, results 有效帧数={sum(1 for v in results.values() if v)}")
                    debug["frames"] = debug_frames
            finally:
                # Clean up clip dir to avoid disk bloat.
                if clip_dir is not None:
                    try:
                        shutil.rmtree(clip_dir, ignore_errors=True)
                    except Exception:
                        pass

        serializable_results = convert_keys_for_json(results)
        # Add debug section (GUI will ignore non-integer frame keys)
        serializable_results["_debug"] = convert_keys_for_json(debug)
        
        # 管理缓存文件
        manage_cache(TMP_DIR, max_files=200)
        
        # 生成唯一的文件名（使用时间戳和请求参数）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"tracking_result_{timestamp}_frame{start_frame_idx}_obj{obj_id}.json"
        file_path = os.path.join(TMP_DIR, file_name)
        
        # 保存结果到JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # 获取绝对路径
        abs_file_path = os.path.abspath(file_path)
        
        # 记录请求处理结束时间
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"跟踪请求处理完成，耗时: {processing_time:.2f}秒，结果已保存至: {abs_file_path}")
        
        # 返回结果文件路径
        return TrackResponse(
            result_file_path=abs_file_path,
            processing_time=processing_time,
            frame_count=len(results)
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    global STATE
    STATE = None
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass
    print("SAM2服务已关闭")

if __name__ == "__main__":
    # 从配置文件读取端口，默认 8848
    port = int(_CONFIG.get("sam_server_port", 8848))
    uvicorn.run(app, host="127.0.0.1", port=port)

#帧索引从0开始