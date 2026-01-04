# jh4player 标注交互工具（PyQt）

## 目标
- 播放 `frame/` 内按 `image_id.jpg` 命名的比赛帧
- 叠加显示跟踪框（来自 `1128_*.json`）
- 支持在未锁定 target 的情况下补标漏检：手动输入 `track_id` 后单帧添加 bbox（SAM 为占位）
- 当目标 `target_id` 在当前帧缺失时自动暂停并引导三类交互：
  1) 目标不在画面内：播放跳到任意帧继续
  2) 目标未检测到：选择“手动标注当前帧 bbox”或 “SAM（占位）”
  3) 目标被检测到但 ID 变了：点击球员并将其 track_id 从当前帧起映射为 target_id

## 播放倍速
- 0.5x ~ 2.0x

## 安装
在本目录打开终端：

```powershell
python -m pip install -r requirements.txt
```

拉sam2模型到本目录：
参照之前方法，文件夹名称为 `sam2-main`


## 运行
在'jh4player_settings.json'中配置好路径后，

运行:
```powershell
python app.py
```

## 配置文件说明

所有配置项统一在 `jh4player_settings.json` 中定义：

```json
{
  "tracking_json": "D:\\path\\to\\your\\tracking.json",  // 跟踪框 JSON 文件路径（必须配置）
  "frame_dir": "D:\\path\\to\\your\\frames",             // 比赛帧目录路径（必须配置）
  "annotations_file": "annotations.json",               // 标注文件名（相对于项目目录）
  "tracking_json_pattern": "*.json",                    // 自动搜索 tracking_json 的文件模式
  "segment_size": 900,                                  // 分段大小（帧数）
  "base_fps": 30,                                       // 基础播放帧率
  "sam_base_url": "http://127.0.0.1:8848",              // SAM2 服务地址
  "sam_server_port": 8848,                              // SAM2 服务端口
  "sam2_model_cfg": "",                                 // SAM2 模型配置路径（留空使用默认）
  "sam2_checkpoint": "",                                // SAM2 模型权重路径（留空使用默认）
  "sam2_device": "",                                    // SAM2 设备（留空自动选择 cuda/cpu）
  "sam2_tmp_dir": "./tmp/sam2_tracking_results",        // SAM2 临时结果目录
  "sam2_max_cache_files": 200                           // SAM2 缓存文件最大数量
}
```

### 必须配置的项
- `tracking_json`：跟踪框 JSON 文件的绝对路径
- `frame_dir`：比赛帧图片目录的绝对路径

### 可选配置项
其他配置项都有默认值，可根据需要修改。


