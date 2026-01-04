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
- `tracking_json`：跟踪框 JSON 文件路径
- `frame_dir`：比赛帧目录路径

运行:
```powershell
python app.py
```


