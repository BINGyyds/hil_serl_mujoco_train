# backup

这个目录用于保存**值得长期保留的训练结果快照**，例如：

- 训练效果较好的 checkpoint
- 对应的 metrics 曲线
- 训练配置记录
- 后续用于汇报或展示的评估视频

## 当前建议结构

```text
train/backup/
└── YYYYMMDD[_tag]/
    ├── README.md
    ├── checkpoint_xxxxxxx/
    ├── metrics/
    └── eval_videos/        # 可选
```

## 说明

- `README.md`
  - 记录这次备份的训练配置、环境参数、关键结果和评估方式
- `checkpoint_xxxxxxx/`
  - 保存一个明确要保留的 checkpoint
- `metrics/`
  - 保存 `episode_metrics.csv` 与导出的 reward / success / error 曲线
- `eval_videos/`
  - 可选，保存用于展示的 MP4

## 对当前结构的建议

你现在的结构整体是合理的，已经具备：

- 训练结果
- 曲线
- checkpoint

这三类最重要的内容。

为了后续推到 GitHub 更清晰，建议再保持两个习惯：

1. 每个备份目录都放一份 `README.md`
   - 说明这次训练到底用了什么配置
   - 否则时间一长，很容易忘记这份 checkpoint 是怎么来的

2. 如果同一天会备份多次，目录名建议加短标签
   - 例如：`20260430_absdepth_success2000`
   - 比只用日期更容易区分

## GitHub 说明

当前 `train/backup/20260430` 大约是：

- checkpoint：约 `3.7M`
- metrics：约 `15M`
- 总计：约 `18M`

这个体量推到 GitHub **没有问题**。  
如果后面开始备份更多 checkpoint、视频或更大的 CSV，建议：

- 只保留代表性 checkpoint
- 或把大视频放到 release / 外部存储
- 或使用 Git LFS
