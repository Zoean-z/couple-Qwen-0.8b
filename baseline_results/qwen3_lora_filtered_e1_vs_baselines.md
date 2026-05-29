# Qwen3-0.6B-Base LoRA E1 vs Baselines

时间：2026-05-20  
LoRA：filtered 10000 subset, 1 epoch  
对照：Qwen zero-shot Prompt C / attn_base_best

1. 上联：风弦未拨心先乱
   参考：夜幕已沉梦更闲
   Zero-shot Prompt C：雨露无声草自荣
   LoRA E1：月笛已吹梦已空
   Attention baseline：竹影犹留客难眠

2. 上联：花梦粘于春袖口
   参考：莺声溅落柳枝头
   Zero-shot Prompt C：月影藏于夜幕中
   LoRA E1：莺歌绕在柳枝头
   Attention baseline：秋风入画图中天

3. 上联：晋世文章昌二陆
   参考：魏家词赋重三曹
   Zero-shot Prompt C：唐时诗赋盛三李
   LoRA E1：唐风诗韵耀千秋
   Attention baseline：春风秋月一枝香

4. 上联：一句相思吟岁月
   参考：千杯美酒醉风情
   Zero-shot Prompt C：一句相思吟岁月
   LoRA E1：三更月色醉流年
   Attention baseline：千年故里人生天

5. 上联：几树梅花数竿竹
   参考：一潭秋水半屏山
   Zero-shot Prompt C：数竿竹几树梅花
   LoRA E1：一池碧水一池荷
   Attention baseline：一枝红杏花香浓

6. 上联：未舍东江开口咏
   参考：且施妙手点睛来
   Zero-shot Prompt C：东江开口咏未舍
   LoRA E1：已开西岭入云歌
   Attention baseline：但凭北斗是春秋

7. 上联：珍藏惟有诗三卷
   参考：馈赠则无画半帧
   Zero-shot Prompt C：珍藏惟有书四部
   LoRA E1：挥洒无非酒一壶
   Attention baseline：事业无边天井红

8. 上联：岁尽方知时日快
   参考：钱空始觉世情薄
   Zero-shot Prompt C：春来方觉岁月长
   LoRA E1：春来更觉岁月新
   Attention baseline：人来自古今多情

9. 上联：无花无酒无花酒
   参考：有洞有天有洞天
   Zero-shot Prompt C：山高水长海阔天宽
   LoRA E1：有花有酒有花酒
   Attention baseline：有限情深红尘埃

10. 上联：马齿草焉无马齿
    参考：猫头鹰好像猫头
    Zero-shot Prompt C：柳絮飞时无柳絮
    LoRA E1：竹竿竹竿有竹竿
    Attention baseline：羊毫春色新春花

## 简短结论

- 这一轮 `LoRA E1` 明显压住了部分 `zero-shot` 的直接复读问题。
- 在不少样例上，`LoRA E1` 比 `attn_base_best` 更像任务内输出，也更少出现完全无关的泛模板句。
- 但当前 `LoRA E1` 还不稳定，仍然会出现机械重复、语义发散和对子不工的问题。
- 这说明路线成立，但 1 个 epoch 还不够，后续需要继续训练，或换更干净的数据子集。
