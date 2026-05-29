# Qwen3-0.6B-Base LoRA E4 vs Baselines

时间：2026-05-20  
LoRA E4：使用过滤后的 10000 条样本训练，最佳 checkpoint 出现在 `epoch 2`  
对照对象：

- `Qwen3-0.6B-Base` zero-shot Prompt C
- `Qwen3-0.6B-Base` LoRA E1
- `attn_base_best`

## 固定 10 条 7 字样例对照

1. 上联：风弦未拨心先乱  
   参考：夜幕已沉梦更闲  
   Zero-shot Prompt C：雨露无声草自荣  
   LoRA E1：月笛已吹梦已空  
   LoRA E4(best@epoch2)：月笛犹弹梦已空  
   Attention baseline：竹影犹留客难眠

2. 上联：花梦粘于春袖口  
   参考：莺声溅落柳枝头  
   Zero-shot Prompt C：月影藏于夜幕中  
   LoRA E1：莺歌绕在柳枝头  
   LoRA E4(best@epoch2)：诗魂凝在柳眉梢  
   Attention baseline：秋风入画图中天

3. 上联：晋世文章昌二陆  
   参考：魏家词赋重三曹  
   Zero-shot Prompt C：唐时诗赋盛三李  
   LoRA E1：唐风诗韵耀千秋  
   LoRA E4(best@epoch2)：唐风诗赋振三曹  
   Attention baseline：春风秋月一枝香

4. 上联：一句相思吟岁月  
   参考：千杯美酒醉风情  
   Zero-shot Prompt C：一句相思吟岁月  
   LoRA E1：三更月色醉流年  
   LoRA E4(best@epoch2)：三生有幸遇知音  
   Attention baseline：千年故里人生天

5. 上联：几树梅花数竿竹  
   参考：一潭秋水半屏山  
   Zero-shot Prompt C：数竿竹几树梅花  
   LoRA E1：一池碧水一池荷  
   LoRA E4(best@epoch2)：一帘秋月半边天  
   Attention baseline：一枝红杏花香浓

6. 上联：未舍东江开口咏  
   参考：且施妙手点睛来  
   Zero-shot Prompt C：东江开口咏未舍  
   LoRA E1：已开西岭入云歌  
   LoRA E4(best@epoch2)：先开西岭出云峰  
   Attention baseline：但凭北斗是春秋

7. 上联：珍藏惟有诗三卷  
   参考：馈赠则无画半帧  
   Zero-shot Prompt C：珍藏惟有书四部  
   LoRA E1：挥洒无非酒一壶  
   LoRA E4(best@epoch2)：挥洒无非墨一笺  
   Attention baseline：事业无边天井红

8. 上联：岁尽方知时日快  
   参考：钱空始觉世情薄  
   Zero-shot Prompt C：春来方觉岁月长  
   LoRA E1：春来更觉岁月新  
   LoRA E4(best@epoch2)：春来又见柳丝长  
   Attention baseline：人来自古今多情

9. 上联：无花无酒无花酒  
   参考：有洞有天有洞天  
   Zero-shot Prompt C：山高水长海阔天宽  
   LoRA E1：有花有酒有花酒  
   LoRA E4(best@epoch2)：有酒有花有花酒  
   Attention baseline：有限情深红尘埃

10. 上联：马齿草焉无马齿  
    参考：猫头鹰好像猫头  
    Zero-shot Prompt C：柳絮飞时无柳絮  
    LoRA E1：竹竿竹竿有竹竿  
    LoRA E4(best@epoch2)：牛头山下有牛头  
    Attention baseline：羊毫春色新春花

## 简短结论

- `LoRA E4(best@epoch2)` 相比 `LoRA E1` 更稳定一些，复读更少，整体更像任务内输出。
- 相比 `zero-shot Prompt C`，`LoRA E4` 对“只输出下联、少复读上联”这件事明显更稳定。
- 相比当前的 `attention baseline`，`LoRA E4` 更少出现完全无关的泛模板句。
- 但当前结果仍然存在语义发散、对仗不工、局部模板化的问题，因此这轮实验更适合作为“路线有效”的展示，而不是“最终效果完成”的展示。
