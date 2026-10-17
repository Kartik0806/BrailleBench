import json, sys, numpy as np
def load(p): return [json.loads(l) for l in open(p)]
def q(a, ps=(5,25,50,75,95)):
    a=np.asarray([x for x in a if x is not None and np.isfinite(x)],float)
    return "n=%4d  "%len(a)+"  ".join(f"p{p}={np.percentile(a,p):7.2f}" for p in ps) if len(a) else "n=0"
if __name__=="__main__":
    for pool in sys.argv[1:]:
        R=[r for r in load(f"out/real_{pool}.jsonl")]
        ok=[r for r in R if "error" not in r]
        rel=[r for r in ok if r["n"]>=8 and r["q"]>=0.5]
        print(f"\n===== {pool}: {len(R)} images, {len(ok)} measured, {len(rel)} reliable (n>=8,q>=0.5)")
        print("native W        ",q([r["W"] for r in ok]))
        print("native H        ",q([r["H"] for r in ok]))
        print("dot pitch dp px ",q([r["dp"] for r in rel]))
        print("cell pitch P px ",q([r["P"] for r in rel if "P" in r]))
        print("R=P/dp          ",q([r["R"] for r in rel if "R" in r]))
        print("diam/dp         ",q([r["diam_over_dp"] for r in rel]))
        print("nlines          ",q([r["nlines"] for r in rel]), " | share 1 line:",round(np.mean([r['nlines']==1 for r in rel]),3))
        print("ink_w (cells)   ",q([r["ink_w"] for r in rel]))
        for m in ("margin_l","margin_r","margin_t","margin_b"): print(f"{m:15s} ",q([r[m] for r in rel]))
        g=[r["gap_b_mean"] for r in rel if "gap_b_mean" in r]
        print("word-gap b (per-image mean of n_words-1 gaps):",q(g))
        gm=[r["gap_b_min"] for r in rel if "gap_b_min" in r]
        print("word-gap b (per-image min):                    ",q(gm))
        print("qwen upscale s  ",q([r["qwen_scale"] for r in ok]))
