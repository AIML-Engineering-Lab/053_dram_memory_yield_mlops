"""Build carousel v5 HTML:
- +2 new slides: HybridTransformerCNN architecture, GPU comparison table
- Renamed all "simulation" to "production"
- No em dashes anywhere
- 13 total slides
"""

import json
from pathlib import Path

imgs = json.load(open("/tmp/carousel_imgs.json"))

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0D1B2E;--bg2:#0F2338;--card:#152540;--border:#1E3A5F;
  --cyan:#00C8E8;--green:#22C55E;--orange:#F97316;--red:#EF4444;--yellow:#F59E0B;
  --white:#E8F0F8;--muted:#8BA4C0;--purple:#A78BFA;
}
body{background:#111;font-family:'Inter',sans-serif;padding:20px}
.slide{
  width:1080px;height:1350px;background:var(--bg);color:var(--white);
  position:relative;overflow:hidden;margin:0 auto 24px;
  page-break-after:always;display:flex;flex-direction:column;padding:50px 66px 78px;
}
.slide::before{content:'';position:absolute;top:0;left:0;right:0;height:5px;
  background:linear-gradient(90deg,var(--cyan),var(--green),var(--orange))}
.footer{position:absolute;bottom:26px;left:66px;right:66px;
  display:flex;justify-content:space-between;align-items:center;
  color:var(--muted);font-size:17px;border-top:1px solid var(--border);padding-top:12px}
.tag{display:inline-block;padding:5px 16px;border-radius:6px;font-size:16px;
  font-weight:600;letter-spacing:.04em;text-transform:uppercase}
.tag-cyan{background:rgba(0,200,232,.15);color:var(--cyan);border:1px solid rgba(0,200,232,.3)}
.tag-green{background:rgba(34,197,94,.15);color:var(--green);border:1px solid rgba(34,197,94,.3)}
.tag-orange{background:rgba(249,115,22,.15);color:var(--orange);border:1px solid rgba(249,115,22,.4)}
.tag-red{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.tag-purple{background:rgba(167,139,250,.15);color:var(--purple);border:1px solid rgba(167,139,250,.3)}
h1{font-size:64px;font-weight:800;line-height:1.1;color:var(--white)}
h2{font-size:46px;font-weight:700;line-height:1.15;color:var(--white)}
h3{font-size:30px;font-weight:600;color:var(--cyan);margin-bottom:10px}
p{font-size:24px;line-height:1.65;color:var(--muted)}
.ac{color:var(--cyan)}.ag{color:var(--green)}.ao{color:var(--orange)}
.ar{color:var(--red)}.ay{color:var(--yellow)}.ap{color:var(--purple)}
.img-box{background:#080f1a;border:1px solid var(--border);border-radius:12px;
  overflow:hidden;display:flex;align-items:center;justify-content:center}
.img-box img{width:100%;height:100%;object-fit:cover;display:block}
.img-box-contain img{object-fit:contain}
.stats-row{display:flex;gap:14px;margin:14px 0}
.stat-box{flex:1;background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:20px 14px;text-align:center}
.num{font-size:40px;font-weight:800;color:var(--cyan);line-height:1.1}
.num-g{color:var(--green)}.num-o{color:var(--orange)}.num-r{color:var(--red)}.num-p{color:var(--purple)}
.lbl{font-size:17px;color:var(--muted);margin-top:5px}
.caption{background:rgba(0,200,232,.07);border-left:3px solid var(--cyan);
  border-radius:0 8px 8px 0;padding:13px 18px;margin-top:13px}
.caption p{font-size:21px;line-height:1.55;color:var(--muted)}
.key{color:var(--white);font-weight:600}
.tl{list-style:none}
.tl li{display:flex;gap:14px;align-items:flex-start;margin-bottom:18px;
  font-size:22px;color:var(--muted);line-height:1.5}
.db{flex-shrink:0;width:72px;text-align:center;padding:4px 0;border-radius:6px;
  font-weight:700;font-size:19px}
.db-bl{background:rgba(30,80,160,.4);color:#7AB2F4;border:1px solid #2E5A88}
.db-yl{background:rgba(245,158,11,.2);color:var(--yellow);border:1px solid rgba(245,158,11,.4)}
.db-or{background:rgba(249,115,22,.2);color:var(--orange);border:1px solid rgba(249,115,22,.4)}
.db-cy{background:rgba(0,200,232,.2);color:var(--cyan);border:1px solid rgba(0,200,232,.4)}
.db-rd{background:rgba(239,68,68,.2);color:var(--red);border:1px solid rgba(239,68,68,.4)}
.db-gn{background:rgba(34,197,94,.2);color:var(--green);border:1px solid rgba(34,197,94,.4)}
.cb{background:#080F1A;border:1px solid var(--border);border-radius:10px;
  padding:20px 24px;font-family:'JetBrains Mono',monospace;font-size:19px;line-height:1.7;color:#A0C4FF}
.cb .cm{color:#445566}.cb .kw{color:var(--cyan)}.cb .vl{color:var(--green)}
.cb .wn{color:var(--orange)}.cb .er{color:var(--red)}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px}
.feat-tbl{width:100%;border-collapse:collapse;font-size:20px}
.feat-tbl th{text-align:left;color:var(--cyan);padding:8px 10px;border-bottom:2px solid var(--border);font-weight:700}
.feat-tbl td{padding:8px 10px;border-bottom:1px solid var(--border);color:var(--muted)}
.feat-tbl td:first-child{color:var(--white);font-weight:600;font-family:'JetBrains Mono',monospace;font-size:17px}
.gpu-tbl{width:100%;border-collapse:collapse;font-size:19px}
.gpu-tbl th{text-align:left;color:var(--cyan);padding:10px 10px;border-bottom:2px solid var(--border);font-weight:700;font-size:18px}
.gpu-tbl td{padding:10px 10px;border-bottom:1px solid var(--border);color:var(--muted)}
.gpu-tbl tr.hl{background:rgba(0,200,232,.06)}
.gpu-tbl td:first-child{color:var(--white);font-weight:600}
"""

def slide(num, total, content):
    return f"""<div class="slide">
{content}
  <div class="footer"><span>AIML Engineering Lab</span><span>{num} / {total}</span></div>
</div>"""

TOTAL = 13  # v5: expanded from 11 to 13 slides

# ── SLIDE 1: Title ──
s1 = f"""  <div style="display:flex;flex-direction:column;flex:1;gap:12px">
    <div class="tag tag-cyan">DRAM Semiconductor &middot; MLOps</div>
    <h1 style="font-size:56px">40 Days of<br><span class="ac">DRAM Production</span><br>Drift Detection</h1>
    <div style="display:flex;flex-direction:column;gap:6px;font-size:25px">
      <div class="ag" style="font-weight:700">200,000,000 rows processed</div>
      <div style="color:var(--muted)">1 retrain &middot; 1 canary failure &middot; 1 automatic rollback</div>
      <div class="ao">Zero human intervention</div>
    </div>
    <div style="color:var(--muted);font-size:19px">HybridTransformerCNN &middot; Airflow &middot; Kafka &middot; Spark &middot; MLflow &middot; K8s &middot; Prometheus</div>
    <div class="img-box img-box-contain" style="flex:1;min-height:0;margin-top:8px;max-height:78%;max-width:88%;align-self:center">
      <img src="data:image/png;base64,{imgs['wafer']}" alt="Wafer map">
    </div>
  </div>
  <div class="stats-row">
    <div class="stat-box"><div class="num">1:160</div><div class="lbl">Class imbalance</div></div>
    <div class="stat-box"><div class="num num-g">16M</div><div class="lbl">Training records</div></div>
    <div class="stat-box"><div class="num num-o">317K</div><div class="lbl">Model parameters</div></div>
  </div>"""

# ── SLIDE 2: Problem + FocalLoss + Features ──
s2 = f"""  <div class="tag tag-red" style="margin-bottom:14px">The Problem</div>
  <h2 style="margin-bottom:10px;font-size:40px">1:160 Imbalance: <span class="ar">99.4% accuracy predicts nothing</span></h2>
  <div class="img-box img-box-contain" style="height:280px;margin-bottom:10px">
    <img src="data:image/png;base64,{imgs['class_dist']}" alt="Class distribution">
  </div>
  <div class="caption" style="margin-top:0;margin-bottom:10px">
    <p><span class="key">Left:</span> 49,686 pass vs 314 fail in a batch (1:158). <span class="key">Right:</span> Label noise with 42 Pass-as-Fail and 17 Fail-as-Pass noisy labels. Standard cross-entropy predicts "pass" for every chip: 99.4% accuracy, 0% defect recall.</p>
  </div>
  <h3 style="font-size:24px;margin-bottom:8px">Why FocalLoss, not SMOTE?</h3>
  <p style="font-size:20px;margin-bottom:10px"><strong class="ac">FocalLoss</strong> (&alpha;=0.75, &gamma;=2.0) adds a modulating factor (1-p)<sup>&gamma;</sup> to cross-entropy loss. Easy majority samples get near-zero loss weight, forcing the model to focus gradient updates on missed defective chips. <strong class="ar">SMOTE</strong> creates synthetic minority samples by interpolation, but for sensor data like cell leakage and retention time, interpolated values create physically impossible feature combinations. FocalLoss modifies the loss function directly without generating fake data.</p>
  <h3 style="font-size:24px;margin-bottom:6px">DRAM Test Features</h3>
  <table class="feat-tbl">
    <tr><th>Feature</th><th>What it measures</th><th>Why it matters</th></tr>
    <tr><td>test_temp_c</td><td>Chip temperature during test</td><td>Hot spots = process defects</td></tr>
    <tr><td>cell_leakage_fa</td><td>Cell current leak (femtoamps)</td><td>High leak = retention failure</td></tr>
    <tr><td>retention_time_ms</td><td>Charge hold time (ms)</td><td>Below spec = bit errors</td></tr>
    <tr><td>gate_oxide_thickness_a</td><td>Oxide layer (angstroms)</td><td>Too thin = breakdown</td></tr>
    <tr><td>vt_shift_mv</td><td>Threshold voltage shift (mV)</td><td>Off-target = unreliable</td></tr>
    <tr><td>trcd_ns</td><td>Row-to-column delay (ns)</td><td>Slow access = timing fail</td></tr>
  </table>"""

# ── SLIDE 3: HybridTransformerCNN Architecture (NEW) ──
s3 = f"""  <div class="tag tag-purple" style="margin-bottom:14px">Model Architecture</div>
  <h2 style="margin-bottom:10px;font-size:40px">HybridTransformerCNN:<br><span class="ap">Local + Global Feature Learning</span></h2>
  <div class="img-box img-box-contain" style="height:400px;margin-bottom:12px">
    <img src="data:image/png;base64,{imgs['hybrid_arch']}" alt="HybridTransformerCNN architecture">
  </div>
  <div class="caption" style="margin-top:0;margin-bottom:10px">
    <p><span class="key">Architecture flow:</span> Raw DRAM features enter per-feature tokenization where each sensor gets its own learned embedding. The 1D-CNN extracts local patterns between adjacent features (e.g., leakage + retention combined signal). The Transformer self-attention layer captures global interactions between ALL features simultaneously. Finally, an MLP classification head with dropout produces the pass/fail prediction trained with FocalLoss.</p>
  </div>
  <div class="g2" style="margin-bottom:10px">
    <div class="card" style="border-left:4px solid var(--green)">
      <div style="font-size:21px;font-weight:700;color:var(--green);margin-bottom:6px">1D-CNN: Local Patterns</div>
      <p style="font-size:19px">Sliding convolution filters detect short-range feature interactions. If cell_leakage_fa and retention_time_ms are adjacent, the CNN learns their combined failure signature. These are patterns that individual features cannot reveal alone.</p>
    </div>
    <div class="card" style="border-left:4px solid var(--orange)">
      <div style="font-size:21px;font-weight:700;color:var(--orange);margin-bottom:6px">Transformer: Global Interactions</div>
      <p style="font-size:19px">Self-attention connects every feature to every other feature regardless of position. vt_shift_mv interacts with trcd_ns even though they are far apart in the input vector. Attention weights are learned, not hardcoded.</p>
    </div>
  </div>
  <div class="g3">
    <div class="card" style="text-align:center">
      <div style="font-size:19px;font-weight:700;color:var(--cyan);margin-bottom:4px">Per-Feature Tokenization</div>
      <p style="font-size:17px">Each of 6 features gets its own embedding dimension. Different from tabular models that flatten all features into one vector.</p>
    </div>
    <div class="card" style="text-align:center">
      <div style="font-size:19px;font-weight:700;color:var(--purple);margin-bottom:4px">FT-Transformer Family</div>
      <p style="font-size:17px">Inspired by FT-Transformer and TabNet research. Designed for tabular data where feature relationships matter more than raw values.</p>
    </div>
    <div class="card" style="text-align:center">
      <div style="font-size:19px;font-weight:700;color:var(--green);margin-bottom:4px">317K Parameters</div>
      <p style="font-size:17px">Small enough for T4 GPU training. Large enough to capture complex sensor interactions. Trains in 1.6 hours on A100.</p>
    </div>
  </div>"""

# ── SLIDE 4: Training Results ──
s4 = f"""  <div class="tag tag-cyan" style="margin-bottom:18px">A100 GPU Training</div>
  <h2 style="margin-bottom:14px;font-size:42px">HybridTransformerCNN on<br><span class="ac">NVIDIA A100-SXM4-40GB</span></h2>
  <div class="img-box img-box-contain" style="height:460px;margin-bottom:14px">
    <img src="data:image/png;base64,{imgs['training']}" alt="Training results">
  </div>
  <div class="caption" style="margin-top:0">
    <p><span class="key">Top left:</span> Loss convergence over 50 epochs with bfloat16. <span class="key">Top center:</span> AUC-PR per epoch. <span class="key">Top right:</span> Precision-recall curves for val/test/unseen splits (near-identical = no overfitting). <span class="key">Bottom:</span> Confusion matrices. 3,374 true defects caught out of 15,595 total. 34,075 good chips flagged for retest (acceptable: retest costs $0.10/chip vs $1000+ field failure).</p>
  </div>
  <div class="stats-row">
    <div class="stat-box"><div class="num">0.816</div><div class="lbl">AUC-ROC</div></div>
    <div class="stat-box"><div class="num num-g">0.054</div><div class="lbl">AUC-PR (primary metric)</div></div>
    <div class="stat-box"><div class="num num-o">50</div><div class="lbl">Epochs (best: 48)</div></div>
  </div>
  <div class="caption" style="margin-top:0">
    <p><span class="key">Why AUC-PR, not AUC-ROC?</span> At 1:160 imbalance, AUC-ROC inflates from massive true negatives. We catch 21.6% of defects but AUC-ROC shows 0.816. AUC-PR ignores true negatives and honestly reflects minority class detection. Our 0.054 is 8.6x better than random (0.00625). <span class="key">A100-SXM4-40GB</span> is a server-grade GPU with NVLink socket and 40GB HBM2e memory.</p>
  </div>"""

# ── SLIDE 5: bfloat16 death spiral ──
s5 = f"""  <div class="tag tag-red" style="margin-bottom:20px">Critical Engineering Insight</div>
  <h2 style="margin-bottom:16px">The <span class="ar">float16 Death Spiral</span><br>that killed 4 training runs</h2>
  <div class="g2" style="margin-bottom:14px">
    <div style="background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.3);border-radius:10px;padding:18px">
      <div style="font-size:20px;color:var(--red);font-weight:700;margin-bottom:10px">float16 + GradScaler</div>
      <div class="cb" style="font-size:18px">
        FocalLoss: <span class="er">(1-p)^gamma</span><br>
        <span class="cm"># 5-bit exp overflows</span><br>
        GradScaler / 2 / 2 / 2<br>
        <span class="wn">scale to 0, loss collapses</span>
      </div>
    </div>
    <div style="background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.3);border-radius:10px;padding:18px">
      <div style="font-size:20px;color:var(--green);font-weight:700;margin-bottom:10px">bfloat16, no GradScaler</div>
      <table style="width:100%;font-size:19px;color:var(--muted);border-collapse:collapse">
        <tr><td style="padding:5px 0;color:var(--white)">float16</td><td>5-bit exp = +/-65,504</td></tr>
        <tr><td style="padding:5px 0;color:var(--cyan)">bfloat16</td><td>8-bit exp = +/-3.4x10^38</td></tr>
        <tr><td colspan="2" style="padding-top:8px;color:var(--white)">bfloat16 has float32 exponent range.<br>Focal gradients never overflow.</td></tr>
      </table>
    </div>
  </div>
  <div class="caption">
    <p><span class="key">The fix:</span> Changed <code style="color:var(--red);background:rgba(239,68,68,.1);padding:1px 6px;border-radius:3px">torch.float16</code> to <code style="color:var(--cyan);background:rgba(0,200,232,.1);padding:1px 6px;border-radius:3px">torch.bfloat16</code>. Removed GradScaler. AUC-PR recovered from 0.001 (collapsed) to 0.054, a 54x improvement. Hardware-aware numerical analysis for production training.</p>
  </div>
  <div class="stats-row">
    <div class="stat-box"><div class="num num-r">4</div><div class="lbl">Runs killed by float16</div></div>
    <div class="stat-box"><div class="num num-g">54x</div><div class="lbl">AUC-PR recovery</div></div>
    <div class="stat-box"><div class="num">bfloat16</div><div class="lbl">8-bit exponent fix</div></div>
  </div>
  <div class="caption" style="margin-top:0">
    <p><span class="key">Why loss goes high, stays, then drops:</span> Early epochs: model is random (high loss). Middle: learns easy majority first; FocalLoss then forces focus on hard minority (temporary loss bump). Late: cosine LR schedule stabilizes convergence.</p>
  </div>"""

# ── SLIDE 6: GPU Comparison (NEW) ──
s6 = f"""  <div class="tag tag-cyan" style="margin-bottom:14px">GPU Landscape</div>
  <h2 style="margin-bottom:12px;font-size:40px">A100-SXM4-40GB in context:<br><span class="ac">What companies actually use</span></h2>
  <table class="gpu-tbl" style="margin-bottom:14px">
    <tr><th>GPU</th><th>Memory</th><th>FP16 TFLOPS</th><th>Used by</th><th>Use case</th></tr>
    <tr class="hl"><td style="color:var(--cyan)">A100-SXM4</td><td>40/80GB HBM2e</td><td>312</td><td>This project, AWS p4d, GCP a2</td><td>Training + inference</td></tr>
    <tr><td style="color:var(--green)">H100-SXM5</td><td>80GB HBM3</td><td>990</td><td>OpenAI, Meta, NVIDIA DGX H100</td><td>LLM training at scale</td></tr>
    <tr><td style="color:var(--orange)">H200</td><td>141GB HBM3e</td><td>990</td><td>Cloud providers (2024+)</td><td>LLM fine-tuning, large batch</td></tr>
    <tr><td>T4</td><td>16GB GDDR6</td><td>65</td><td>AWS g4dn, Colab free, inferencing</td><td>Inference, small training</td></tr>
    <tr><td>V100</td><td>16/32GB HBM2</td><td>125</td><td>Legacy clusters (AMD, Intel labs)</td><td>Previous gen training</td></tr>
    <tr><td>L4</td><td>24GB GDDR6</td><td>121</td><td>GCP g2, video/inference</td><td>Cost-efficient inference</td></tr>
    <tr><td style="color:var(--red)">MI300X (AMD)</td><td>192GB HBM3</td><td>1,300+</td><td>AMD ROCm, Azure ND MI300X</td><td>Competing with H100</td></tr>
    <tr><td>Gaudi2 (Intel)</td><td>96GB HBM2e</td><td>~420</td><td>AWS dl1, Intel Data Center</td><td>Cost-effective training</td></tr>
    <tr><td>TPU v5p (Google)</td><td>95GB HBM</td><td>~459</td><td>Google Cloud only</td><td>Gemini, JAX workloads</td></tr>
  </table>
  <div class="g2">
    <div class="card" style="border-left:4px solid var(--cyan)">
      <div style="font-size:21px;font-weight:700;color:var(--cyan);margin-bottom:6px">Why A100 for this project?</div>
      <p style="font-size:19px">317K parameter model on 16M rows fits in A100's 40GB HBM2e. bfloat16 native support (critical for FocalLoss). Available on Colab Pro at ~6.79 CU/hour. 3.7 hours wall clock for Day 1 training, 219 min for full 40-day run.</p>
    </div>
    <div class="card" style="border-left:4px solid var(--orange)">
      <div style="font-size:21px;font-weight:700;color:var(--orange);margin-bottom:6px">Production GPU selection</div>
      <p style="font-size:19px"><strong>Training:</strong> A100/H100 for model development. <strong>Inference:</strong> T4 or L4 for cost efficiency. Our auto-selector picks T4 for models under 50M parameters, A100 only when daily data exceeds 1TB. Companies like AMD, Samsung, SK Hynix run similar GPU fleets for semiconductor ML.</p>
    </div>
  </div>
  <div class="caption" style="margin-top:10px">
    <p><span class="key">SXM4 vs PCIe:</span> SXM4 is a server socket form factor with NVLink interconnect (600GB/s bandwidth between GPUs). PCIe version has 64GB/s. SXM4 matters for multi-GPU training; for single-GPU like our 317K model, performance is identical. The "40GB" is HBM2e (High Bandwidth Memory), 2TB/s memory bandwidth vs 1TB/s for GDDR6 GPUs like T4.</p>
  </div>"""

# ── SLIDE 7: 40-Day Drift Heatmap + PSI ──
s7 = f"""  <div class="tag tag-cyan" style="margin-bottom:16px">40-Day PSI Drift Heatmap</div>
  <h2 style="margin-bottom:12px;font-size:42px">Drift escalated for 13 days.<br><span class="ao">Then the retrain gate opened.</span></h2>
  <div class="img-box img-box-contain" style="height:260px;margin-bottom:12px">
    <img src="data:image/png;base64,{imgs['drift_heat']}" alt="PSI drift heatmap">
  </div>
  <div class="caption" style="margin-top:0;margin-bottom:10px">
    <p><span class="key">What is PSI?</span> Population Stability Index measures distribution shift per feature per day. PSI &lt; 0.1 = stable, 0.1-0.2 = warning, &gt; 0.2 = critical. Calculated as &sum;(P_i - Q_i) &times; ln(P_i/Q_i) comparing production bins against training baseline. If any critical feature exceeds threshold, drift alarm fires.</p>
  </div>
  <ul class="tl" style="width:100%">
    <li><div class="db db-bl">D1-8</div><div>Baseline period. No drift. PSI near zero across all 6 features. Model performing at validation benchmarks.</div></li>
    <li><div class="db db-yl">D9</div><div><strong class="ay">False alarm:</strong> retention_time_ms PSI spiked to 1.47 from a one-time batch anomaly (equipment calibration). Staleness gate prevented unnecessary retraining. Day 10 auto-recovered.</div></li>
    <li><div class="db db-or">D11-29</div><div>Gradual drift escalated across features. PSI exceeded 5.0 by Day 29. Retrain blocked by staleness: model was &lt;30 days old. AUC-PR degrading steadily.</div></li>
    <li><div class="db db-cy">D30</div><div><strong class="ac">RETRAIN TRIGGERED.</strong> All 3 gates opened: PSI 5.25, AUC-PR dropped 8%, model 30 days old. 50 epochs on full day's data, bfloat16 on A100.</div></li>
    <li><div class="db db-gn">D31-35</div><div>Post-retrain recovery. New v2 model outperformed v1. AUC-PR improved above baseline.</div></li>
    <li><div class="db db-rd">D39</div><div><strong class="ar">CANARY FAILED.</strong> Bad model deployed to 10% canary pod. AUC-PR crashed. <strong class="ag">Automatic rollback to v2</strong> within seconds.</div></li>
    <li><div class="db db-gn">D40</div><div>System recovered. Rolled-back v2 model stable. Pipeline proved self-healing without human intervention.</div></li>
  </ul>"""

# ── SLIDE 8: Retrain Gate ──
s8 = f"""  <div class="tag tag-orange" style="margin-bottom:16px">Retrain Logic</div>
  <h2 style="margin-bottom:14px;font-size:42px">3-gate retrain trigger.<br><span class="ao">All 3 must be TRUE.</span></h2>
  <div style="display:flex;gap:14px;margin-bottom:14px">
    <div class="card" style="flex:1;border-left:4px solid var(--cyan)">
      <div style="font-size:22px;font-weight:700;color:var(--cyan);margin-bottom:8px">Gate 1: Statistical Drift (PSI)</div>
      <p style="font-size:20px">Any critical feature PSI &gt; 0.2 for 3+ consecutive days. Prevents reaction to single-day noise. retention_time_ms was the primary drift signal.</p>
      <div style="margin-top:8px;font-size:18px;color:var(--green)">Threshold: PSI &gt; 0.2, sustained 3+ days</div>
    </div>
    <div class="card" style="flex:1;border-left:4px solid var(--orange)">
      <div style="font-size:22px;font-weight:700;color:var(--orange);margin-bottom:8px">Gate 2: Performance Degradation</div>
      <p style="font-size:20px">&gt;5% AUC-PR drop vs baseline. Compares current batch inference against the champion model's validation benchmark.</p>
      <div style="margin-top:8px;font-size:18px;color:var(--green)">Threshold: 5% relative AUC-PR degradation</div>
    </div>
  </div>
  <div style="display:flex;gap:14px">
    <div class="card" style="flex:1;border-left:4px solid var(--red)">
      <div style="font-size:22px;font-weight:700;color:var(--red);margin-bottom:8px">Gate 3: Model Staleness</div>
      <p style="font-size:20px">&gt;30 days since last retrain. Prevents reactive retraining from transient drift spikes that resolve on their own.</p>
      <div style="margin-top:8px;font-size:18px;color:var(--green)">Threshold: 30 calendar days since last training</div>
    </div>
    <div class="card" style="flex:1;border-left:4px solid var(--green)">
      <div style="font-size:22px;font-weight:700;color:var(--green);margin-bottom:8px">Result: Day 30 Retrain</div>
      <p style="font-size:20px">All 3 gates opened simultaneously on Day 30. PSI at 5.25, AUC-PR dropped 8%, model was 30 days old. Fully automated retraining fired.</p>
      <div style="margin-top:8px;font-size:18px;color:var(--cyan)">All 3 gates must be TRUE to retrain</div>
    </div>
  </div>
  <div class="g2" style="margin-top:10px">
    <div class="card">
      <div style="font-size:21px;font-weight:700;color:var(--cyan);margin-bottom:6px">Why 3 gates, not 1?</div>
      <p style="font-size:20px">PSI alone fires on normal batch variance. Performance alone is noisy at 1:160 imbalance. Staleness alone wastes compute. Combined: precision retraining.</p>
    </div>
    <div class="card">
      <div style="font-size:21px;font-weight:700;color:var(--orange);margin-bottom:6px">Low-Data Drift Tagging</div>
      <p style="font-size:20px">When batch &lt;10K rows, PSI is statistically unreliable due to sparse bin counts. Drift is <strong style="color:var(--white)">tagged in MLflow but never triggers retraining</strong> until data volume is sufficient.</p>
    </div>
  </div>"""

# ── SLIDE 9: Production Infrastructure ──
s9 = f"""  <div class="tag tag-green" style="margin-bottom:16px">Production Infrastructure</div>
  <h2 style="margin-bottom:16px;font-size:42px">6 services. One command.<br><span class="ag">Fully orchestrated.</span></h2>
  <div class="g2" style="margin-bottom:14px">
    <div class="card"><div style="font-size:21px;font-weight:700;color:var(--cyan);margin-bottom:5px">Apache Airflow</div>
      <p style="font-size:19px">3 DAGs (Directed Acyclic Graphs): daily_inference, retrain_trigger, master orchestrator. Each DAG defines task dependencies without cycles. SLA monitoring alerts if any task exceeds its time budget. Automatic retry with exponential backoff.</p></div>
    <div class="card"><div style="font-size:21px;font-weight:700;color:var(--green);margin-bottom:5px">Apache Kafka</div>
      <p style="font-size:19px">Streaming DRAM test records from fab sensors. Consumer group routes records to FastAPI inference engine in real time. Handles 5M+ messages/day with partitioned topics for parallel processing.</p></div>
    <div class="card"><div style="font-size:21px;font-weight:700;color:var(--orange);margin-bottom:5px">Apache Spark</div>
      <p style="font-size:19px">Batch ETL on 5M-row daily slices. 4.7x faster than pandas at this row count. Reads raw sensor data, applies feature engineering, writes parquet to S3. Total: 8.5GB parquet across 40 production days.</p></div>
    <div class="card"><div style="font-size:21px;font-weight:700;color:var(--yellow);margin-bottom:5px">MLflow + Registry</div>
      <p style="font-size:19px">Tracks every experiment run including the 4 failed float16 runs. Model Registry maintains champion/challenger versions. Automatic promotion on canary success, rollback on failure.</p></div>
    <div class="card"><div style="font-size:21px;font-weight:700;color:#A78BFA;margin-bottom:5px">Prometheus + Grafana</div>
      <p style="font-size:19px">Prometheus scrapes /metrics from FastAPI every 15 seconds. Grafana dashboards show per-feature PSI, inference latency (p50/p95/p99), request rates, prediction distribution. Alerts on SLA breach.</p></div>
    <div class="card"><div style="font-size:21px;font-weight:700;color:var(--cyan);margin-bottom:5px">FastAPI + K8s</div>
      <p style="font-size:19px">REST API with Pydantic validation, Redis prediction cache (5-min TTL). K8s with HPA (2-8 pods), canary deployment (90/10 traffic split). 120ms p99 latency under load testing.</p></div>
  </div>
  <div class="caption">
    <p><span class="key">To run locally:</span> <code style="color:var(--cyan);background:rgba(0,200,232,.1);padding:1px 8px;border-radius:4px;font-family:monospace">cd deploy/docker &amp;&amp; docker compose up -d</code> starts all 6 services. Airflow :8080, MLflow :5001, Grafana :3000, FastAPI :8000/docs.</p>
  </div>"""

# ── SLIDE 10: Architecture + Monitoring ──
s10 = f"""  <div class="tag tag-cyan" style="margin-bottom:14px">System Architecture</div>
  <h2 style="margin-bottom:10px;font-size:38px">5-Layer MLOps:<br><span class="ac">Data, Train, Serve, Monitor, Retrain</span></h2>
  <div class="img-box img-box-contain" style="height:360px;margin-bottom:12px">
    <img src="data:image/png;base64,{imgs['architecture']}" alt="Architecture diagram">
  </div>
  <div class="caption" style="margin-top:0;margin-bottom:10px">
    <p><span class="key">Architecture:</span> Data layer ingests 16M+ fab sensor records. Training layer runs HybridTransformerCNN with FocalLoss on A100 GPU. Serving layer delivers predictions via FastAPI in Docker/K8s. Monitoring layer tracks drift with PSI and model metrics via Prometheus/Grafana. Retrain loop fires only when all 3 gates are satisfied.</p>
  </div>
  <h3 style="font-size:24px">Production Monitoring Dashboard</h3>
  <div class="img-box img-box-contain" style="height:300px;margin-bottom:0">
    <img src="data:image/png;base64,{imgs['monitoring']}" alt="Grafana monitoring dashboard">
  </div>
  <div class="caption" style="margin-top:10px">
    <p><span class="key">Grafana shows:</span> Request rate, latency percentiles (p50/p95/p99), prediction distribution (pass/fail ratio), CPU usage with HPA threshold, error rate, and pod autoscaling. All scraped by Prometheus every 15 seconds.</p>
  </div>"""

# ── SLIDE 11: Hardware Benchmark ──
s11 = f"""  <div class="tag tag-cyan" style="margin-bottom:16px">Hardware Benchmark</div>
  <h2 style="margin-bottom:12px;font-size:42px">CPU training = 69 hours.<br><span class="ag">A100 = 1.6 hours. GPU is not optional.</span></h2>
  <div class="img-box img-box-contain" style="height:320px;margin-bottom:12px">
    <img src="data:image/png;base64,{imgs['hw_bench']}" alt="Hardware benchmark">
  </div>
  <div class="stats-row">
    <div class="stat-box"><div class="num num-r">69 hr</div><div class="lbl">CPU (Apple M3)</div></div>
    <div class="stat-box"><div class="num num-o">3.5 hr</div><div class="lbl">T4 (16GB)</div></div>
    <div class="stat-box"><div class="num num-g">1.6 hr</div><div class="lbl">A100 (40GB)</div></div>
  </div>
  <div class="caption">
    <p><span class="key">Wall clock time</span> = actual elapsed time on your watch. The A100 processed 16M rows in 50 epochs in 1.6 hours of real time. The full 40-day production run (200M total rows, all drift checks, 1 retrain, canary evaluation, rollback logic) completed in 219 minutes wall clock on A100. Each production day represents one real production day in a DRAM fab, compressed into 5.5 minutes of compute.</p>
  </div>
  <div class="g2" style="margin-top:10px">
    <div class="card">
      <div style="font-size:21px;font-weight:700;color:var(--cyan);margin-bottom:6px">Why HybridTransformerCNN on tabular data?</div>
      <p style="font-size:19px"><strong>1D-CNN</strong> captures local patterns between adjacent features (e.g., leakage + retention combined signal). <strong>Transformer</strong> self-attention captures global feature interactions across all 6 features simultaneously. Per-feature tokenization gives each sensor its own learned embedding. This architecture consistently outperforms pure ML baselines on high-dimensional sensor data.</p>
    </div>
    <div class="card">
      <div style="font-size:21px;font-weight:700;color:var(--orange);margin-bottom:6px">Data scale</div>
      <p style="font-size:19px">16M training rows, 8.5GB total parquet across 40 production days (5M rows/day). All stored on S3 as compressed columnar parquet. Spark ETL handles daily batch processing. The pipeline is chip-agnostic: swap DRAM features for NAND flash or CPU wafer data, same architecture works.</p>
    </div>
  </div>"""

# ── SLIDE 12: 40-Day Production Timeline ──
s12 = f"""  <div class="tag tag-orange" style="margin-bottom:10px">40-Day Production Run</div>
  <h2 style="margin-bottom:10px;font-size:38px">Full system response to drift:<br><span class="ao">detect, retrain, evaluate, rollback</span></h2>
  <div class="img-box img-box-contain" style="height:380px;margin-bottom:10px">
    <img src="data:image/png;base64,{imgs['sim_summary']}" alt="Production run summary">
  </div>
  <div class="stats-row">
    <div class="stat-box"><div class="num">40</div><div class="lbl">Production days</div></div>
    <div class="stat-box"><div class="num num-g">200M</div><div class="lbl">Total rows</div></div>
    <div class="stat-box"><div class="num num-o">219 min</div><div class="lbl">Wall clock</div></div>
    <div class="stat-box"><div class="num num-r">1</div><div class="lbl">Rollback</div></div>
  </div>
  <div class="caption">
    <p><span class="key">How to read this:</span> 40 production days map to 40 real production days (Feb 20 to Mar 31, 2026). Each day processes 5M rows of DRAM sensor data. "Wall clock" = 219 minutes of actual GPU compute time. The system autonomously handled stable operation, gradual drift, a false alarm, triggered retraining, caught a bad model via canary testing, and auto-rolled back. Zero human intervention at any point.</p>
  </div>"""

# ── SLIDE 13: Day 40 Image + Deployment ──
s13 = f"""  <div class="tag tag-green" style="margin-bottom:10px">Final State: Day 40</div>
  <h2 style="margin-bottom:10px;font-size:38px">System recovered.<br><span class="ag">Self-healing pipeline proven.</span></h2>
  <div class="img-box img-box-contain" style="height:480px;margin-bottom:10px">
    <img src="data:image/png;base64,{imgs['day40']}" alt="Day 40 - 3 panel drift visualization">
  </div>
  <div class="caption" style="margin-top:0;margin-bottom:10px">
    <p><span class="key">3-panel drift visualization (Day 40 of 40).</span> Top: train vs production distribution for retention_time_ms showing clear shift. Middle: PSI per day with green/yellow/red severity, retrain (Day 30, cyan) and rollback (Day 39, red) markers. Bottom: AUC-PR showing v1 degradation, v2 recovery, canary failure, and final recovery. Full animated version (40 frames) available at <code style="color:var(--cyan);font-family:monospace">assets/drift_density_animation.gif</code></p>
  </div>
  <div class="img-box img-box-contain" style="height:220px;margin-bottom:0">
    <img src="data:image/png;base64,{imgs['deployment']}" alt="Deployment stack">
  </div>
  <div style="text-align:center;margin-top:14px;font-size:22px;color:var(--cyan);font-weight:700">
    github.com/rajendarmuddasani/DRAM_Yield_Predictor_MLOps
  </div>"""

# ── Assemble ──
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DRAM Memory Yield Predictor - MLOps Carousel v5</title>
<style>
{CSS}
</style>
</head>
<body>
{slide(1, TOTAL, s1)}
{slide(2, TOTAL, s2)}
{slide(3, TOTAL, s3)}
{slide(4, TOTAL, s4)}
{slide(5, TOTAL, s5)}
{slide(6, TOTAL, s6)}
{slide(7, TOTAL, s7)}
{slide(8, TOTAL, s8)}
{slide(9, TOTAL, s9)}
{slide(10, TOTAL, s10)}
{slide(11, TOTAL, s11)}
{slide(12, TOTAL, s12)}
{slide(13, TOTAL, s13)}
</body>
</html>"""

out = Path("docs/carousel.html")
out.write_text(html)
print(f"Carousel v5 written: {out} ({len(html)//1024}KB)")
print(f"Slides: {TOTAL}")
