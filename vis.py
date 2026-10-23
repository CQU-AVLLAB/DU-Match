<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body { font-family: "Segoe UI", Arial, sans-serif; background: #f8f9fa; margin: 0; padding: 20px; }
  .container { max-width: 1100px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); padding: 30px; }
  h2 { text-align: center; color: #1a1a2e; margin-bottom: 30px; font-size: 22px; }
  .flow { display: flex; flex-direction: column; align-items: center; gap: 8px; }
  .box { 
    border: 2px solid #333; border-radius: 10px; padding: 14px 22px; text-align: center; 
    font-weight: 600; font-size: 15px; min-width: 220px; position: relative;
  }
  .input { background: #e3f2fd; border-color: #1976d2; }
  .backbone { background: #fff3e0; border-color: #f57c00; }
  .module { background: #e8f5e9; border-color: #388e3c; }
  .uncertainty { background: #fce4ec; border-color: #c2185b; }
  .seg { background: #ede7f6; border-color: #7b1fa2; }
  .arrow { font-size: 22px; color: #555; }
  .side-flow { display: flex; gap: 40px; margin-top: 20px; justify-content: center; flex-wrap: wrap; }
  .side-box { 
    border: 2px dashed #c2185b; border-radius: 10px; padding: 12px 18px; 
    background: #fff5f7; text-align: center; font-size: 14px; max-width: 260px;
  }
  .control-arrow { 
    font-size: 13px; color: #c2185b; font-weight: 600; 
    background: #fce4ec; padding: 4px 10px; border-radius: 20px; margin: 6px 0;
  }
  .note { 
    margin-top: 25px; padding: 15px; background: #f5f5f5; border-left: 4px solid #1976d2; 
    font-size: 14px; color: #333; line-height: 1.6;
  }
</style>
</head>
<body>
<div class="container">
  <h2>Proposed Framework (based on SegFormer-B2)</h2>
  
  <div class="flow">
    <div class="box input">Input Image<br><small>(Adverse Weather)</small></div>
    <div class="arrow">↓</div>
    <div class="box backbone">SegFormer-B2 Backbone<br><small>(Hierarchical Transformer Encoder)</small></div>
    <div class="arrow">↓</div>
    
    <div style="display:flex; gap:30px; align-items:center; flex-wrap:wrap; justify-content:center;">
      <div class="box module">Frequency-aware<br>Weather Feature Module</div>
      <div class="box module">Boundary Refinement<br>Module</div>
    </div>
    
    <div class="arrow">↓</div>
    <div class="box seg">Segmentation Head<br><small>(MLP Decoder → Final Mask)</small></div>
  </div>

  <div class="side-flow">
    <div class="side-box">
      <strong>Uncertainty / Confidence Branch</strong><br>
      <small>（从 Backbone 或中间特征引出）</small>
      <div style="margin:10px 0; font-size:20px;">↓</div>
      <strong>Reliability Map</strong><br>
      <small>（像素级可靠性/置信度图）</small>
    </div>
  </div>

  <div style="text-align:center; margin-top:15px;">
    <div class="control-arrow">Reliability Map 动态控制增强强度</div>
    <div style="font-size:13px; color:#666; margin-top:6px;">
      （高不确定性区域 → 更强 Frequency / Boundary 增强）
    </div>
  </div>

  <div class="note">
    <strong>核心创新点标注建议：</strong><br>
    1. Frequency-aware Weather Feature Module：针对雨雾等天气在频域进行特征增强。<br>
    2. Boundary Refinement Module：进一步锐化目标边界。<br>
    3. <span style="color:#c2185b; font-weight:bold;">Uncertainty-guided Adaptive Control</span>：由可靠性图动态调节上述两个模块的增强强度，实现“按需增强”。
  </div>
</div>
</body>
</html>