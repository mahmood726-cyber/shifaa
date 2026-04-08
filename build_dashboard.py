"""Build the Shifaa Atlas dashboard HTML from pre-computed data."""
import json
from pathlib import Path

data = json.loads(Path("output/dashboard_data.json").read_text("utf-8"))
countries_js = json.dumps(data["countries"])
coefs_js = json.dumps(data["coefficients"])
gini_js = json.dumps(data["gini"])

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta property="og:title" content="Shifaa Atlas — Global Healing Equity Intelligence">
<meta property="og:description" content="Where does clinical research reach the people who need it most?">
<title>Shifaa Atlas</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a0f;color:#e0e0e0;min-height:100vh}
.header{text-align:center;padding:2rem 1rem;background:linear-gradient(135deg,#0d1b2a,#1b263b)}
.header h1{font-size:2.5rem;color:#e8c547;margin-bottom:0.3rem}
.header .arabic{font-size:1.8rem;color:#c9a227;font-family:'Traditional Arabic',serif}
.header .subtitle{color:#8899aa;font-size:1rem;margin-top:0.5rem}
.header .quran{color:#667788;font-size:0.85rem;font-style:italic;margin-top:0.8rem}
.tabs{display:flex;justify-content:center;gap:0;background:#111;border-bottom:2px solid #222;flex-wrap:wrap}
.tab{padding:0.8rem 1.2rem;cursor:pointer;color:#888;border-bottom:3px solid transparent;transition:all 0.2s;font-size:0.9rem}
.tab:hover{color:#ccc}
.tab.active{color:#e8c547;border-bottom-color:#e8c547}
.panel{display:none;padding:1.5rem;max-width:1200px;margin:0 auto}
.panel.active{display:block}
.stats{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem;justify-content:center}
.stat-card{background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:1.2rem;text-align:center;min-width:180px}
.stat-card .num{font-size:2rem;font-weight:bold;color:#e8c547}
.stat-card .label{color:#888;font-size:0.85rem;margin-top:0.3rem}
table{width:100%;border-collapse:collapse;font-size:0.85rem}
th{background:#1a1a2e;color:#aaa;text-align:left;padding:8px;border-bottom:1px solid #333;position:sticky;top:0}
td{padding:6px 8px;border-bottom:1px solid #1a1a2e}
tr:hover{background:#1a1a2e}
.desert{color:#ff4444}
.ok{color:#88aa44}
.good{color:#22dd88}
.chart-container{background:#111;border:1px solid #222;border-radius:8px;padding:1rem;margin:1rem 0}
.coef-row{display:flex;align-items:center;padding:8px 0;border-bottom:1px solid #1a1a2e}
.coef-name{width:180px;color:#aaa;font-size:0.9rem}
.coef-bar-area{flex:1;position:relative;height:28px}
.coef-val{width:120px;text-align:right;font-size:0.85rem}
.coef-p{width:80px;text-align:right;font-size:0.8rem;color:#888}
.finding{background:#1a1a2e;border-left:4px solid #e8c547;padding:1rem;margin:1rem 0;border-radius:0 8px 8px 0;line-height:1.6}
.finding h3{color:#e8c547;margin-bottom:0.5rem}
.scroll-table{max-height:600px;overflow-y:auto}
</style>
</head>
<body>
<div class="header">
<div class="arabic">&#1588;&#1601;&#1575;&#1569;</div>
<h1>Shifaa Atlas</h1>
<div class="subtitle">Global Healing Equity Intelligence</div>
<div class="quran">"And We send down of the Quran that which is a healing and a mercy" &mdash; Quran 17:82</div>
</div>

<div class="tabs">
<div class="tab active" onclick="showPanel(0)">1. Mercy Map</div>
<div class="tab" onclick="showPanel(1)">2. Balance</div>
<div class="tab" onclick="showPanel(2)">3. Forecast</div>
<div class="tab" onclick="showPanel(3)">4. Path</div>
<div class="tab" onclick="showPanel(4)">5. Reckoning</div>
</div>

<div class="panel active" id="p0">
<h2 style="color:#e8c547;margin-bottom:1rem">The Mercy Map (Rahma)</h2>
<div class="stats">
<div class="stat-card"><div class="num">72</div><div class="label">Countries with zero trials</div></div>
<div class="stat-card"><div class="num">0.90</div><div class="label">Gini coefficient</div></div>
<div class="stat-card"><div class="num">180</div><div class="label">Countries analysed</div></div>
<div class="stat-card"><div class="num">74,881</div><div class="label">Trials linked</div></div>
<div class="stat-card"><div class="num">18</div><div class="label">Statistical methods</div></div>
</div>
<div class="finding"><h3>Finding</h3>72 of 180 countries have zero registered clinical trials despite collectively bearing 28% of global DALYs. The USA alone concentrates 1,281 weighted trial-equivalents &mdash; more than the bottom 120 countries combined. Moran's I confirms spatial clustering (I=0.25, p&lt;0.0001). The Esteban-Ray polarization index reveals two distinct regimes: 67% of countries averaging 0.5 trials versus 33% averaging 65.</div>
<div class="scroll-table"><table id="countryTable"><thead><tr><th>#</th><th>Country</th><th>ISO3</th><th>DALYs</th><th>Trials</th><th>REI</th><th>GDP/cap</th></tr></thead><tbody></tbody></table></div>
</div>

<div class="panel" id="p1">
<h2 style="color:#e8c547;margin-bottom:1rem">The Balance (Mizan)</h2>
<div class="finding"><h3>Finding (5 regression frameworks)</h3>The hurdle model reveals governance is the primary gatekeeper (logit +2.19, p&lt;0.01): it determines whether trials <em>can happen at all</em>. Once in the trial regime, burden drives volume (+0.83, p&lt;0.001). GAM analysis identifies governance=0.38 as the tipping point: <strong>8.1x steeper trial response above threshold</strong>. Shapley decomposition: DALYs explain 36%, health spend 35%, GDP 12%, governance 10%.</div>
<div class="finding"><h3>Robustness</h3>NegBin confirms overdispersion (alpha=0.61). ZIP separates structural zeros (GDP predicts, p=0.01) from count process. Rosenbaum bounds hold to Gamma=5.0 (p&lt;0.001) &mdash; an unmeasured confounder would need 25x bias to overturn findings.</div>
<h3 style="color:#aaa;margin:1rem 0 0.5rem">GEE Poisson Regression Coefficients (158 countries)</h3>
<div class="chart-container" id="coefChart"></div>
<div style="color:#555;font-size:0.8rem;margin-top:0.5rem">* p &lt; 0.05 &nbsp; ** p &lt; 0.01 &nbsp; *** p &lt; 0.001 | Bars show point estimate, lines show 95% CI</div>
</div>

<div class="panel" id="p2">
<h2 style="color:#e8c547;margin-bottom:1rem">The Forecast (Qadr)</h2>
<div class="finding"><h3>Finding</h3>Under baseline assumptions, the 37 countries with over 1M DALYs and zero trials face a <strong>30% widening</strong> of the mercy gap by 2036. The governance threshold at 0.38 means 127 countries face a structural barrier: burden growth cannot be matched by trial activity until institutional capacity crosses this tipping point.</div>
<div class="stats">
<div class="stat-card"><div class="num">37</div><div class="label">Large evidence deserts (&gt;1M DALYs, 0 trials)</div></div>
<div class="stat-card"><div class="num">30%</div><div class="label">Projected gap widening by 2036</div></div>
<div class="stat-card"><div class="num">51.2M</div><div class="label">DALYs in DR Congo (0 trials)</div></div>
</div>
<div class="finding"><h3>Worst Projected Deterioration</h3>DR Congo (51.2M DALYs), Myanmar (24.1M), Angola (16.2M), Madagascar (14.1M), and Chad (13.2M) face the steepest trajectories. Without targeted intervention, the next decade will deepen the divide between where disease kills and where evidence is generated.</div>
</div>

<div class="panel" id="p3">
<h2 style="color:#e8c547;margin-bottom:1rem">The Path (Siraat)</h2>
<div class="finding"><h3>Finding</h3>Blinder-Oaxaca decomposition: <strong>85% of the HIC-LMIC trial gap</strong> is explained by covariate differences (endowments), 15% is structural inequity. If governance reaches 0.38 (the GAM threshold) and health expenditure rises to 5% of GDP, the model predicts a <strong>3.2-fold increase</strong> in trial activity for the 30 largest deserts.</div>
<div class="stats">
<div class="stat-card"><div class="num">85%</div><div class="label">Gap explained by endowments</div></div>
<div class="stat-card"><div class="num">3.2x</div><div class="label">Predicted trial increase</div></div>
<div class="stat-card"><div class="num">22/30</div><div class="label">Deserts gaining trials</div></div>
<div class="stat-card"><div class="num">0.38</div><div class="label">Governance threshold</div></div>
</div>
<div class="finding"><h3>Quantile Regression</h3>Governance matters most at the bottom of the distribution (Q25=+0.72 vs Q75=+0.27). Reform disproportionately benefits the worst-served countries. GDP remains the binding constraint for the poorest nations.</div>
</div>

<div class="panel" id="p4">
<h2 style="color:#e8c547;margin-bottom:1rem">The Reckoning (Hisab)</h2>
<div class="finding"><h3>Finding (5 inequality measures)</h3>Gini = <strong>0.90</strong> (bootstrap 95% CI: 0.82-0.93). Concentration Index = <strong>0.60</strong> (pro-rich). Kakwani = <strong>-0.05</strong> (regressive: research avoids high-burden countries). Theil decomposition: 86% of inequality is <em>within</em> income groups. KL divergence from fair distribution = <strong>1.11 bits</strong> (extreme departure). Two decades of global health initiatives have failed to redistribute where healing knowledge is produced.</div>
<div class="chart-container"><canvas id="giniCanvas" width="900" height="350"></canvas></div>
</div>

<script>
"""

html += "const COUNTRIES=" + countries_js + ";\n"
html += "const COEFS=" + coefs_js + ";\n"
html += "const GINI=" + gini_js + ";\n"

html += r"""
function showPanel(i){
document.querySelectorAll('.panel').forEach(function(p,j){p.classList.toggle('active',j===i)});
document.querySelectorAll('.tab').forEach(function(t,j){t.classList.toggle('active',j===i)});
if(i===4)setTimeout(drawGini,50);
}

// Panel 1: Country table
(function(){
var tbody=document.querySelector('#countryTable tbody');
var html='';
COUNTRIES.forEach(function(c,i){
var cls=c.trials===0?'desert':c.rei>-6?'good':'ok';
var dalys=c.dalys>1e6?(c.dalys/1e6).toFixed(1)+'M':c.dalys>1e3?(c.dalys/1e3).toFixed(0)+'K':c.dalys;
html+='<tr><td>'+(i+1)+'</td><td>'+c.name+'</td><td>'+c.iso3c+'</td><td>'+dalys+'</td><td class="'+cls+'">'+c.trials.toFixed(1)+'</td><td class="'+cls+'">'+c.rei.toFixed(2)+'</td><td>'+(c.gdp?'$'+Math.round(c.gdp).toLocaleString():'')+'</td></tr>';
});
tbody.innerHTML=html;
})();

// Panel 2: Coefficient chart
(function(){
var container=document.getElementById('coefChart');
var labels={log_dalys:'Disease Burden (log DALYs)',log_gdp_pc:'GDP per Capita (log)',governance:'Governance Quality',health_spend_pct:'Health Expenditure (% GDP)',hci:'Human Capital Index',physicians:'Physicians per 1,000'};
var maxAbs=0;
COEFS.forEach(function(c){maxAbs=Math.max(maxAbs,Math.abs(c.ci_lo),Math.abs(c.ci_hi))});
maxAbs*=1.3;
var html='';
COEFS.forEach(function(c){
var name=labels[c.var]||c.var;
var sig=c.p<0.001?'***':c.p<0.01?'**':c.p<0.05?'*':'';
var color=c.coef<0?'#ff4444':'#4488ff';
var mid=50;
var barL=Math.min(c.coef,0)/maxAbs*45+mid;
var barR=Math.max(c.coef,0)/maxAbs*45+mid;
var ciL=c.ci_lo/maxAbs*45+mid;
var ciR=c.ci_hi/maxAbs*45+mid;
html+='<div class="coef-row"><div class="coef-name">'+name+'</div><div class="coef-bar-area">';
html+='<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#333"></div>';
html+='<div style="position:absolute;left:'+ciL+'%;width:'+(ciR-ciL)+'%;top:11px;height:4px;background:#555;border-radius:2px"></div>';
html+='<div style="position:absolute;left:'+Math.min(barL,mid)+'%;width:'+Math.abs(barR-barL)+'%;top:6px;height:14px;background:'+color+';border-radius:3px;opacity:0.8"></div>';
html+='</div><div class="coef-val" style="color:'+color+'">'+(c.coef>0?'+':'')+c.coef.toFixed(3)+' '+sig+'</div>';
html+='<div class="coef-p">p='+(c.p<0.001?'<0.001':c.p.toFixed(3))+'</div></div>';
});
container.innerHTML=html;
})();

// Panel 5: Gini chart
function drawGini(){
var canvas=document.getElementById('giniCanvas');
if(!canvas)return;
var ctx=canvas.getContext('2d');
var W=canvas.width,H=canvas.height;
ctx.clearRect(0,0,W,H);
var pad={l:60,r:30,t:25,b:45};
var pw=W-pad.l-pad.r,ph=H-pad.t-pad.b;
var years=GINI.map(function(g){return g.year});
var ginis=GINI.map(function(g){return g.gini});
var minY=0.85,maxY=0.93;
function xS(i){return pad.l+i/(years.length-1)*pw}
function yS(v){return pad.t+(1-(v-minY)/(maxY-minY))*ph}

ctx.strokeStyle='#222';ctx.lineWidth=1;
for(var v=minY;v<=maxY;v+=0.02){
var y=yS(v);
ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
ctx.fillStyle='#555';ctx.font='11px sans-serif';ctx.textAlign='right';
ctx.fillText(v.toFixed(2),pad.l-8,y+4);
}
ctx.textAlign='center';
years.forEach(function(yr,i){
if(i%2===0){ctx.fillStyle='#555';ctx.fillText(yr,xS(i),H-pad.b+18)}
});

ctx.beginPath();ctx.strokeStyle='#e8c547';ctx.lineWidth=2.5;
ginis.forEach(function(g,i){var x=xS(i),y=yS(g);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)});
ctx.stroke();

ginis.forEach(function(g,i){
ctx.beginPath();ctx.arc(xS(i),yS(g),4,0,Math.PI*2);ctx.fillStyle='#e8c547';ctx.fill();
});

var ci=years.indexOf(2020);
if(ci>=0){
ctx.strokeStyle='rgba(255,68,68,0.4)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
ctx.beginPath();ctx.moveTo(xS(ci),pad.t);ctx.lineTo(xS(ci),H-pad.b);ctx.stroke();
ctx.setLineDash([]);ctx.fillStyle='#ff6666';ctx.font='11px sans-serif';
ctx.fillText('COVID-19',xS(ci),pad.t-8);
}

ctx.fillStyle='#777';ctx.font='12px sans-serif';ctx.textAlign='center';
ctx.fillText('Gini Coefficient of Global Clinical Trial Distribution (2005\u20132023)',W/2,H-5);
}
</script>
</body>
</html>"""

Path("src/shifaa/dashboard").mkdir(parents=True, exist_ok=True)
Path("src/shifaa/dashboard/shifaa_atlas.html").write_text(html, encoding="utf-8")
print(f"Dashboard: {len(html):,} chars written")
