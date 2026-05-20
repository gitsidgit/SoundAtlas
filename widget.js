(function(g){
'use strict';
var CSS='\
.tlw{font-family:inherit;font-size:13px}\
.tlw-sec{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#999}\
.tlw-hint{font-size:10px;color:#bbb;margin-left:6px}\
.tlw-ov-wrap{position:relative;border:1px solid #ddd;border-radius:4px;\
  overflow:hidden;background:#fff;\
  -webkit-user-select:none;-ms-user-select:none;user-select:none}\
.tlw-ov-wrap canvas{width:100%;display:block}\
.tlw-cur-default{cursor:crosshair}\
.tlw-cur-ew{cursor:ew-resize}\
.tlw-cur-ns{cursor:ns-resize}\
.tlw-cur-grab{cursor:grab}\
.tlw-cur-grabbing{cursor:grabbing}\
.tlw-brush{position:absolute;top:0;height:100%;pointer-events:none;background:rgba(51,122,183,0.07)}\
.tlw-vbrush{position:absolute;pointer-events:none;background:rgba(100,149,200,0.06);border-top:1px solid rgba(100,149,200,0.35);border-bottom:1px solid rgba(100,149,200,0.35);box-sizing:border-box}\
.tlw-tip{position:absolute;bottom:4px;left:50%;transform:translateX(-50%);\
  background:rgba(0,0,0,.7);color:#fff;font-size:9px;padding:2px 7px;\
  border-radius:3px;white-space:nowrap;pointer-events:none}\
.tlw-axis{display:flex;justify-content:space-between;font-size:10px;\
  color:#999;margin-top:2px;padding:0 2px;margin-bottom:6px}\
.tlw-sel{font-size:11px;color:#555;background:#fff;border:1px solid #ddd;\
  border-radius:4px;padding:5px 12px;display:flex;flex-wrap:wrap;\
  gap:0;margin-bottom:10px}\
.tlw-sel span{margin-right:16px}\
.tlw-det-toggle{background:none;border:none;padding:0;cursor:pointer;\
  display:flex;align-items:center}\
.tlw-det-body{overflow:hidden;transition:max-height .25s ease}\
.tlw-det-wrap{position:relative;border:1px solid #ddd;border-radius:4px;\
  overflow:hidden;background:#fff;cursor:ew-resize}\
.tlw-det-wrap canvas{width:100%;display:block}\
.tlw-ctrl{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:8px}\
.tlw-ovz{display:flex;align-items:center;gap:6px;margin-bottom:6px}\
.tlw-ovz input[type=range]{width:100px;vertical-align:middle}\
.tlw-scope-on{background-color:#337ab7!important;color:#fff!important;\
  border-color:#2e6da4!important}\
.tlw-row-hi{background:#fffbe6!important;\
  outline:2px solid #f0ad4e;outline-offset:-2px}';

function injectCSS(){
  if(document.getElementById('tlw-css')) return;
  var s=document.createElement('style');
  s.id='tlw-css'; s.textContent=CSS;
  document.head.appendChild(s);
}
function fmtD(ts){ return new Date(ts).toLocaleString('en-GB',{month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit'}); }
function fmtS(ms){
  if(ms<60000) return Math.round(ms)+'ms';
  if(ms<3600000) return Math.round(ms/60000)+'m';
  if(ms<86400000) return (ms/3600000).toFixed(1)+'h';
  return (ms/86400000).toFixed(1)+'d';
}
function toMs(v){ return typeof v==='number'?v:new Date(v).getTime(); }

function TimelineWidget(cfg){
  injectCSS();
  this.tableId   = cfg.tableId;
  this.ctnId     = cfg.containerId;
  this.f_start   = cfg.timeStart   || 'timestamp';
  this.f_end     = cfg.timeEnd     || 'end_time';
  this.f_wave    = cfg.waveform    || null;
  this.f_id      = cfg.idField     || 'id';
  this.lazyWave  = !!cfg.lazyWave;
  this.rows      = [];
  this.gMin=0; this.gMax=1; this.gSpan=1;
  this.vS=0; this.vE=1;
  this.oS=0; this.oE=1;
  this._suppressPageSync = false;
  this.focusId   = null;
  this.detOpen   = true;
  this.scope     = 'all';
  this._detScrollTimer = null;
  this._benchSamples   = null;
  this._lastDrawMs     = 0;
  this._lastWaveCount  = 0;
  this._lastHighlightedRow = null;
  
  this.drag      = null; // {mode: 'L'|'R'|'M'|'V', anchor: num}
  this.edgeRAF   = null;
  this.EDGE      = 0.05;
  this.ESPD      = 0.005;
  this.HPIX      = 16;
  this.MIN_BRUSH = 0.02;
  this.MIN_EDGE  = 0.02;
  this.OV_PAD    = 20; // px reserved each side of canvas for time-brush handles

  // Vertical focus state
  this.focusY = 0.5; // 0..1 fraction of overview height
  this.channels = [];
  this.showLaneVbrush = true; // toggled by checkbox

  this._buildDOM();
  this._bindDOM();
  this._waitForTable();
}

TimelineWidget.prototype._buildDOM=function(){
  var uid='tlw'+Math.random().toString(36).slice(2,6);
  this.uid=uid;
  var H=[
    '<div class="tlw" id="'+uid+'">',
    '<div style="margin-bottom:4px">',
    '  <span class="tlw-sec">Overview</span>',
    '  <span class="tlw-hint">drag brush · drag lane bar (snaps to swim lane) · click segment · scroll zooms</span>',
    '</div>',
    '<div class="tlw-ov-wrap tlw-cur-default" id="'+uid+'_ov">',
    '  <canvas id="'+uid+'_ovc" height="80"></canvas>',
    '  <div class="tlw-brush" id="'+uid+'_br">',
    '    <div class="tlw-tip" id="'+uid+'_tip"></div>',
    '  </div>',
    '  <div class="tlw-vbrush" id="'+uid+'_vbr"></div>',
    '</div>',
    '<div class="tlw-axis">',
    '  <span id="'+uid+'_oL"></span><span id="'+uid+'_oM"></span><span id="'+uid+'_oR"></span>',
    '</div>',
    '<div class="tlw-ovz">',
    '  <span class="tlw-sec" style="margin:0">Zoom:</span>',
    '  <div class="btn-group btn-group-xs">',
    '    <button class="btn btn-default" id="'+uid+'_zin">+</button>',
    '    <button class="btn btn-default" id="'+uid+'_zout">&minus;</button>',
    '    <button class="btn btn-default" id="'+uid+'_zfull">Full</button>',
    '  </div>',
    '  <span id="'+uid+'_zlbl" style="font-size:10px;color:#999;min-width:32px">1x</span>',
    '  <input type="range" id="'+uid+'_zslide" min="1" max="50" value="1" step="1">',
    '</div>',
    '<div class="tlw-sel">',
    '  <span><strong>From:</strong>&nbsp;<span id="'+uid+'_sF">—</span></span>',
    '  <span><strong>To:</strong>&nbsp;<span id="'+uid+'_sT">—</span></span>',
    '  <span><strong>Span:</strong>&nbsp;<span id="'+uid+'_sS">—</span></span>',
    '</div>',
    '<div style="margin-bottom:4px">',
    '  <button class="tlw-det-toggle" id="'+uid+'_dtog">',
    '    <span class="tlw-sec" style="vertical-align:middle">Detail (Single Lane Focus)</span>',
    '    <span id="'+uid+'_dcaret" style="font-size:10px;color:#aaa;margin-left:4px">&#9660;</span>',
    '    <span class="tlw-hint">click segment to zoom · scroll to pan</span>',
    '  </button>',
    '</div>',
    '<div class="tlw-det-body" id="'+uid+'_dbody" style="max-height:200px">',
    '  <div class="tlw-det-wrap" id="'+uid+'_dw">',
    '    <canvas id="'+uid+'_detc" height="80"></canvas>',
    '  </div>',
    '  <div class="tlw-axis">',
    '    <span id="'+uid+'_dL"></span><span id="'+uid+'_dM"></span><span id="'+uid+'_dR"></span>',
    '  </div>',
    '</div>',
    '<div class="tlw-ctrl">',
    '  <div class="btn-group btn-group-sm">',
    '    <button class="btn btn-default" id="'+uid+'_v2">2x</button>',
    '    <button class="btn btn-default" id="'+uid+'_v5">5x</button>',
    '    <button class="btn btn-default" id="'+uid+'_v10">10x</button>',
    '    <button class="btn btn-default" id="'+uid+'_vr">Reset</button>',
    '  </div>',
    '  <div class="btn-group btn-group-sm" style="margin-left:auto">',
    '    <button class="btn btn-default tlw-scope-on" id="'+uid+'_sAll">All data</button>',
    '    <button class="btn btn-default" id="'+uid+'_sPage">This page</button>',
    '  </div>',
    '  <label style="font-size:11px;color:#555;margin:0 0 0 6px;cursor:pointer">',
    '    <input type="checkbox" id="'+uid+'_lazy"'+(this.lazyWave?' checked':'')+'>&nbsp;Lazy waveforms',
    '  </label>',
    '  <label style="font-size:11px;color:#555;margin:0 0 0 6px;cursor:pointer" id="'+uid+'_vbrlbl">',
    '    <input type="checkbox" id="'+uid+'_vbrchk" checked>&nbsp;Lane brush',
    '  </label>',
    '  <button class="btn btn-default btn-xs" id="'+uid+'_bench" style="margin-left:6px">&#9654; Benchmark</button>',
    '</div>',
    '<div id="'+uid+'_benchout" style="display:none;margin-bottom:8px;font-size:11px;background:#fff;border:1px solid #ddd;border-radius:4px;padding:6px 10px">',
    '</div>',
    '</div>'
  ].join('');
  document.querySelector(this.ctnId).innerHTML=H;
};
TimelineWidget.prototype._el=function(s){ return document.getElementById(this.uid+'_'+s); };
TimelineWidget.prototype._set=function(s,v){ var e=this._el(s); if(e) e.textContent=v; };

TimelineWidget.prototype._waitForTable=function(attempt){
  var self=this;
  attempt = attempt || 0;
  if(attempt > 100){
    console.error('TimelineWidget: Tabulator table not found after 10s –– check tableId "'+this.tableId+'"');
    return;
  }
  var tables=Tabulator.findTable(this.tableId);
  if(!tables||!tables.length){ setTimeout(function(){ self._waitForTable(attempt+1); },100); return; }
  this.tbl=tables[0];
  this.tbl.on('dataLoaded', function(){ setTimeout(function(){ self._ingest(); },80); });
  this.tbl.on('pageLoaded', function(){ if(self._suppressPageSync) return; self._onPageChange(); });
  this.tbl.on('rowClick', function(e,row){ self._onRowClick(row); });
  setTimeout(function(){ self._ingest(); },120);
};

TimelineWidget.prototype._ingest=function(){
  if(!this.tbl) return;
  var raw=this.tbl.getData();
  if(!raw||!raw.length) return;
  var self=this;
  this.rows=raw.map(function(r){
    return { id : String(r[self.f_id]), ts : toMs(r[self.f_start]), te : toMs(r[self.f_end]), svg: self.f_wave ? (r[self.f_wave]||null) : null, raw: r };
  }).filter(function(r){ return !isNaN(r.ts)&&!isNaN(r.te); }).sort(function(a,b){ return a.ts-b.ts; });
  if(!this.rows.length) return;
  this.gMin = this.rows[0].ts;
  this.gMax = Math.max.apply(null,this.rows.map(function(r){return r.te;}));
  this.gSpan = this.gMax-this.gMin||1;
  var seen={}, ch=[];
  this.rows.forEach(function(r){ var c=r.raw.channel; if(c!==undefined&&!seen[c]){ seen[c]=1; ch.push(c); } });
  this.channels=ch;
  this.vS=0; this.vE=1; this.oS=0; this.oE=1;
  this._draw();
};

TimelineWidget.prototype._zoomToRows=function(pageRows){
  if(!pageRows.length) return;
  var ts=pageRows[0].ts, te=pageRows[0].te;
  pageRows.forEach(function(r){ ts=Math.min(ts,r.ts); te=Math.max(te,r.te); });
  var pad=Math.max((te-ts)*0.08, this.gSpan*0.005);
  this.vS=Math.max(0,(ts-pad-this.gMin)/this.gSpan);
  this.vE=Math.min(1,(te+pad-this.gMin)/this.gSpan);
  var ovPad=(te-ts)*0.15;
  this.oS=Math.max(0,(ts-ovPad-this.gMin)/this.gSpan);
  this.oE=Math.min(1,(te+ovPad-this.gMin)/this.gSpan);
  if(this.oE-this.oS<0.04){ this.oE=Math.min(1,this.oS+0.04); }
};

TimelineWidget.prototype._onPageChange=function(){
  var pageRows=this._getPageRows();
  if(!pageRows.length){ this._draw(); return; }
  this._zoomToRows(pageRows);
  this._draw();
};

TimelineWidget.prototype._syncTableToView=function(){
  if(!this.tbl) return;
  var self=this;
  var vs=this.gMin+this.vS*this.gSpan, ve=this.gMin+this.vE*this.gSpan;
  var hit=this.rows.filter(function(r){ return r.te>vs&&r.ts<ve; });
  if(!hit.length) return;
  var targetId=hit[0].id;
  try{
    var active=this.tbl.getData('active');
    var ps=this.tbl.getPageSize()||25;
    var idx=-1;
    for(var i=0;i<active.length;i++){ if(String(active[i][self.f_id])===targetId){ idx=i; break; } }
    if(idx<0) return;
    var targetPage=Math.floor(idx/ps)+1;
    var curPage=this.tbl.getPage();
    if(targetPage===curPage){ self._drawDet(); return; }
    this._suppressPageSync=true;
    this.tbl.setPage(targetPage).then(function(){
      self._suppressPageSync=false;
      self._drawDet();
    });
  }catch(e){ this._suppressPageSync=false; }
};

TimelineWidget.prototype._onRowClick=function(row){
  var rd=row.getData(), id=String(rd[this.f_id]), match=this._rowById(id);
  if(!match) return;
  this.focusId=id;
  
  // Sync Vertical Focus to this row's channel
  var laneIdx = this.channels.indexOf(match.raw.channel);
  this.focusY = (laneIdx + 0.5) / this.channels.length;

  var pad=(match.te-match.ts)*0.4;
  this.vS=Math.max(0,(match.ts-pad-this.gMin)/this.gSpan);
  this.vE=Math.min(1,(match.te+pad-this.gMin)/this.gSpan);
  this._scrollOvToBrush();
  if(!this.detOpen) this._toggleDet();
  this._draw();
  this._highlightRow(id);
};

TimelineWidget.prototype._getPageRows=function(){
  if(!this.tbl) return [];
  try{
    var active=this.tbl.getData('active');
    var ps=this.tbl.getPageSize()||25, pg=this.tbl.getPage()||1;
    if(pg<1) pg=1;
    var self=this, idField=this.f_id, pageIds={};
    active.slice((pg-1)*ps, pg*ps).forEach(function(r){ pageIds[String(r[idField])]=1; });
    return this.rows.filter(function(r){ return !!pageIds[r.id]; });
  }catch(e){ return []; }
};

TimelineWidget.prototype._detailRows=function(){
  var self=this;
  var vs=this.gMin+this.vS*this.gSpan, ve=this.gMin+this.vE*this.gSpan;
  var inView=this.rows.filter(function(r){ return r.te>vs&&r.ts<ve; });
  if(!this.lazyWave) return inView;
  var pageRows=this._getPageRows(), pageSet={};
  pageRows.forEach(function(r){ pageSet[r.id]=1; });
  return inView.filter(function(r){ return !!pageSet[r.id]; });
};

var PAL=['#337ab7','#f0ad4e','#d9534f','#5cb85c','#9b59b6','#e67e22','#1abc9c','#e91e63'];
var LVL={'INFO':'#337ab7','WARN':'#f0ad4e','ERROR':'#d9534f'};
TimelineWidget.prototype._color=function(r){
  if(r.raw.level&&LVL[r.raw.level]) return LVL[r.raw.level];
  return PAL[parseInt(r.id,10)%PAL.length]||PAL[0];
};

TimelineWidget.prototype._ovW=function(){ return this._el('ov').clientWidth||900; };
// Map a global fraction to a canvas pixel, respecting OV_PAD margins
TimelineWidget.prototype._f2px=function(f){
  var pad=this.OV_PAD, W=this._ovW();
  return Math.round(pad+(f-this.oS)/(this.oE-this.oS)*(W-pad*2));
};
TimelineWidget.prototype._px2f=function(px){
  var pad=this.OV_PAD, W=this._ovW();
  return this.oS+((px-pad)/(W-pad*2))*(this.oE-this.oS);
};
TimelineWidget.prototype._mPx=function(e){
  var rc=this._el('ov').getBoundingClientRect();
  return Math.max(0,Math.min(rc.width,e.clientX-rc.left));
};
TimelineWidget.prototype._gf=function(ts){ return Math.max(0,Math.min(1,(ts-this.gMin)/this.gSpan)); };

TimelineWidget.prototype._scrollOvToBrush=function(){
  var mid=(this.vS+this.vE)/2, ow=this.oE-this.oS;
  if(mid<this.oS||mid>this.oE){
    this.oS=Math.max(0,mid-ow/2);
    this.oE=Math.min(1,this.oS+ow);
    if(this.oE>1){this.oE=1;this.oS=Math.max(0,1-ow);}
  }
};

TimelineWidget.prototype._snapFocusY=function(rawY){
  // rawY is 0..1 fraction of canvas height. Snap to the centre of the nearest lane.
  var n=this.channels.length||1;
  var lane=Math.floor(rawY*n);
  lane=Math.max(0,Math.min(n-1,lane));
  return (lane+0.5)/n;
};

TimelineWidget.prototype._drawOv=function(){
  var cv=this._el('ovc');
  var W=this._el('ov').clientWidth||900; cv.width=W; cv.height=80;
  var ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,80);
  var self=this, n=this.rows.length; if(!n) return;

  var TOP=5, BOT=75, DH=BOT-TOP;
  var laneCount = this.channels.length || 1;
  var slotH = DH / laneCount;
  var barH = Math.min(slotH * 0.8, 6);
  var pad = this.OV_PAD;

  this.rows.forEach(function(r){
    var f1=self._gf(r.ts), f2=self._gf(r.te);
    if(f2<self.oS||f1>self.oE) return;
    var x1=Math.max(pad,self._f2px(f1)), x2=Math.min(W-pad,self._f2px(f2));
    if(x2<x1+2) x2=x1+2;

    var laneIdx = self.channels.indexOf(r.raw.channel);
    var y = TOP + (laneIdx * slotH) + (slotH - barH)/2;

    ctx.fillStyle=self._color(r)+'cc';
    ctx.fillRect(x1,y,x2-x1,barH);
  });

  // Time Brush
  var bx1=Math.max(pad,this._f2px(this.vS)), bx2=Math.min(W-pad,this._f2px(this.vE));
  ctx.fillStyle='rgba(51,122,183,0.1)'; ctx.fillRect(bx1,0,bx2-bx1,80);
  ctx.strokeStyle='#337ab7'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(bx1,0); ctx.lineTo(bx1,80); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(bx2,0); ctx.lineTo(bx2,80); ctx.stroke();
  this._drawHandle(ctx,bx1,40); this._drawHandle(ctx,bx2,40);

  var br=this._el('br');
  br.style.left=bx1+'px'; br.style.width=Math.max(8,bx2-bx1)+'px';
  br.style.top='0'; br.style.height='80px';

  // Lane-focus brush — only when: enabled, and more than one lane
  var vbr=this._el('vbr');
  var showVB = this.showLaneVbrush && laneCount > 1;
  if(!showVB){
    vbr.style.display='none';
    // Also hide the label so it doesn't clutter single-lane views
    var lbl=this._el('vbrlbl'); if(lbl) lbl.style.display= laneCount<=1 ? 'none':'';
    return;
  }
  var lbl=this._el('vbrlbl'); if(lbl) lbl.style.display='';

  var focusedLane=Math.floor(this.focusY*laneCount);
  focusedLane=Math.max(0,Math.min(laneCount-1,focusedLane));
  var fy1=TOP+focusedLane*slotH, fy2=fy1+slotH;
  var fyCtr=(fy1+fy2)/2;

  // Shaded band — subtle tint only
  ctx.fillStyle='rgba(100,149,200,0.08)';
  ctx.fillRect(0,fy1,W,fy2-fy1);
  // Dashed top and bottom border lines — lighter than time brush
  ctx.save();
  ctx.strokeStyle='rgba(100,149,200,0.5)'; ctx.lineWidth=1;
  ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(0,fy1); ctx.lineTo(W,fy1); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,fy2); ctx.lineTo(W,fy2); ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
  // Handles — same pill shape, muted colour
  this._drawHHandle(ctx, pad/2, fyCtr, slotH);
  this._drawHHandle(ctx, W-pad/2, fyCtr, slotH);

  // Sync DOM overlay
  vbr.style.display='';
  vbr.style.left=pad+'px'; vbr.style.width=(W-pad*2)+'px';
  vbr.style.top=fy1+'px'; vbr.style.height=(fy2-fy1)+'px';
};

TimelineWidget.prototype._drawDet=function(){
  var t0=performance.now();
  var waveCount=0;
  var cv=this._el('detc');
  var W=this._el('dw').clientWidth||900; cv.width=W; cv.height=80;
  var ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,80);
  if(!this.rows.length) return;
  
  var vs=this.gMin+this.vS*this.gSpan, ve=this.gMin+this.vE*this.gSpan, sp=ve-vs||1;
  var self=this;
  function toX(ts){ return (ts-vs)/sp*W; }

  ctx.strokeStyle='#ececec'; ctx.lineWidth=1;
  [1,2,3].forEach(function(i){
    ctx.beginPath(); ctx.moveTo(W/4*i,0); ctx.lineTo(W/4*i,80); ctx.stroke();
  });

  // Identify which channel is currently focused by focusY
  var focusedLaneIdx = Math.floor(this.focusY * this.channels.length);
  focusedLaneIdx = Math.max(0, Math.min(this.channels.length-1, focusedLaneIdx));
  var focusedChan = this.channels[focusedLaneIdx];

  var rows=this._detailRows();
  rows.forEach(function(r){
    // Only draw the row if it belongs to the focused channel
    if(r.raw.channel !== focusedChan) return;

    var x1 = toX(r.ts);
    var x2 = toX(r.te);
    var actualWidth = x2 - x1;
    var c=self._color(r);
    
    ctx.save();
    ctx.beginPath(); ctx.rect(0,0,W,80); ctx.clip();

    ctx.fillStyle=c+'22'; ctx.fillRect(x1,4,actualWidth,72);
    ctx.fillStyle=c+'cc'; ctx.fillRect(x1,4,2,72);

    if(r.svg && actualWidth >= 3){
      ctx.save();
      ctx.beginPath(); ctx.rect(x1,4,actualWidth,72); ctx.clip();
      ctx.strokeStyle=c+'dd'; ctx.lineWidth=1.2;
      self._svgToCanvas(ctx,r.svg,x1,4,actualWidth,72);
      ctx.restore();
      waveCount++;
    }
    if(r.id===String(self.focusId)){
      ctx.strokeStyle='#f0ad4e'; ctx.lineWidth=2.5;
      ctx.strokeRect(x1+1,5,actualWidth-2,70);
    }
    ctx.restore();
  });
  
  var elapsed=performance.now()-t0;
  this._lastDrawMs=elapsed;
  this._lastWaveCount=waveCount;
  if(this._benchSamples) this._benchSamples.push({ms:elapsed,waves:waveCount});
};

TimelineWidget.prototype._svgToCanvas=function(ctx,svg,x,y,w,h){
  var vb=svg.match(/viewBox=["'][^"']*["']/);
  var svgW=400, svgH=80;
  if(vb){
    var p=vb[0].replace(/viewBox=["']/,'').replace(/["']/,'').trim().split(/[\s,]+/);
    svgW=parseFloat(p[2])||400; svgH=parseFloat(p[3])||80;
  }
  var sx=w/svgW, sy=h/svgH;
  var pd=svg.match(/\bd=["']([^"']+)["']/);
  if(pd){
    var cmds=pd[1].match(/[MmLlCcQqZz][^MmLlCcQqZz]*/g)||[];
    var curX=x, curY=y; // track pen position for relative commands
    ctx.beginPath();
    cmds.forEach(function(cmd){
      var t=cmd[0];
      var nums=(cmd.slice(1).trim().match(/-?[\d.eE+-]+/g)||[]).map(Number);
      if(t==='M'&&nums.length>=2){
        for(var i=0;i+1<nums.length;i+=2){
          curX=x+nums[i]*sx; curY=y+nums[i+1]*sy;
          ctx.moveTo(curX,curY);
        }
      } else if(t==='L'&&nums.length>=2){
        for(var i=0;i+1<nums.length;i+=2){
          curX=x+nums[i]*sx; curY=y+nums[i+1]*sy;
          ctx.lineTo(curX,curY);
        }
      } else if(t==='m'&&nums.length>=2){
        for(var i=0;i+1<nums.length;i+=2){
          curX+=nums[i]*sx; curY+=nums[i+1]*sy;
          ctx.moveTo(curX,curY);
        }
      } else if(t==='l'&&nums.length>=2){
        for(var i=0;i+1<nums.length;i+=2){
          curX+=nums[i]*sx; curY+=nums[i+1]*sy;
          ctx.lineTo(curX,curY);
        }
      } else if(t==='Z'||t==='z'){ ctx.closePath(); }
    });
    ctx.stroke(); return;
  }
  var pp=svg.match(/points=["']([^"']+)["']/);
  if(pp){
    var nums=(pp[1].trim().match(/-?[\d.eE+-]+/g)||[]).map(Number);
    ctx.beginPath();
    for(var i=0;i+1<nums.length;i+=2)
      i===0?ctx.moveTo(x+nums[i]*sx,y+nums[i+1]*sy)
           :ctx.lineTo(x+nums[i]*sx,y+nums[i+1]*sy);
    ctx.stroke();
  }
};

TimelineWidget.prototype._drawHandle=function(ctx,cx,cy){
  var W=11,H=22,R=4,xl=cx-W/2,yt=cy-H/2;
  function rr(){
    ctx.beginPath();
    ctx.moveTo(xl+R,yt); ctx.lineTo(xl+W-R,yt); ctx.quadraticCurveTo(xl+W,yt,xl+W,yt+R);
    ctx.lineTo(xl+W,yt+H-R); ctx.quadraticCurveTo(xl+W,yt+H,xl+W-R,yt+H);
    ctx.lineTo(xl+R,yt+H); ctx.quadraticCurveTo(xl,yt+H,xl,yt+H-R);
    ctx.lineTo(xl,yt+R); ctx.quadraticCurveTo(xl,yt,xl+R,yt); ctx.closePath();
  }
  ctx.save();
  ctx.shadowColor='rgba(0,0,0,0.18)'; ctx.shadowBlur=4;
  ctx.fillStyle='#fff'; rr(); ctx.fill(); ctx.shadowBlur=0;
  ctx.strokeStyle='#337ab7'; ctx.lineWidth=1.5; rr(); ctx.stroke();
  ctx.strokeStyle='#aaa'; ctx.lineWidth=1;
  [-4,0,4].forEach(function(dy){
    ctx.beginPath(); ctx.moveTo(cx-3,cy+dy); ctx.lineTo(cx+3,cy+dy); ctx.stroke();
  });
  ctx.restore();
};

TimelineWidget.prototype._drawHHandle=function(ctx,cx,cy,laneH){
  // Horizontal handle: wider than tall, griplines run vertically
  var W=22, H=Math.max(10,Math.min(18,laneH*0.7)), R=4;
  var xl=cx-W/2, yt=cy-H/2;
  function rr(){
    ctx.beginPath();
    ctx.moveTo(xl+R,yt); ctx.lineTo(xl+W-R,yt); ctx.quadraticCurveTo(xl+W,yt,xl+W,yt+R);
    ctx.lineTo(xl+W,yt+H-R); ctx.quadraticCurveTo(xl+W,yt+H,xl+W-R,yt+H);
    ctx.lineTo(xl+R,yt+H); ctx.quadraticCurveTo(xl,yt+H,xl,yt+H-R);
    ctx.lineTo(xl,yt+R); ctx.quadraticCurveTo(xl,yt,xl+R,yt); ctx.closePath();
  }
  ctx.save();
  ctx.shadowColor='rgba(0,0,0,0.12)'; ctx.shadowBlur=3;
  ctx.fillStyle='#fff'; rr(); ctx.fill(); ctx.shadowBlur=0;
  ctx.strokeStyle='rgba(100,149,200,0.7)'; ctx.lineWidth=1; rr(); ctx.stroke();
  // Vertical griplines
  ctx.strokeStyle='#bbb'; ctx.lineWidth=1;
  [-4,0,4].forEach(function(dx){
    ctx.beginPath(); ctx.moveTo(cx+dx,cy-3); ctx.lineTo(cx+dx,cy+3); ctx.stroke();
  });
  ctx.restore();
};


TimelineWidget.prototype._drawLabels=function(){
  if(!this.rows.length) return;
  var vs=this.gMin+this.vS*this.gSpan, ve=this.gMin+this.vE*this.gSpan;
  var os=this.gMin+this.oS*this.gSpan, oe=this.gMin+this.oE*this.gSpan;
  this._set('oL',fmtD(os)); this._set('oM',fmtD((os+oe)/2)); this._set('oR',fmtD(oe));
  this._set('dL',fmtD(vs)); this._set('dM',fmtD((vs+ve)/2)); this._set('dR',fmtD(ve));
  this._set('sF',fmtD(vs)); this._set('sT',fmtD(ve)); this._set('sS',fmtS(ve-vs));
  this._el('tip').textContent=fmtD(vs)+' \u2192 '+fmtD(ve)+' ('+fmtS(ve-vs)+')';
  var lvl=Math.round(1/(this.oE-this.oS));
  this._set('zlbl',lvl+'x');
  var sl=this._el('zslide'); if(sl) sl.value=Math.min(50,Math.max(1,lvl));
};

TimelineWidget.prototype._runBenchmark=function(){
  var self=this;
  var btn=this._el('bench'), out=this._el('benchout');
  btn.disabled=true; btn.textContent='Running…';
  out.style.display='block';
  out.innerHTML='<em>Benchmarking…</em>';
  var STEPS=30;
  var origLazy=this.lazyWave, origVS=this.vS, origVE=this.vE;
  var W=this.vE-this.vS;
  function runPass(lazy, done){
    self.lazyWave=lazy;
    self._benchSamples=[];
    var step=0, pos=0;
    function frame(){
      pos=step/STEPS;
      self.vS=Math.max(0,Math.min(1-W,pos*(1-W)));
      self.vE=self.vS+W;
      self._drawDet();
      step++;
      if(step<=STEPS) requestAnimationFrame(frame);
      else done(self._benchSamples.slice());
    }
    requestAnimationFrame(frame);
  }
  function stats(samples){
    if(!samples.length) return {med:0,min:0,max:0,waves:0};
    var ms=samples.map(function(s){return s.ms;}).sort(function(a,b){return a-b;});
    var med=ms[Math.floor(ms.length/2)];
    var waves=samples.reduce(function(a,s){return a+s.waves;},0)/samples.length;
    return {med:med.toFixed(2), min:ms[0].toFixed(2), max:ms[ms.length-1].toFixed(2), waves:Math.round(waves)};
  }
  function bar(ms, maxMs){
    var pct=Math.min(100, (ms/maxMs*100)).toFixed(1);
    return '<div style="height:10px;background:#e9ecef;border-radius:3px;margin:3px 0 6px">'
         + '<div style="height:100%;width:'+pct+'%;border-radius:3px;background:#337ab7"></div></div>';
  }
  runPass(false, function(samplesOff){
    runPass(true, function(samplesOn){
      self.lazyWave=origLazy;
      self.vS=origVS; self.vE=origVE;
      self._draw();
      self._benchSamples=null;
      btn.disabled=false; btn.textContent='▶ Benchmark';
      var off=stats(samplesOff), on=stats(samplesOn);
      var maxMs=Math.max(parseFloat(off.med),parseFloat(on.med),1)*1.3;
      var speedup=(parseFloat(off.med)/Math.max(parseFloat(on.med),0.01)).toFixed(1);
      var waveReduce=off.waves>0?Math.round((1-on.waves/off.waves)*100):0;
      out.innerHTML=
        '<strong>Detail draw benchmark</strong> &nbsp;<small style="color:#999">('+STEPS+' pan frames each pass)</small><br>'
        +'<table style="width:100%;margin-top:6px;font-size:11px;border-collapse:collapse">'
        +'<tr><th style="text-align:left;padding:2px 6px 2px 0;color:#555"></th>'
        +'<th style="color:#555;padding:2px 4px">Median</th>'
        +'<th style="color:#555;padding:2px 4px">Min</th>'
        +'<th style="color:#555;padding:2px 4px">Max</th>'
        +'<th style="color:#555;padding:2px 4px">Avg waveforms/frame</th></tr>'
        +'<tr><td style="padding:2px 6px 2px 0"><strong>Lazy OFF</strong></td>'
        +'<td style="padding:2px 4px">'+off.med+'ms</td>'
        +'<td style="padding:2px 4px;color:#999">'+off.min+'ms</td>'
        +'<td style="padding:2px 4px;color:#999">'+off.max+'ms</td>'
        +'<td style="padding:2px 4px">'+off.waves+'</td></tr>'
        +'<tr><td colspan="5">'+bar(off.med,maxMs)+'</td></tr>'
        +'<tr><td style="padding:2px 6px 2px 0"><strong>Lazy ON</strong></td>'
        +'<td style="padding:2px 4px">'+on.med+'ms</td>'
        +'<td style="padding:2px 4px;color:#999">'+on.min+'ms</td>'
        +'<td style="padding:2px 4px;color:#999">'+on.max+'ms</td>'
        +'<td style="padding:2px 4px">'+on.waves+'</td></tr>'
        +'<tr><td colspan="5">'+bar(on.med,maxMs)+'</td></tr>'
        +'</table>'
        +'<div style="margin-top:4px;padding:4px 8px;background:#f8f9fa;border-radius:3px">'
        +'&#9889; Lazy mode is <strong>'+speedup+'x faster</strong> per frame '
        +'&mdash; drawing <strong>'+on.waves+'</strong> vs <strong>'+off.waves+'</strong> waveforms '
        +'(<strong>'+waveReduce+'%</strong> fewer SVG path operations)'
        +'</div>';
    });
  });
};

TimelineWidget.prototype._draw=function(){
  this._drawOv();
  this._drawDet();
  this._drawLabels();
};

TimelineWidget.prototype._highlightRow=function(id){
  if(!this.tbl) return;
  var self=this, f=this.f_id;
  try{
    // Remove highlight from the previously highlighted row only (O(1) vs O(n))
    if(this._lastHighlightedRow){
      try{ this._lastHighlightedRow.getElement().classList.remove('tlw-row-hi'); }catch(e){}
      this._lastHighlightedRow=null;
    }
    var found=this.tbl.getRows().filter(function(r){ return String(r.getData()[f])===id; });
    if(found.length){
      found[0].getElement().classList.add('tlw-row-hi');
      this._lastHighlightedRow=found[0];
    }
  }catch(e){}
};

TimelineWidget.prototype._rowById=function(id){
  for(var i=0;i<this.rows.length;i++) if(this.rows[i].id===id) return this.rows[i];
  return null;
};

TimelineWidget.prototype._toggleDet=function(){
  this.detOpen=!this.detOpen;
  this._el('dbody').style.maxHeight=this.detOpen?'200px':'0';
  this._el('dcaret').innerHTML=this.detOpen?'&#9660;':'&#9654;';
};

TimelineWidget.prototype._setScope=function(s){
  this.scope=s;
  this._el('sAll').className ='btn btn-default'+(s==='all'?' tlw-scope-on':'');
  this._el('sPage').className='btn btn-default'+(s==='page'?' tlw-scope-on':'');
  if(s==='all'){
    this.vS=0; this.vE=1; this.oS=0; this.oE=1;
  } else {
    var pageRows=this._getPageRows();
    if(pageRows.length) this._zoomToRows(pageRows);
  }
  this._draw();
};

TimelineWidget.prototype._ovZoom=function(f){
  var c=(this.oS+this.oE)/2, hw=(this.oE-this.oS)*f/2;
  hw=Math.max(0.01,Math.min(0.5,hw));
  this.oS=Math.max(0,c-hw); this.oE=Math.min(1,c+hw);
  this._draw();
};
TimelineWidget.prototype._ovFull=function(){ this.oS=0; this.oE=1; this._draw(); };
TimelineWidget.prototype._ovSlide=function(v){
  var w=Math.min(1,Math.max(0.02,1/Math.max(1,parseFloat(v))));
  var c=(this.oS+this.oE)/2;
  this.oS=Math.max(0,c-w/2); this.oE=Math.min(1,this.oS+w);
  if(this.oE>1){this.oE=1;this.oS=Math.max(0,1-w);}
  this._draw();
};

TimelineWidget.prototype._viewZoom=function(f){
  var m=(this.vS+this.vE)/2, h=(1/f)/2;
  this.vS=Math.max(0,m-h); this.vE=Math.min(1,m+h);
  this._scrollOvToBrush(); this._draw();
};

TimelineWidget.prototype._stopEdge=function(){
  if(this.edgeRAF){ cancelAnimationFrame(this.edgeRAF); this.edgeRAF=null; }
};
TimelineWidget.prototype._startEdge=function(dir){
  this._stopEdge();
  var self=this;
  (function step(){
    var w=self.oE-self.oS;
    self.oS=Math.max(0,Math.min(1-w,self.oS+dir*self.ESPD));
    self.oE=self.oS+w;
    self._drawOv(); self._drawLabels();
    self.edgeRAF=requestAnimationFrame(step);
  })();
};

TimelineWidget.prototype._bindDOM=function(){
  var self=this;
  var ov=this._el('ov');
  this._el('dtog').addEventListener('click',function(){ self._toggleDet(); });
  this._el('v2') .addEventListener('click',function(){ self._viewZoom(2);  });
  this._el('v5') .addEventListener('click',function(){ self._viewZoom(5);  });
  this._el('v10').addEventListener('click',function(){ self._viewZoom(10); });
  this._el('vr') .addEventListener('click',function(){ self.vS=0;self.vE=1;self._scrollOvToBrush();self._draw(); });
  this._el('sAll') .addEventListener('click',function(){ self._setScope('all');  });
  this._el('sPage').addEventListener('click',function(){ self._setScope('page'); });
  this._el('bench').addEventListener('click',function(){ self._runBenchmark(); });
  this._el('lazy').addEventListener('change',function(){
    self.lazyWave=this.checked;
    self._draw();
  });
  this._el('vbrchk').addEventListener('change',function(){
    self.showLaneVbrush=this.checked;
    self._draw();
  });
  this._el('zin')   .addEventListener('click',function(){ self._ovZoom(0.5); });
  this._el('zout')  .addEventListener('click',function(){ self._ovZoom(2);   });
  this._el('zfull') .addEventListener('click',function(){ self._ovFull();    });
  this._el('zslide').addEventListener('input',function(){ self._ovSlide(this.value); });
  
  ov.addEventListener('mousemove',function(e){
    if(self.drag) return;
    var px=self._mPx(e);
    var bx1=self._f2px(self.vS), bx2=self._f2px(self.vE);
    var gf=self._px2f(px);
    
    if(Math.abs(px-bx1)<self.HPIX||Math.abs(px-bx2)<self.HPIX)
      ov.className='tlw-ov-wrap tlw-cur-ew';
    else if(gf>self.vS&&gf<self.vE)
      ov.className='tlw-ov-wrap tlw-cur-grab';
    else {
        // Check if we are near the lane brush band — only when visible
        var nLanes=self.channels.length||1;
        if(self.showLaneVbrush && nLanes>1){
          var slotH2=(75-5)/nLanes;
          var focLane=Math.floor(self.focusY*nLanes);
          focLane=Math.max(0,Math.min(nLanes-1,focLane));
          var fy1=5+focLane*slotH2, fy2=fy1+slotH2;
          var ry2 = e.clientY - ov.getBoundingClientRect().top;
          if(ry2>=fy1-2&&ry2<=fy2+2){
            ov.className='tlw-ov-wrap tlw-cur-ns'; return;
          }
        }
        ov.className='tlw-ov-wrap tlw-cur-default';
    }
  });
  ov.addEventListener('mouseleave',function(){ if(!self.drag) ov.className='tlw-ov-wrap tlw-cur-default'; });
  ov.addEventListener('mousedown',function(e){
    var px=self._mPx(e), gf=self._px2f(px);
    var bx1=self._f2px(self.vS), bx2=self._f2px(self.vE);
    var ry = e.clientY - ov.getBoundingClientRect().top;
    var nLanesMD=self.channels.length||1;
    var inVBand=false;
    if(self.showLaneVbrush && nLanesMD>1){
      var slotHMD=(75-5)/nLanesMD;
      var focLaneMD=Math.floor(self.focusY*nLanesMD);
      focLaneMD=Math.max(0,Math.min(nLanesMD-1,focLaneMD));
      var mfy1=5+focLaneMD*slotHMD, mfy2=mfy1+slotHMD;
      inVBand=(ry>=mfy1-4&&ry<=mfy2+4);
    }

    var mode;
    if(Math.abs(px-bx1)<self.HPIX)           mode='L';
    else if(Math.abs(px-bx2)<self.HPIX)      mode='R';
    else if(gf>self.vS&&gf<self.vE)          mode='M';
    else if(inVBand)                          mode='V';
    else                                      mode='N';

    self.drag={mode:mode,anchor:gf,anchorPx:px,snapS:self.vS,snapE:self.vE};
    ov.className='tlw-ov-wrap '+((mode==='L'||mode==='R')?'tlw-cur-ew':(mode==='V'?'tlw-cur-ns':'tlw-cur-grabbing'));
    e.preventDefault();
  });

  var onMove=function(e){
    if(!self.drag) return;
    var px=self._mPx(e), gf=self._px2f(px);
    var d=self.drag, w=d.snapE-d.snapS;
    var MIN=self.MIN_BRUSH, EDGE=self.MIN_EDGE;
    
    if(d.mode==='V'){
        var ry = e.clientY - ov.getBoundingClientRect().top;
        var rawY = Math.max(0, Math.min(1, ry / 80));
        self.focusY = self._snapFocusY(rawY);
    } else if(d.mode==='L') self.vS=Math.max(EDGE,Math.min(self.vE-MIN,gf));
    else if(d.mode==='R') self.vE=Math.min(1-EDGE,Math.max(self.vS+MIN,gf));
    else if(d.mode==='M'){
      var delta=gf-d.anchor;
      self.vS=Math.max(EDGE,Math.min(1-EDGE-w,d.snapS+delta));
      self.vE=self.vS+w;
    } else {
      var lo=Math.max(EDGE,Math.min(d.anchor,gf)), hi=Math.min(1-EDGE,Math.max(d.anchor,gf));
      if(hi-lo<MIN) hi=lo+MIN;
      self.vS=lo; self.vE=hi;
    }
    var raw=px/self._ovW();
    if(raw<self.EDGE&&self.oS>0) self._startEdge(-1);
    else if(raw>(1-self.EDGE)&&self.oE<1) self._startEdge(1);
    else self._stopEdge();
    self._drawOv(); self._drawLabels();
    if(!self.lazyWave) self._drawDet();
  };
  var onUp=function(e){
    if(!self.drag) return;
    var wasMode=self.drag.mode, wasPx=self.drag.anchorPx;
    self.drag=null;
    self._stopEdge();
    ov.className='tlw-ov-wrap tlw-cur-default';
    self._draw();
    self._syncTableToView();

    // Overview click-to-navigate: N mode + cursor barely moved = treat as click
    if(wasMode==='N' && self.rows.length){
      var curPx=self._mPx(e);
      if(Math.abs(curPx-wasPx)<6){
        var gf2=self._px2f(curPx);
        var ts2=self.gMin+gf2*self.gSpan;
        var ry3=e.clientY - ov.getBoundingClientRect().top;
        var nL=self.channels.length||1;
        var sH=(75-5)/nL;
        var laneClicked=Math.max(0,Math.min(nL-1,Math.floor((ry3-5)/sH)));
        var chanClicked=self.channels[laneClicked];
        // Find closest segment in that lane around the click time
        var hits=self.rows.filter(function(r){
          return r.raw.channel===chanClicked && r.ts<=ts2 && r.te>=ts2;
        });
        if(!hits.length){
          // Widen search to nearest segment within 5% of global span
          var margin=self.gSpan*0.05;
          hits=self.rows.filter(function(r){
            return r.raw.channel===chanClicked && r.te>=ts2-margin && r.ts<=ts2+margin;
          });
          hits.sort(function(a,b){ return Math.abs((a.ts+a.te)/2-ts2)-Math.abs((b.ts+b.te)/2-ts2); });
        }
        if(hits.length){
          var r=hits[0], pad2=(r.te-r.ts)*0.4;
          self.focusId=r.id;
          self.focusY=(self.channels.indexOf(r.raw.channel)+0.5)/Math.max(1,self.channels.length);
          self.vS=Math.max(0,(r.ts-pad2-self.gMin)/self.gSpan);
          self.vE=Math.min(1,(r.te+pad2-self.gMin)/self.gSpan);
          self._scrollOvToBrush();
          if(!self.detOpen) self._toggleDet();
          self._draw();
          self._highlightRow(r.id);
          self._syncTableToView();
        }
      }
    }
  };
  document.addEventListener('mousemove',onMove);
  document.addEventListener('mouseup',  onUp);
  ov.addEventListener('wheel',function(e){
    e.preventDefault();
    var px=self._mPx(e), pivot=self._px2f(px);
    var factor=e.deltaY>0?1.3:0.77;
    var ow=self.oE-self.oS, nw=Math.min(1,Math.max(0.02,ow*factor));
    var ratio=(pivot-self.oS)/ow;
    self.oS=Math.max(0,pivot-ratio*nw);
    self.oE=Math.min(1,self.oS+nw);
    if(self.oE>1){self.oE=1;self.oS=Math.max(0,1-nw);}
    self._draw();
  },{passive:false});
  this._el('dw').addEventListener('wheel',function(e){
    e.preventDefault();
    var w=self.vE-self.vS;
    var delta=(Math.abs(e.deltaX)>Math.abs(e.deltaY)?e.deltaX:e.deltaY);
    var step=w*0.12*(delta>0?1:-1);
    self.vS=Math.max(0,Math.min(1-w,self.vS+step));
    self.vE=self.vS+w;
    self._scrollOvToBrush();
    self._drawOv(); self._drawDet(); self._drawLabels();
    clearTimeout(self._detScrollTimer);
    self._detScrollTimer=setTimeout(function(){ self._syncTableToView(); },200);
  },{passive:false});
  this._el('dw').addEventListener('click',function(e){
    if(!self.rows.length) return;
    var cv=self._el('detc'), rc=cv.getBoundingClientRect();
    var vs=self.gMin+self.vS*self.gSpan, ve=self.gMin+self.vE*self.gSpan;
    var ts=vs+((e.clientX-rc.left)/rc.width)*(ve-vs);
    var hit=self._detailRows().filter(function(r){ return r.ts<=ts&&r.te>=ts; });
    if(!hit.length) return;
    var r=hit[0], pad=(r.te-r.ts)*0.3;
    self.vS=Math.max(0,(r.ts-pad-self.gMin)/self.gSpan);
    self.vE=Math.min(1,(r.te+pad-self.gMin)/self.gSpan);
    self.focusId=r.id;
    self.focusY = (self.channels.indexOf(r.raw.channel) + 0.5) / self.channels.length;
    self._scrollOvToBrush();
    self._draw();
    self._highlightRow(r.id);
  });
  if(window.ResizeObserver){
    new ResizeObserver(function(){ self._draw(); }).observe(document.querySelector(this.ctnId));
  }
};
g.TimelineWidget=TimelineWidget;
}(window));
