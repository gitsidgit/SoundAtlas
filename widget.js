(function(g){
'use strict';

/* ─── CSS ─────────────────────────────────────────────────────────────────── */
var CSS='\
.tlw{font-family:inherit;font-size:13px}\
.tlw-sec{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#555}\
.tlw-hint{font-size:10px;color:#777;margin-left:6px}\
.tlw-ov-wrap{position:relative;border:1px solid #ddd;border-radius:4px;\
  overflow:hidden;background:#fafafa;\
  -webkit-user-select:none;-ms-user-select:none;user-select:none}\
.tlw-ov-wrap canvas{width:100%;display:block}\
.tlw-det-wrap{position:relative;border:1px solid #ddd;border-radius:4px;\
  overflow:hidden;background:#fff;\
  -webkit-user-select:none;-ms-user-select:none;user-select:none}\
.tlw-det-wrap canvas{width:100%;display:block}\
.tlw-cur-default{cursor:default}\
.tlw-cur-ew{cursor:ew-resize}\
.tlw-cur-grab{cursor:grab}\
.tlw-cur-grabbing{cursor:grabbing}\
.tlw-cur-pointer{cursor:pointer}\
.tlw-brush{position:absolute;top:0;height:100%;pointer-events:none;\
  background:rgba(51,122,183,0.07);box-sizing:border-box}\
.tlw-tip{position:absolute;bottom:4px;left:50%;transform:translateX(-50%);\
  background:rgba(0,0,0,.85);color:#fff;font-size:11px;padding:3px 8px;\
  border-radius:3px;white-space:nowrap;pointer-events:none;z-index:2}\
.tlw-anchors{display:flex;justify-content:space-between;font-size:11px;\
  color:#333;padding:2px 4px 0;font-weight:600;letter-spacing:0}\
.tlw-det-axis{display:flex;justify-content:space-between;font-size:11px;\
  color:#333;margin-top:2px;padding:0 4px;margin-bottom:4px;font-weight:500}\
.tlw-toolbar{display:flex;align-items:center;flex-wrap:nowrap;gap:6px;\
  padding:5px 8px;background:#f8f9fa;border:1px solid #e0e0e0;\
  border-radius:4px;margin:5px 0}\
.tlw-toolbar .tlw-stats{margin-left:auto;font-size:11px;color:#333;\
  white-space:nowrap;display:flex;gap:14px;align-items:center}\
.tlw-toolbar .tlw-stats strong{color:#111}\
.tlw-scope-on{background-color:#337ab7!important;color:#fff!important;\
  border-color:#2e6da4!important}\
.tlw-row-hi{background:#fffbe6!important;\
  outline:2px solid #6baed6;outline-offset:-2px}\
.tlw-det-header{display:flex;align-items:center;margin-bottom:3px}';

function injectCSS(){
  if(document.getElementById('tlw-css')) return;
  var s=document.createElement('style');
  s.id='tlw-css'; s.textContent=CSS;
  document.head.appendChild(s);
}

/* ─── Utilities ───────────────────────────────────────────────────────────── */
function fmtD(ts){
  return new Date(ts).toLocaleString('en-GB',{month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit'});
}
function fmtS(ms){
  if(ms<1000)    return Math.round(ms)+'ms';
  if(ms<60000)   return (ms/1000).toFixed(1)+'s';
  if(ms<3600000) return Math.round(ms/60000)+'m';
  if(ms<86400000)return (ms/3600000).toFixed(1)+'h';
  return (ms/86400000).toFixed(1)+'d';
}
function toMs(v){ return typeof v==='number'?v:new Date(v).getTime(); }
function clamp(v,lo,hi){ return v<lo?lo:v>hi?hi:v; }

/* ─── Constructor ─────────────────────────────────────────────────────────── */
function TimelineWidget(cfg){
  injectCSS();
  this.tableId  = cfg.tableId;
  this.ctnId    = cfg.containerId;
  this.f_start  = cfg.timeStart  || 'timestamp';
  this.f_end    = cfg.timeEnd    || 'end_time';
  this.f_wave   = cfg.waveform   || null;
  this.f_id     = cfg.idField    || 'id';
  this.lazyWave = !!cfg.lazyWave;

  this.rows   = [];
  this.gMin=0; this.gMax=1; this.gSpan=1;
  this.vS=0; this.vE=1;   // detail window, fractions of global span
  this.oS=0; this.oE=1;   // overview viewport, fractions of global span

  this.focusId  = null;
  this.detOpen  = true;
  this.scope    = 'all';
  this._suppressPageSync   = false;
  this._detScrollTimer     = null;
  this._lastHighlightedRow = null;
  this._benchSamples       = null;
  this._lastDrawMs         = 0;
  this._lastWaveCount      = 0;

  this.drag    = null;
  this.edgeRAF = null;
  this.EDGE       = 0.04;
  this.ESPD       = 0.004;
  this.HPIX       = 14;
  this.MIN_BRUSH  = 0.001;
  this.MIN_EDGE   = 0.0;
  this.OV_PAD     = 18;    // px each side for brush handles
  this.DET_H      = 96;    // detail canvas px height
  // Overview canvas layout — from top:
  //   ANCHOR_H  = 18px  (top timestamp labels drawn IN canvas)
  //   BAR_H     = 18px  (coalescing bar strip)
  //   GAP       =  4px
  //   RULER_H   = 22px  (tick baseline + ticks + labels)
  // Total OV_H  = 62px
  this.OV_H       = 62;
  this.OV_ANCHOR  = 0;     // y-top of anchor label zone
  this.OV_BAR_TOP = 18;    // y-top of bar strip
  this.OV_BAR_H   = 18;    // bar strip height
  this.OV_RULER   = 42;    // y of ruler baseline
  this.MERGE_PX   = 4;
  this.MIN_WAVE_PX= 40;

  this._kbZone='none';

  this._buildDOM();
  this._bindDOM();
  this._waitForTable();
}

/* ─── DOM ─────────────────────────────────────────────────────────────────── */
TimelineWidget.prototype._buildDOM=function(){
  var uid='tlw'+Math.random().toString(36).slice(2,6);
  this.uid=uid;
  var OVH=this.OV_H, DTH=this.DET_H;
  var H=[
    '<div class="tlw" id="'+uid+'">',

    /* ── Overview ── */
    '<div style="margin-bottom:2px">',
    '  <span class="tlw-sec">Overview</span>',
    '  <span class="tlw-hint">drag brush · scroll or ↑↓ zoom · click item · ← → pan</span>',
    '</div>',
    '<div class="tlw-ov-wrap tlw-cur-default" id="'+uid+'_ov">',
    '  <canvas id="'+uid+'_ovc" height="'+OVH+'"></canvas>',
    '  <div class="tlw-brush" id="'+uid+'_br">',
    '    <div class="tlw-tip" id="'+uid+'_tip"></div>',
    '  </div>',
    '</div>',

    /* ── Single toolbar between Overview and Detail ── */
    '<div class="tlw-toolbar">',
    /* Zoom controls */
    '  <span class="tlw-sec" style="margin:0;white-space:nowrap">Zoom:</span>',
    '  <div class="btn-group btn-group-xs">',
    '    <button class="btn btn-default" id="'+uid+'_zin" title="Zoom in overview">+</button>',
    '    <button class="btn btn-default" id="'+uid+'_zout" title="Zoom out overview">&minus;</button>',
    '    <button class="btn btn-default" id="'+uid+'_zfull" title="Full overview">All</button>',
    '  </div>',
    '  <span id="'+uid+'_zlbl" style="font-size:11px;color:#444;font-weight:700;min-width:28px">1x</span>',
    '  <input type="range" id="'+uid+'_zslide" min="1" max="200" value="1" step="1" style="width:80px;vertical-align:middle">',
    /* Separator */
    '  <span style="color:#ccc;margin:0 2px">|</span>',
    /* Detail zoom */
    '  <div class="btn-group btn-group-xs">',
    '    <button class="btn btn-default" id="'+uid+'_v2" title="Detail 2x zoom">2x</button>',
    '    <button class="btn btn-default" id="'+uid+'_v5" title="Detail 5x zoom">5x</button>',
    '    <button class="btn btn-default" id="'+uid+'_v10" title="Detail 10x zoom">10x</button>',
    '    <button class="btn btn-default" id="'+uid+'_vr" title="Reset detail">Reset</button>',
    '  </div>',
    /* Separator */
    '  <span style="color:#ccc;margin:0 2px">|</span>',
    /* Scope */
    '  <div class="btn-group btn-group-xs">',
    '    <button class="btn btn-default tlw-scope-on" id="'+uid+'_sAll">All</button>',
    '    <button class="btn btn-default" id="'+uid+'_sPage">Page</button>',
    '  </div>',
    /* Lazy */
    '  <label style="font-size:11px;color:#555;margin:0 0 0 2px;cursor:pointer;white-space:nowrap">',
    '    <input type="checkbox" id="'+uid+'_lazy"'+(this.lazyWave?' checked':'')+'>&nbsp;Lazy',
    '  </label>',
    /* Separator */
    '  <span style="color:#ccc;margin:0 2px">|</span>',
    '  <button class="btn btn-default btn-xs" id="'+uid+'_bench" title="Render benchmark">&#9654;</button>',
    /* Stats pushed to the right */
    '  <div class="tlw-stats">',
    '    <span><strong>From</strong> <span id="'+uid+'_sF">—</span></span>',
    '    <span><strong>To</strong> <span id="'+uid+'_sT">—</span></span>',
    '    <span><strong>Span</strong> <span id="'+uid+'_sS">—</span></span>',
    '    <span><strong>Items</strong> <span id="'+uid+'_sN">—</span></span>',
    '  </div>',
    '</div>',

    /* Benchmark output */
    '<div id="'+uid+'_benchout" style="display:none;margin-bottom:6px;font-size:11px;background:#fff;border:1px solid #ddd;border-radius:4px;padding:6px 10px"></div>',

    /* ── Detail ── */
    '<div class="tlw-det-header">',
    '  <button id="'+uid+'_dtog" style="background:none;border:none;padding:0;cursor:pointer;display:flex;align-items:center">',
    '    <span class="tlw-sec" style="vertical-align:middle">Detail</span>',
    '    <span id="'+uid+'_dcaret" style="font-size:11px;color:#666;margin-left:4px">&#9660;</span>',
    '  </button>',
    '  <span class="tlw-hint">click item to focus · scroll to pan · ← → step items</span>',
    '</div>',
    '<div id="'+uid+'_dbody" style="overflow:hidden;transition:max-height .2s ease;max-height:'+(DTH+24)+'px">',
    '  <div class="tlw-det-wrap tlw-cur-default" id="'+uid+'_dw">',
    '    <canvas id="'+uid+'_detc" height="'+DTH+'"></canvas>',
    '  </div>',
    '  <div class="tlw-det-axis">',
    '    <span id="'+uid+'_dL"></span><span id="'+uid+'_dM"></span><span id="'+uid+'_dR"></span>',
    '  </div>',
    '</div>',

    '</div>'
  ].join('');
  document.querySelector(this.ctnId).innerHTML=H;
};
TimelineWidget.prototype._el =function(s){ return document.getElementById(this.uid+'_'+s); };
TimelineWidget.prototype._set=function(s,v){ var e=this._el(s); if(e) e.textContent=v; };

/* ─── Table binding ───────────────────────────────────────────────────────── */
TimelineWidget.prototype._waitForTable=function(attempt){
  var self=this;
  attempt=attempt||0;
  if(attempt>100){
    console.error('TimelineWidget: table not found – check tableId "'+this.tableId+'"');
    return;
  }
  var tables=Tabulator.findTable(this.tableId);
  if(!tables||!tables.length){ setTimeout(function(){ self._waitForTable(attempt+1); },100); return; }
  this.tbl=tables[0];
  this.tbl.on('dataLoaded',    function(){ setTimeout(function(){ self._ingest(); },80); });
  this.tbl.on('dataFiltered',  function(){ setTimeout(function(){ self._ingestFiltered(); },80); });
  this.tbl.on('pageLoaded',    function(){ if(self._suppressPageSync) return; self._onPageChange(); });
  this.tbl.on('rowClick',      function(e,row){ self._onRowClick(row); });
  setTimeout(function(){ self._ingest(); },120);
};

/* Full re-ingest — resets viewport */
TimelineWidget.prototype._ingest=function(){
  if(!this.tbl) return;
  var raw=this.tbl.getData();
  if(!raw||!raw.length) return;
  this._buildRows(raw);
  this.vS=0; this.vE=1; this.oS=0; this.oE=1;
  this._draw();
};

/* Filter re-ingest — preserves viewport where possible */
TimelineWidget.prototype._ingestFiltered=function(){
  if(!this.tbl) return;
  var raw=this.tbl.getData('active');
  if(!raw||!raw.length){
    this.rows=[];
    this._draw();
    return;
  }
  var oldGMin=this.gMin, oldGSpan=this.gSpan;
  var oldVS=this.vS, oldVE=this.vE;
  this._buildRows(raw);
  // Re-express vS/vE in terms of new global span
  if(oldGSpan>0&&this.gSpan>0){
    var vsMs=oldGMin+oldVS*oldGSpan;
    var veMs=oldGMin+oldVE*oldGSpan;
    this.vS=clamp((vsMs-this.gMin)/this.gSpan,0,1);
    this.vE=clamp((veMs-this.gMin)/this.gSpan,0,1);
    if(this.vE-this.vS<this.MIN_BRUSH) this.vE=Math.min(1,this.vS+this.MIN_BRUSH);
    this.oS=0; this.oE=1; // reset overview to show all filtered data
  } else {
    this.vS=0; this.vE=1; this.oS=0; this.oE=1;
  }
  this._draw();
};

TimelineWidget.prototype._buildRows=function(raw){
  var self=this;
  this.rows=raw.map(function(r){
    return {
      id : String(r[self.f_id]),
      ts : toMs(r[self.f_start]),
      te : toMs(r[self.f_end]),
      svg: self.f_wave?(r[self.f_wave]||null):null,
      raw: r
    };
  }).filter(function(r){ return !isNaN(r.ts)&&!isNaN(r.te)&&r.te>=r.ts; })
    .sort(function(a,b){ return a.ts-b.ts; });
  if(!this.rows.length) return;
  this.gMin  = this.rows[0].ts;
  this.gMax  = Math.max.apply(null,this.rows.map(function(r){return r.te;}));
  this.gSpan = this.gMax-this.gMin||1;
};

/* ─── Coordinate helpers ──────────────────────────────────────────────────── */
TimelineWidget.prototype._ovW =function(){ return this._el('ov').clientWidth||900; };
TimelineWidget.prototype._detW=function(){ return this._el('dw').clientWidth||900; };
TimelineWidget.prototype._f2px=function(f){
  var pad=this.OV_PAD, W=this._ovW();
  return pad+(f-this.oS)/(this.oE-this.oS)*(W-pad*2);
};
TimelineWidget.prototype._px2f=function(px){
  var pad=this.OV_PAD, W=this._ovW();
  return this.oS+((px-pad)/(W-pad*2))*(this.oE-this.oS);
};
TimelineWidget.prototype._mPx=function(e){
  var rc=this._el('ov').getBoundingClientRect();
  return clamp(e.clientX-rc.left,0,rc.width);
};
TimelineWidget.prototype._gf=function(ts){ return clamp((ts-this.gMin)/this.gSpan,0,1); };

/* ─── View helpers ────────────────────────────────────────────────────────── */
TimelineWidget.prototype._scrollOvToBrush=function(){
  var mid=(this.vS+this.vE)/2, ow=this.oE-this.oS;
  if(mid<this.oS||mid>this.oE){
    this.oS=clamp(mid-ow/2,0,1-ow);
    this.oE=this.oS+ow;
  }
};
TimelineWidget.prototype._zoomToRows=function(rows){
  if(!rows.length) return;
  var ts=rows[0].ts, te=rows[0].te;
  rows.forEach(function(r){ ts=Math.min(ts,r.ts); te=Math.max(te,r.te); });
  var pad=Math.max((te-ts)*0.06,this.gSpan*0.004);
  this.vS=clamp((ts-pad-this.gMin)/this.gSpan,0,1);
  this.vE=clamp((te+pad-this.gMin)/this.gSpan,0,1);
  var op=Math.max((te-ts)*0.12,this.gSpan*0.01);
  this.oS=clamp((ts-op-this.gMin)/this.gSpan,0,1);
  this.oE=clamp((te+op-this.gMin)/this.gSpan,0,1);
  if(this.oE-this.oS<0.02) this.oE=Math.min(1,this.oS+0.02);
};
TimelineWidget.prototype._focusRow=function(r){
  var dur=r.te-r.ts, pad=Math.max(dur*0.3,this.gSpan*0.002);
  this.vS=clamp((r.ts-pad-this.gMin)/this.gSpan,0,1);
  this.vE=clamp((r.te+pad-this.gMin)/this.gSpan,0,1);
  this.focusId=r.id;
  this._scrollOvToBrush();
  if(!this.detOpen) this._toggleDet();
  this._draw();
  this._highlightRow(r.id);
  this._syncTableToView();
};

/* ─── Page / table sync ───────────────────────────────────────────────────── */
TimelineWidget.prototype._getPageRows=function(){
  if(!this.tbl) return [];
  try{
    var active=this.tbl.getData('active');
    var ps=this.tbl.getPageSize()||25, pg=Math.max(1,this.tbl.getPage()||1);
    var self=this, f=this.f_id, ids={};
    active.slice((pg-1)*ps,pg*ps).forEach(function(r){ ids[String(r[f])]=1; });
    return this.rows.filter(function(r){ return !!ids[r.id]; });
  }catch(e){ return []; }
};
TimelineWidget.prototype._onPageChange=function(){
  var pr=this._getPageRows();
  if(!pr.length){ this._draw(); return; }
  this._zoomToRows(pr); this._draw();
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
    var ps=this.tbl.getPageSize()||25, f=this.f_id, idx=-1;
    for(var i=0;i<active.length;i++){ if(String(active[i][f])===targetId){idx=i;break;} }
    if(idx<0) return;
    var tp=Math.floor(idx/ps)+1, cp=this.tbl.getPage();
    if(tp===cp){ self._drawDet(); return; }
    this._suppressPageSync=true;
    this.tbl.setPage(tp).then(function(){ self._suppressPageSync=false; self._drawDet(); });
  }catch(e){ this._suppressPageSync=false; }
};
TimelineWidget.prototype._onRowClick=function(row){
  var rd=row.getData(), id=String(rd[this.f_id]);
  var match=this._rowById(id);
  if(!match) return;
  this._focusRow(match);
};
TimelineWidget.prototype._detailRows=function(){
  var vs=this.gMin+this.vS*this.gSpan, ve=this.gMin+this.vE*this.gSpan;
  var inView=this.rows.filter(function(r){ return r.te>vs&&r.ts<ve; });
  if(!this.lazyWave) return inView;
  var pageIds={};
  this._getPageRows().forEach(function(r){ pageIds[r.id]=1; });
  return inView.filter(function(r){ return !!pageIds[r.id]; });
};

/* ─── Colour ──────────────────────────────────────────────────────────────── */
var BAR_CLR='#6baed6';

/* ─── Overview draw ───────────────────────────────────────────────────────── */
TimelineWidget.prototype._drawOv=function(){
  var cv=this._el('ovc');
  var W=this._ovW(); cv.width=W; cv.height=this.OV_H;
  var ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,this.OV_H);
  if(!this.rows.length) return;

  var self=this;
  var pad=this.OV_PAD;
  var OVH=this.OV_H;
  var BAR_TOP=this.OV_BAR_TOP;
  var BAR_H  =this.OV_BAR_H;
  var RULER_Y=this.OV_RULER;
  var MERGE  =this.MERGE_PX;

  /* ── Top anchor labels (L / M / R) drawn inside canvas ── */
  var os=this.gMin+this.oS*this.gSpan;
  var oe=this.gMin+this.oE*this.gSpan;
  ctx.save();
  ctx.font='bold 10px sans-serif';
  ctx.fillStyle='#222';
  ctx.textBaseline='top';
  ctx.textAlign='left';  ctx.fillText(fmtD(os), pad, 2);
  ctx.textAlign='center';ctx.fillText(fmtD((os+oe)/2), W/2, 2);
  ctx.textAlign='right'; ctx.fillText(fmtD(oe), W-pad, 2);
  ctx.restore();

  /* ── Bar strip ── */
  var intervals=[];
  this.rows.forEach(function(r){
    var f1=self._gf(r.ts), f2=self._gf(r.te);
    if(f2<self.oS||f1>self.oE) return;
    var x1=clamp(Math.round(self._f2px(f1)),pad,W-pad);
    var x2=clamp(Math.round(self._f2px(f2)),pad,W-pad);
    if(x2<x1+1) x2=x1+1;
    intervals.push({x1:x1,x2:x2,id:r.id});
  });
  intervals.sort(function(a,b){ return a.x1-b.x1; });
  var merged=[];
  intervals.forEach(function(iv){
    if(!merged.length){ merged.push({x1:iv.x1,x2:iv.x2,ids:[iv.id]}); return; }
    var last=merged[merged.length-1];
    if(iv.x1-last.x2<=MERGE){ last.x2=Math.max(last.x2,iv.x2); last.ids.push(iv.id); }
    else merged.push({x1:iv.x1,x2:iv.x2,ids:[iv.id]});
  });

  var focId=String(this.focusId);
  merged.forEach(function(m){
    var isFocus=m.ids.indexOf(focId)>=0;
    var solo=m.ids.length===1;
    var w=m.x2-m.x1;
    ctx.fillStyle=BAR_CLR+(isFocus?'ff':solo?'cc':'99');
    ctx.fillRect(m.x1,BAR_TOP,w,BAR_H);
    if(solo){ ctx.fillStyle=BAR_CLR+'ff'; ctx.fillRect(m.x1,BAR_TOP,w,2); }
    if(isFocus){
      ctx.strokeStyle='#2171b5'; ctx.lineWidth=1.5;
      ctx.strokeRect(m.x1+0.5,BAR_TOP+0.5,w-1,BAR_H-1);
    }
  });

  /* Median line */
  var MID_Y=Math.round(BAR_TOP+BAR_H/2);
  ctx.strokeStyle='rgba(107,174,214,0.35)'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,MID_Y); ctx.lineTo(W-pad,MID_Y); ctx.stroke();

  /* ── Tick ruler ── */
  this._drawTicks(ctx,W,pad,RULER_Y);

  /* ── Time brush ── */
  var bx1=clamp(Math.round(this._f2px(this.vS)),0,W);
  var bx2=clamp(Math.round(this._f2px(this.vE)),0,W);
  ctx.fillStyle='rgba(51,122,183,0.10)';
  ctx.fillRect(bx1,BAR_TOP-1,bx2-bx1,OVH-BAR_TOP+1);
  ctx.strokeStyle='#337ab7'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(bx1,BAR_TOP-1); ctx.lineTo(bx1,OVH); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(bx2,BAR_TOP-1); ctx.lineTo(bx2,OVH); ctx.stroke();
  this._drawHandle(ctx,bx1,BAR_TOP+BAR_H/2);
  this._drawHandle(ctx,bx2,BAR_TOP+BAR_H/2);

  /* DOM brush div for cursor detection */
  var br=this._el('br');
  br.style.left=bx1+'px'; br.style.width=Math.max(4,bx2-bx1)+'px';
  br.style.top='0'; br.style.height=OVH+'px';

  /* Update tooltip */
  var vs=this.gMin+this.vS*this.gSpan, ve=this.gMin+this.vE*this.gSpan;
  this._el('tip').textContent=fmtD(vs)+' \u2192 '+fmtD(ve)+' ('+fmtS(ve-vs)+')';
};

/* ─── Tick ruler ──────────────────────────────────────────────────────────── */
TimelineWidget.prototype._drawTicks=function(ctx,W,pad,RULER_Y){
  if(!this.rows.length) return;
  var os=this.gMin+this.oS*this.gSpan;
  var oe=this.gMin+this.oE*this.gSpan;
  var span=oe-os||1;

  var NICE=[100,200,500,1000,2000,5000,10000,15000,30000,
    60000,120000,300000,600000,1800000,
    3600000,7200000,21600000,43200000,
    86400000,172800000,604800000];
  var interval=NICE[NICE.length-1];
  for(var ni=0;ni<NICE.length;ni++){
    if(span/NICE[ni]<=7){ interval=NICE[ni]; break; }
  }
  var minor=interval/5;

  var MAJ_H=5, MIN_H=3;
  // Labels sit ABOVE the baseline (baseline at RULER_Y, labels above)
  var LBL_Y=RULER_Y-MAJ_H-12; // label top — 10px font + 2px gap above tick top

  ctx.save();
  ctx.font='bold 10px sans-serif';
  ctx.textAlign='center';
  ctx.textBaseline='top';

  /* Ruler baseline */
  ctx.strokeStyle='rgba(80,80,80,0.5)'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,RULER_Y); ctx.lineTo(W-pad,RULER_Y); ctx.stroke();

  /* Minor ticks — hang downward from baseline */
  ctx.strokeStyle='rgba(120,120,120,0.5)'; ctx.lineWidth=1;
  var t0m=Math.ceil(os/minor)*minor;
  for(var tm=t0m;tm<=oe;tm+=minor){
    var xm=pad+(tm-os)/span*(W-pad*2);
    if(xm<pad||xm>W-pad) continue;
    ctx.beginPath(); ctx.moveTo(xm,RULER_Y); ctx.lineTo(xm,RULER_Y+MIN_H); ctx.stroke();
  }

  /* Major ticks + labels */
  ctx.strokeStyle='rgba(40,40,40,0.8)'; ctx.lineWidth=1.5;
  ctx.fillStyle='#222';
  var t0=Math.ceil(os/interval)*interval;
  for(var t=t0;t<=oe;t+=interval){
    var x=pad+(t-os)/span*(W-pad*2);
    if(x<pad||x>W-pad) continue;
    // Tick goes both up (short) and down from baseline
    ctx.beginPath(); ctx.moveTo(x,RULER_Y-MAJ_H); ctx.lineTo(x,RULER_Y+2); ctx.stroke();
    var lbl=this._tickLabel(t,interval);
    if(x>pad+22&&x<W-pad-22) ctx.fillText(lbl,x,LBL_Y);
  }

  ctx.restore();
};

TimelineWidget.prototype._tickLabel=function(ts,interval){
  var d=new Date(ts);
  if(interval>=86400000) return d.toLocaleDateString('en-GB',{day:'2-digit',month:'short'});
  if(interval>=60000)    return d.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit'});
  return d.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
};

/* ─── Detail draw ─────────────────────────────────────────────────────────── */
TimelineWidget.prototype._drawDet=function(){
  var t0=performance.now(), waveCount=0;
  var cv=this._el('detc');
  var W=this._detW(); cv.width=W; cv.height=this.DET_H;
  var ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,this.DET_H);
  if(!this.rows.length) return;

  var DTH=this.DET_H;
  var vs=this.gMin+this.vS*this.gSpan;
  var ve=this.gMin+this.vE*this.gSpan;
  var sp=ve-vs||1;
  var self=this;
  var focId=String(this.focusId);
  var MWPX=this.MIN_WAVE_PX;
  var PAD_Y=4;

  /* Grid lines */
  ctx.strokeStyle='#ececec'; ctx.lineWidth=1;
  [1,2,3].forEach(function(i){
    ctx.beginPath(); ctx.moveTo(W/4*i,0); ctx.lineTo(W/4*i,DTH); ctx.stroke();
  });

  var rows=this._detailRows();
  rows.forEach(function(r){
    // Use true item bounds (not clamped) for position/width — waveform clips naturally
    var x1=(r.ts-vs)/sp*W;
    var x2=(r.te-vs)/sp*W;
    var bw=x2-x1;
    if(bw<1) return;

    var isFocus=(r.id===focId);
    var itemH=DTH-PAD_Y*2;

    // Clip to visible canvas only — prevents waveform edge distortion
    var clipX1=Math.max(0,x1);
    var clipW =Math.min(W,x2)-clipX1;
    if(clipW<1) return;

    ctx.save();
    ctx.beginPath(); ctx.rect(clipX1,PAD_Y,clipW,itemH); ctx.clip();

    /* Background */
    ctx.fillStyle=BAR_CLR+(isFocus?'28':'18');
    ctx.fillRect(x1,PAD_Y,bw,itemH);

    /* Left edge accent */
    if(x1>=0&&x1<W){
      ctx.fillStyle=BAR_CLR+(isFocus?'ff':'cc');
      ctx.fillRect(x1,PAD_Y,Math.min(2,bw),itemH);
    }

    /* Waveform — draw at TRUE x1 with full bw so SVG isn't compressed */
    if(r.svg&&bw>=MWPX){
      ctx.strokeStyle=BAR_CLR+'cc';
      ctx.lineWidth=1.3;
      self._svgToCanvas(ctx,r.svg,x1,PAD_Y,bw,itemH);
      waveCount++;
    }

    ctx.restore();

    /* Focus ring */
    if(isFocus){
      ctx.strokeStyle='#2171b5'; ctx.lineWidth=2;
      ctx.strokeRect(clipX1+1,PAD_Y+1,clipW-2,itemH-2);
    }

    /* Divider at right edge */
    if(x2>0&&x2<W&&bw>4){
      ctx.strokeStyle='rgba(255,255,255,0.7)'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(x2,PAD_Y); ctx.lineTo(x2,PAD_Y+itemH); ctx.stroke();
    }
  });

  this._lastDrawMs=performance.now()-t0;
  this._lastWaveCount=waveCount;
  if(this._benchSamples) this._benchSamples.push({ms:this._lastDrawMs,waves:waveCount});
};

/* ─── SVG → Canvas ────────────────────────────────────────────────────────── */
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
    var curX=x, curY=y;
    ctx.beginPath();
    cmds.forEach(function(cmd){
      var t=cmd[0];
      var nums=(cmd.slice(1).trim().match(/-?[\d.eE+-]+/g)||[]).map(Number);
      if(t==='M'&&nums.length>=2){
        for(var i=0;i+1<nums.length;i+=2){ curX=x+nums[i]*sx; curY=y+nums[i+1]*sy; ctx.moveTo(curX,curY); }
      }else if(t==='L'&&nums.length>=2){
        for(var i=0;i+1<nums.length;i+=2){ curX=x+nums[i]*sx; curY=y+nums[i+1]*sy; ctx.lineTo(curX,curY); }
      }else if(t==='m'&&nums.length>=2){
        for(var i=0;i+1<nums.length;i+=2){ curX+=nums[i]*sx; curY+=nums[i+1]*sy; ctx.moveTo(curX,curY); }
      }else if(t==='l'&&nums.length>=2){
        for(var i=0;i+1<nums.length;i+=2){ curX+=nums[i]*sx; curY+=nums[i+1]*sy; ctx.lineTo(curX,curY); }
      }else if(t==='Z'||t==='z'){ ctx.closePath(); }
    });
    ctx.stroke(); return;
  }
  var pp=svg.match(/points=["']([^"']+)["']/);
  if(pp){
    var nums=(pp[1].trim().match(/-?[\d.eE+-]+/g)||[]).map(Number);
    ctx.beginPath();
    for(var i=0;i+1<nums.length;i+=2)
      i===0?ctx.moveTo(x+nums[i]*sx,y+nums[i+1]*sy):ctx.lineTo(x+nums[i]*sx,y+nums[i+1]*sy);
    ctx.stroke();
  }
};

/* ─── Handle drawing ──────────────────────────────────────────────────────── */
TimelineWidget.prototype._drawHandle=function(ctx,cx,cy){
  var HW=10, HH=20, R=3, xl=cx-HW/2, yt=cy-HH/2;
  function pill(){
    ctx.beginPath();
    ctx.moveTo(xl+R,yt); ctx.lineTo(xl+HW-R,yt); ctx.quadraticCurveTo(xl+HW,yt,xl+HW,yt+R);
    ctx.lineTo(xl+HW,yt+HH-R); ctx.quadraticCurveTo(xl+HW,yt+HH,xl+HW-R,yt+HH);
    ctx.lineTo(xl+R,yt+HH); ctx.quadraticCurveTo(xl,yt+HH,xl,yt+HH-R);
    ctx.lineTo(xl,yt+R); ctx.quadraticCurveTo(xl,yt,xl+R,yt); ctx.closePath();
  }
  ctx.save();
  ctx.shadowColor='rgba(0,0,0,0.18)'; ctx.shadowBlur=4;
  ctx.fillStyle='#fff'; pill(); ctx.fill(); ctx.shadowBlur=0;
  ctx.strokeStyle='#337ab7'; ctx.lineWidth=1.5; pill(); ctx.stroke();
  ctx.strokeStyle='#888'; ctx.lineWidth=1;
  [-3,0,3].forEach(function(dy){
    ctx.beginPath(); ctx.moveTo(cx-2.5,cy+dy); ctx.lineTo(cx+2.5,cy+dy); ctx.stroke();
  });
  ctx.restore();
};

/* ─── Labels ──────────────────────────────────────────────────────────────── */
TimelineWidget.prototype._drawLabels=function(){
  if(!this.rows.length) return;
  var vs=this.gMin+this.vS*this.gSpan, ve=this.gMin+this.vE*this.gSpan;
  this._set('dL',fmtD(vs)); this._set('dM',fmtD((vs+ve)/2)); this._set('dR',fmtD(ve));
  this._set('sF',fmtD(vs)); this._set('sT',fmtD(ve)); this._set('sS',fmtS(ve-vs));
  var nVis=this.rows.filter(function(r){ return r.te>vs&&r.ts<ve; }).length;
  this._set('sN',nVis+'');
  var lvl=Math.round(1/(this.oE-this.oS));
  this._set('zlbl',lvl+'x');
  var sl=this._el('zslide'); if(sl) sl.value=clamp(lvl,1,200);
};

/* ─── Master draw ─────────────────────────────────────────────────────────── */
TimelineWidget.prototype._draw=function(){
  this._drawOv();
  this._drawDet();
  this._drawLabels();
};

/* ─── Row helpers ─────────────────────────────────────────────────────────── */
TimelineWidget.prototype._rowById=function(id){
  for(var i=0;i<this.rows.length;i++) if(this.rows[i].id===id) return this.rows[i];
  return null;
};
TimelineWidget.prototype._highlightRow=function(id){
  if(!this.tbl) return;
  var f=this.f_id;
  try{
    if(this._lastHighlightedRow){
      try{ this._lastHighlightedRow.getElement().classList.remove('tlw-row-hi'); }catch(e){}
      this._lastHighlightedRow=null;
    }
    var found=this.tbl.getRows().filter(function(r){ return String(r.getData()[f])===id; });
    if(found.length){ found[0].getElement().classList.add('tlw-row-hi'); this._lastHighlightedRow=found[0]; }
  }catch(e){}
};

/* ─── UI state ────────────────────────────────────────────────────────────── */
TimelineWidget.prototype._toggleDet=function(){
  this.detOpen=!this.detOpen;
  this._el('dbody').style.maxHeight=this.detOpen?(this.DET_H+24)+'px':'0';
  this._el('dcaret').innerHTML=this.detOpen?'&#9660;':'&#9654;';
};
TimelineWidget.prototype._setScope=function(s){
  this.scope=s;
  this._el('sAll').className ='btn btn-default btn-xs'+(s==='all'?' tlw-scope-on':'');
  this._el('sPage').className='btn btn-default btn-xs'+(s==='page'?' tlw-scope-on':'');
  if(s==='all'){ this.vS=0;this.vE=1;this.oS=0;this.oE=1; }
  else { var pr=this._getPageRows(); if(pr.length) this._zoomToRows(pr); }
  this._draw();
};
TimelineWidget.prototype._viewZoom=function(f){
  var m=(this.vS+this.vE)/2, h=(this.vE-this.vS)/(2*f);
  this.vS=clamp(m-h,0,1); this.vE=clamp(m+h,0,1);
  if(this.vE-this.vS<this.MIN_BRUSH) this.vE=Math.min(1,this.vS+this.MIN_BRUSH);
  this._scrollOvToBrush(); this._draw();
};
TimelineWidget.prototype._ovZoom=function(f){
  /* Pivot on the brush midpoint so selection stays centred */
  var pivot=(this.vS+this.vE)/2;
  var ow=this.oE-this.oS, nw=clamp(ow*f,0.005,1);
  var ratio=(pivot-this.oS)/ow;
  this.oS=clamp(pivot-ratio*nw,0,1-nw);
  this.oE=this.oS+nw;
  this._draw();
};
TimelineWidget.prototype._ovFull=function(){ this.oS=0;this.oE=1;this._draw(); };
TimelineWidget.prototype._ovSlide=function(v){
  var w=clamp(1/Math.max(1,parseFloat(v)),0.005,1);
  var c=(this.oS+this.oE)/2;
  this.oS=clamp(c-w/2,0,1-w); this.oE=this.oS+w;
  this._draw();
};

/* ─── Edge pan ────────────────────────────────────────────────────────────── */
TimelineWidget.prototype._stopEdge=function(){
  if(this.edgeRAF){ cancelAnimationFrame(this.edgeRAF); this.edgeRAF=null; }
};
TimelineWidget.prototype._startEdge=function(dir){
  this._stopEdge();
  var self=this;
  (function step(){
    var w=self.oE-self.oS;
    self.oS=clamp(self.oS+dir*self.ESPD,0,1-w);
    self.oE=self.oS+w;
    self._drawOv(); self._drawLabels();
    self.edgeRAF=requestAnimationFrame(step);
  })();
};

/* ─── Benchmark ───────────────────────────────────────────────────────────── */
TimelineWidget.prototype._runBenchmark=function(){
  var self=this;
  var btn=this._el('bench'), out=this._el('benchout');
  btn.disabled=true;
  out.style.display='block'; out.innerHTML='<em>Benchmarking…</em>';
  var STEPS=30, origLazy=this.lazyWave, origVS=this.vS, origVE=this.vE, W=this.vE-this.vS;
  function runPass(lazy,done){
    self.lazyWave=lazy; self._benchSamples=[];
    var step=0;
    (function frame(){
      var pos=step/STEPS;
      self.vS=clamp(pos*(1-W),0,1-W); self.vE=self.vS+W;
      self._drawDet(); step++;
      if(step<=STEPS) requestAnimationFrame(frame); else done(self._benchSamples.slice());
    })();
  }
  function stats(s){
    if(!s.length) return {med:0,min:0,max:0,waves:0};
    var ms=s.map(function(x){return x.ms;}).sort(function(a,b){return a-b;});
    return {med:ms[Math.floor(ms.length/2)].toFixed(2),min:ms[0].toFixed(2),
            max:ms[ms.length-1].toFixed(2),
            waves:Math.round(s.reduce(function(a,x){return a+x.waves;},0)/s.length)};
  }
  function bar(ms,mx){
    var pct=Math.min(100,ms/mx*100).toFixed(1);
    return '<div style="height:8px;background:#e9ecef;border-radius:3px;margin:2px 0 5px">'
          +'<div style="height:100%;width:'+pct+'%;border-radius:3px;background:#6baed6"></div></div>';
  }
  runPass(false,function(off_s){
    runPass(true,function(on_s){
      self.lazyWave=origLazy; self.vS=origVS; self.vE=origVE;
      self._draw(); self._benchSamples=null;
      btn.disabled=false;
      var off=stats(off_s), on=stats(on_s);
      var mx=Math.max(parseFloat(off.med),parseFloat(on.med),1)*1.3;
      var spd=(parseFloat(off.med)/Math.max(parseFloat(on.med),0.01)).toFixed(1);
      out.innerHTML='<strong>Draw benchmark</strong>'
        +'<table style="width:100%;margin-top:4px;font-size:11px;border-collapse:collapse">'
        +'<tr><th></th><th style="color:#555">Median</th><th style="color:#555">Min</th>'
        +'<th style="color:#555">Max</th><th style="color:#555">Waves/frame</th></tr>'
        +'<tr><td><strong>Lazy OFF</strong></td><td>'+off.med+'ms</td><td style="color:#999">'+off.min+'ms</td>'
        +'<td style="color:#999">'+off.max+'ms</td><td>'+off.waves+'</td></tr>'
        +'<tr><td colspan="5">'+bar(off.med,mx)+'</td></tr>'
        +'<tr><td><strong>Lazy ON</strong></td><td>'+on.med+'ms</td><td style="color:#999">'+on.min+'ms</td>'
        +'<td style="color:#999">'+on.max+'ms</td><td>'+on.waves+'</td></tr>'
        +'<tr><td colspan="5">'+bar(on.med,mx)+'</td></tr>'
        +'</table>'
        +'<div style="margin-top:3px;padding:3px 7px;background:#f8f9fa;border-radius:3px">'
        +'&#9889; Lazy is <strong>'+spd+'x faster</strong></div>';
    });
  });
};

/* ─── Event binding ───────────────────────────────────────────────────────── */
TimelineWidget.prototype._bindDOM=function(){
  var self=this;
  var ov=this._el('ov');

  this._el('dtog') .addEventListener('click',function(){ self._toggleDet(); });
  this._el('v2')   .addEventListener('click',function(){ self._viewZoom(2);  });
  this._el('v5')   .addEventListener('click',function(){ self._viewZoom(5);  });
  this._el('v10')  .addEventListener('click',function(){ self._viewZoom(10); });
  this._el('vr')   .addEventListener('click',function(){ self.vS=0;self.vE=1;self._scrollOvToBrush();self._draw(); });
  this._el('sAll') .addEventListener('click',function(){ self._setScope('all');  });
  this._el('sPage').addEventListener('click',function(){ self._setScope('page'); });
  this._el('bench').addEventListener('click',function(){ self._runBenchmark(); });
  this._el('lazy') .addEventListener('change',function(){ self.lazyWave=this.checked; self._draw(); });
  this._el('zin')  .addEventListener('click',function(){ self._ovZoom(0.5); });
  this._el('zout') .addEventListener('click',function(){ self._ovZoom(2);   });
  this._el('zfull').addEventListener('click',function(){ self._ovFull();    });
  this._el('zslide').addEventListener('input',function(){ self._ovSlide(this.value); });

  /* ── Overview mouse ── */
  ov.addEventListener('mousemove',function(e){
    if(self.drag) return;
    var px=self._mPx(e), gf=self._px2f(px);
    var bx1=self._f2px(self.vS), bx2=self._f2px(self.vE);
    if(Math.abs(px-bx1)<self.HPIX||Math.abs(px-bx2)<self.HPIX)
      ov.className='tlw-ov-wrap tlw-cur-ew';
    else if(gf>self.vS&&gf<self.vE)
      ov.className='tlw-ov-wrap tlw-cur-grab';
    else
      ov.className='tlw-ov-wrap tlw-cur-pointer';
  });
  ov.addEventListener('mouseleave',function(){
    if(!self.drag) ov.className='tlw-ov-wrap tlw-cur-default';
  });

  ov.addEventListener('mousedown',function(e){
    var px=self._mPx(e), gf=self._px2f(px);
    var bx1=self._f2px(self.vS), bx2=self._f2px(self.vE);
    var mode;
    if(Math.abs(px-bx1)<self.HPIX)      mode='L';
    else if(Math.abs(px-bx2)<self.HPIX) mode='R';
    else if(gf>self.vS&&gf<self.vE)     mode='M';
    else                                 mode='N';
    self.drag={mode:mode,anchor:gf,anchorPx:px,snapS:self.vS,snapE:self.vE};
    ov.className='tlw-ov-wrap '+((mode==='L'||mode==='R')?'tlw-cur-ew':mode==='M'?'tlw-cur-grabbing':'tlw-cur-crosshair');
    e.preventDefault();
  });

  var onMove=function(e){
    if(!self.drag) return;
    var px=self._mPx(e), gf=self._px2f(px);
    var d=self.drag, w=d.snapE-d.snapS;
    var MIN=self.MIN_BRUSH, EDGE=self.MIN_EDGE;
    if(d.mode==='L')       self.vS=clamp(gf,EDGE,self.vE-MIN);
    else if(d.mode==='R')  self.vE=clamp(gf,self.vS+MIN,1-EDGE);
    else if(d.mode==='M'){ var delta=gf-d.anchor; self.vS=clamp(d.snapS+delta,EDGE,1-EDGE-w); self.vE=self.vS+w; }
    else if(d.mode==='N'){ var lo=clamp(Math.min(d.anchor,gf),EDGE,1); var hi=clamp(Math.max(d.anchor,gf),0,1-EDGE); if(hi-lo<MIN)hi=lo+MIN; self.vS=lo;self.vE=hi; }
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
    self.drag=null; self._stopEdge();
    ov.className='tlw-ov-wrap tlw-cur-default';
    self._draw(); self._syncTableToView();
    /* Click-to-navigate */
    if(wasMode==='N'&&self.rows.length){
      var curPx=self._mPx(e);
      if(Math.abs(curPx-wasPx)<5){
        var gf2=self._px2f(curPx);
        var ts2=self.gMin+gf2*self.gSpan;
        var hits=self.rows.filter(function(r){ return r.ts<=ts2&&r.te>=ts2; });
        if(!hits.length){
          var margin=self.gSpan*0.02;
          hits=self.rows.filter(function(r){ return r.te>=ts2-margin&&r.ts<=ts2+margin; });
          hits.sort(function(a,b){ return Math.abs((a.ts+a.te)/2-ts2)-Math.abs((b.ts+b.te)/2-ts2); });
        }
        if(hits.length) self._focusRow(hits[0]);
      }
    }
  };

  document.addEventListener('mousemove',onMove);
  document.addEventListener('mouseup',  onUp);

  /* Overview scroll zoom — pivots on brush midpoint */
  ov.addEventListener('wheel',function(e){
    e.preventDefault();
    var factor=e.deltaY>0?1.25:0.8;
    self._ovZoom(factor);
  },{passive:false});

  /* Detail scroll pan */
  this._el('dw').addEventListener('wheel',function(e){
    e.preventDefault();
    var w=self.vE-self.vS;
    var delta=Math.abs(e.deltaX)>Math.abs(e.deltaY)?e.deltaX:e.deltaY;
    var step=w*0.10*(delta>0?1:-1);
    self.vS=clamp(self.vS+step,0,1-w); self.vE=self.vS+w;
    self._scrollOvToBrush();
    self._drawOv(); self._drawDet(); self._drawLabels();
    clearTimeout(self._detScrollTimer);
    self._detScrollTimer=setTimeout(function(){ self._syncTableToView(); },200);
  },{passive:false});

  /* Detail click — focus item AND centre overview on it */
  this._el('dw').addEventListener('click',function(e){
    if(!self.rows.length) return;
    var cv=self._el('detc'), rc=cv.getBoundingClientRect();
    var vs=self.gMin+self.vS*self.gSpan, ve=self.gMin+self.vE*self.gSpan;
    var ts=vs+((e.clientX-rc.left)/rc.width)*(ve-vs);
    var hits=self._detailRows().filter(function(r){ return r.ts<=ts&&r.te>=ts; });
    if(!hits.length) return;
    var r=hits[0];
    // Focus in detail and centre overview brush on it
    self._focusRow(r);
    // Centre the overview viewport on the focused item
    var frac=self._gf((r.ts+r.te)/2);
    var ow=self.oE-self.oS;
    self.oS=clamp(frac-ow/2,0,1-ow);
    self.oE=self.oS+ow;
    self._draw();
  });

  /* ── Keyboard ── */
  this._kbZone='none';
  ov.addEventListener('mouseenter',function(){ self._kbZone='ov'; });
  ov.addEventListener('mouseleave',function(){ if(self._kbZone==='ov') self._kbZone='none'; });
  this._el('dw').addEventListener('mouseenter',function(){ self._kbZone='det'; });
  this._el('dw').addEventListener('mouseleave',function(){ if(self._kbZone==='det') self._kbZone='none'; });

  document.addEventListener('keydown',function(e){
    if(self._kbZone==='none') return;
    var key=e.key;

    /* Overview: ↑↓ zoom, ←→ pan brush */
    if(self._kbZone==='ov'){
      if(key==='ArrowUp'||key==='ArrowDown'){
        e.preventDefault();
        self._ovZoom(key==='ArrowUp'?0.7:1.4);
        return;
      }
      if(key==='ArrowLeft'||key==='ArrowRight'){
        e.preventDefault();
        var w=self.vE-self.vS, step=w*0.4*(key==='ArrowRight'?1:-1);
        self.vS=clamp(self.vS+step,0,1-w); self.vE=self.vS+w;
        self._scrollOvToBrush(); self._draw();
        clearTimeout(self._detScrollTimer);
        self._detScrollTimer=setTimeout(function(){ self._syncTableToView(); },200);
        return;
      }
    }

    /* Detail: ←→ step items */
    if(self._kbZone==='det'&&(key==='ArrowLeft'||key==='ArrowRight')){
      e.preventDefault();
      var dir=key==='ArrowRight'?1:-1;
      if(!self.rows.length) return;
      var idx=-1;
      for(var i=0;i<self.rows.length;i++){
        if(self.rows[i].id===String(self.focusId)){ idx=i; break; }
      }
      if(idx<0){
        var vs2=self.gMin+self.vS*self.gSpan, ve2=self.gMin+self.vE*self.gSpan;
        for(var j=0;j<self.rows.length;j++){
          if(self.rows[j].te>vs2&&self.rows[j].ts<ve2){ idx=j; break; }
        }
        if(idx<0) return;
      }
      var next=clamp(idx+dir,0,self.rows.length-1);
      if(next!==idx) self._focusRow(self.rows[next]);
    }
  });

  /* Resize */
  if(window.ResizeObserver){
    new ResizeObserver(function(){ self._draw(); }).observe(document.querySelector(self.ctnId));
  }
};

g.TimelineWidget=TimelineWidget;
}(window));
