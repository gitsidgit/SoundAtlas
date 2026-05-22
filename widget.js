/* ═══════════════════════════════════════════════════════════════════════════
   TimelineWidget  v3.0
   Depends on: widget.css (loaded separately), Tabulator ≥ 6
   ═══════════════════════════════════════════════════════════════════════════ */
(function(g){
'use strict';

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

/* ─── SVG pre-parser ──────────────────────────────────────────────────────
 * Called once per row during _buildRows.  Returns a compact parsed
 * representation so the hot draw path never re-parses strings.
 * Handles: path d= (M L m l C c Q q A a Z z), polyline/polygon points=
 * and nested <path> / <polyline> elements via simple tag scanning.
 * ───────────────────────────────────────────────────────────────────────── */
function parseSVG(svgStr){
  if(!svgStr) return null;
  // Extract viewBox dimensions
  var svgW=400, svgH=80;
  var vbm=svgStr.match(/viewBox=["']\s*([\d.+-]+)[\s,]+([\d.+-]+)[\s,]+([\d.+-]+)[\s,]+([\d.+-]+)/i);
  if(vbm){ svgW=parseFloat(vbm[3])||400; svgH=parseFloat(vbm[4])||80; }

  var shapes=[];

  /* ── Extract all <path d="…"> segments ── */
  var pathRe=/<path[^>]*\bd=["']([^"']+)["'][^>]*\/?>/gi;
  var pm;
  while((pm=pathRe.exec(svgStr))!==null){
    var cmds=parsePathD(pm[1]);
    if(cmds.length) shapes.push({type:'path',cmds:cmds});
  }

  /* ── Extract all <polyline/polygon points="…"> ── */
  var polyRe=/<poly(?:line|gon)[^>]*\bpoints=["']([^"']+)["'][^>]*\/?>/gi;
  var polym;
  while((polym=polyRe.exec(svgStr))!==null){
    var pts=parsePoints(polym[1]);
    if(pts.length) shapes.push({type:'poly',pts:pts});
  }

  return {w:svgW, h:svgH, shapes:shapes};
}

function parsePathD(d){
  var raw=d.match(/[MmLlCcQqAaZz][^MmLlCcQqAaZz]*/g)||[];
  var out=[];
  raw.forEach(function(seg){
    var t=seg[0];
    var nums=(seg.slice(1).trim().match(/-?[\d.eE+-]+/g)||[]).map(Number);
    out.push({t:t,n:nums});
  });
  return out;
}

function parsePoints(str){
  var raw=(str.trim().match(/-?[\d.eE+-]+/g)||[]).map(Number);
  var pts=[];
  for(var i=0;i+1<raw.length;i+=2) pts.push(raw[i],raw[i+1]);
  return pts;
}

/* ─── SVG → Canvas renderer ───────────────────────────────────────────────
 * Uses pre-parsed representation.  Supports full M/L/m/l/C/c/Q/q/Z/z.
 * A/a arcs are approximated as a straight line to avoid heavy math
 * (good enough for waveform display at small sizes).
 * ───────────────────────────────────────────────────────────────────────── */
function renderSVG(ctx, parsed, x, y, w, h){
  if(!parsed||!parsed.shapes.length) return;
  var sx=w/parsed.w, sy=h/parsed.h;
  var curX=x, curY=y, startX=x, startY=y;

  parsed.shapes.forEach(function(shape){
    if(shape.type==='poly'){
      var pts=shape.pts;
      if(!pts.length) return;
      ctx.beginPath();
      ctx.moveTo(x+pts[0]*sx, y+pts[1]*sy);
      for(var i=2;i+1<pts.length;i+=2) ctx.lineTo(x+pts[i]*sx, y+pts[i+1]*sy);
      ctx.stroke();
      return;
    }
    // path
    ctx.beginPath();
    curX=x; curY=y;
    shape.cmds.forEach(function(seg){
      var t=seg.t, n=seg.n, i=0;
      switch(t){
        case 'M':
          while(i+1<n.length){ curX=x+n[i]*sx; curY=y+n[i+1]*sy; ctx.moveTo(curX,curY); startX=curX; startY=curY; i+=2; }
          break;
        case 'm':
          while(i+1<n.length){ curX+=n[i]*sx; curY+=n[i+1]*sy; ctx.moveTo(curX,curY); startX=curX; startY=curY; i+=2; }
          break;
        case 'L':
          while(i+1<n.length){ curX=x+n[i]*sx; curY=y+n[i+1]*sy; ctx.lineTo(curX,curY); i+=2; }
          break;
        case 'l':
          while(i+1<n.length){ curX+=n[i]*sx; curY+=n[i+1]*sy; ctx.lineTo(curX,curY); i+=2; }
          break;
        case 'H': while(i<n.length){ curX=x+n[i]*sx; ctx.lineTo(curX,curY); i++; } break;
        case 'h': while(i<n.length){ curX+=n[i]*sx;  ctx.lineTo(curX,curY); i++; } break;
        case 'V': while(i<n.length){ curY=y+n[i]*sy; ctx.lineTo(curX,curY); i++; } break;
        case 'v': while(i<n.length){ curY+=n[i]*sy;  ctx.lineTo(curX,curY); i++; } break;
        case 'C':
          while(i+5<n.length){
            ctx.bezierCurveTo(x+n[i]*sx,y+n[i+1]*sy, x+n[i+2]*sx,y+n[i+3]*sy, x+n[i+4]*sx,y+n[i+5]*sy);
            curX=x+n[i+4]*sx; curY=y+n[i+5]*sy; i+=6;
          } break;
        case 'c':
          while(i+5<n.length){
            ctx.bezierCurveTo(curX+n[i]*sx,curY+n[i+1]*sy, curX+n[i+2]*sx,curY+n[i+3]*sy, curX+n[i+4]*sx,curY+n[i+5]*sy);
            curX+=n[i+4]*sx; curY+=n[i+5]*sy; i+=6;
          } break;
        case 'Q':
          while(i+3<n.length){
            ctx.quadraticCurveTo(x+n[i]*sx,y+n[i+1]*sy, x+n[i+2]*sx,y+n[i+3]*sy);
            curX=x+n[i+2]*sx; curY=y+n[i+3]*sy; i+=4;
          } break;
        case 'q':
          while(i+3<n.length){
            ctx.quadraticCurveTo(curX+n[i]*sx,curY+n[i+1]*sy, curX+n[i+2]*sx,curY+n[i+3]*sy);
            curX+=n[i+2]*sx; curY+=n[i+3]*sy; i+=4;
          } break;
        case 'A': case 'a':
          // Approximate arc as line to endpoint
          if(i+6<n.length){
            if(t==='A'){ curX=x+n[i+5]*sx; curY=y+n[i+6]*sy; }
            else        { curX+=n[i+5]*sx;  curY+=n[i+6]*sy;  }
            ctx.lineTo(curX,curY);
          } break;
        case 'Z': case 'z':
          ctx.closePath(); curX=startX; curY=startY; break;
      }
    });
    ctx.stroke();
  });
}

/* ─── Constructor ─────────────────────────────────────────────────────────── */
function TimelineWidget(cfg){
  this.tableId   = cfg.tableId;
  this.ctnId     = cfg.containerId;
  this.f_start   = cfg.timeStart  || 'timestamp';
  this.f_end     = cfg.timeEnd    || 'end_time';
  this.f_wave    = cfg.waveform   || null;
  this.f_id      = cfg.idField    || 'id';
  // Optional whitelist of raw-data fields to keep (saves memory on large datasets).
  // Always includes f_id, f_start, f_end, f_wave automatically.
  this.f_keep    = cfg.keepFields || null;
  this.lazyWave  = !!cfg.lazyWave;
  this.autoSync  = true;   // sync table page to detail view

  this.rows  = [];
  this.gMin=0; this.gMax=1; this.gSpan=1;
  this.vS=0; this.vE=1;    // detail window, 0-1 fractions of global span
  this.oS=0; this.oE=1;    // overview viewport, 0-1 fractions

  this.focusId  = null;
  this.ovOpen   = true;
  this.detOpen  = true;
  this.scope    = 'all';

  this._suppressPageSync   = false;
  this._syncTimer          = null;   // debounce for table sync
  this._drawRAF            = null;   // rAF handle for throttled draws
  this._lastHighlightedRow = null;
  this._benchSamples       = null;

  // Pointer interaction
  this.drag    = null;
  this.edgeRAF = null;

  // Layout / tuning constants
  this.HPIX        = 14;      // brush handle hit px radius
  this.OV_PAD      = 18;      // px each side for brush handles
  this.DET_H       = 96;      // detail canvas height px
  this.OV_H        = 62;      // overview canvas height px
  this.OV_BAR_TOP  = 18;      // y-top of bar strip
  this.OV_BAR_H    = 18;      // bar strip height
  this.OV_RULER    = 42;      // y of tick ruler baseline
  this.MERGE_PX    = 4;       // coalesce threshold px
  this.MIN_WAVE_PX = 40;      // min px width to render waveform
  this.MIN_BRUSH   = 0.001;
  this.EDGE        = 0.04;    // auto-pan activation zone
  this.ESPD        = 0.004;   // auto-pan speed per frame

  this._kbZone = 'none';

  // Bound listener refs (so we can remove them on destroy)
  this._onDocMove = null;
  this._onDocUp   = null;
  this._onDocKey  = null;
  this._resizeObs = null;

  this._buildDOM();
  this._bindDOM();
  this._waitForTable();
}

/* ─── DOM ─────────────────────────────────────────────────────────────────── */
TimelineWidget.prototype._buildDOM=function(){
  var uid='tlw'+Math.random().toString(36).slice(2,7);
  this.uid=uid;
  var OVH=this.OV_H, DTH=this.DET_H;

  var H=[
  '<div class="tlw" id="'+uid+'">',

  /* ── Overview section header ── */
  '<div class="tlw-ov-header">',
  '  <button class="tlw-toggle-btn" id="'+uid+'_ovtog" title="Collapse overview">',
  '    <span class="tlw-sec">Overview</span>',
  '    <span class="tlw-caret" id="'+uid+'_ovcaret">&#9660;</span>',
  '  </button>',
  '  <span class="tlw-hint">drag brush · ↑↓ zoom · ← → pan · click item</span>',
  '</div>',

  /* ── Overview collapsible body ── */
  '<div class="tlw-collapsible" id="'+uid+'_ovbody" style="max-height:'+OVH+'px">',
  '  <div class="tlw-ov-wrap tlw-cur-default" id="'+uid+'_ov">',
  '    <canvas id="'+uid+'_ovc" height="'+OVH+'"></canvas>',
  '    <div class="tlw-brush" id="'+uid+'_br">',
  '      <div class="tlw-tip" id="'+uid+'_tip"></div>',
  '    </div>',
  '  </div>',
  '</div>',

  /* ── Toolbar ── */
  '<div class="tlw-toolbar" id="'+uid+'_tb">',
  '  <span class="tlw-sec" style="white-space:nowrap">Zoom:</span>',
  '  <div class="btn-group btn-group-xs">',
  '    <button class="btn btn-default" id="'+uid+'_zin"  title="Zoom overview in" >+</button>',
  '    <button class="btn btn-default" id="'+uid+'_zout" title="Zoom overview out">&minus;</button>',
  '    <button class="btn btn-default" id="'+uid+'_zall" title="Full overview"    >All</button>',
  '  </div>',
  '  <span id="'+uid+'_zlbl" style="font-size:11px;font-weight:700;min-width:28px">1x</span>',
  '  <input type="range" id="'+uid+'_zslide" min="1" max="200" value="1" style="width:80px">',
  '  <span class="tlw-sep">|</span>',
  '  <div class="btn-group btn-group-xs">',
  '    <button class="btn btn-default" id="'+uid+'_v2"  title="Detail 2x" >2x</button>',
  '    <button class="btn btn-default" id="'+uid+'_v5"  title="Detail 5x" >5x</button>',
  '    <button class="btn btn-default" id="'+uid+'_v10" title="Detail 10x">10x</button>',
  '    <button class="btn btn-default" id="'+uid+'_vr"  title="Reset detail">&#8635;</button>',
  '  </div>',
  '  <span class="tlw-sep">|</span>',
  '  <div class="btn-group btn-group-xs">',
  '    <button class="btn btn-default tlw-scope-on" id="'+uid+'_sAll" >All</button>',
  '    <button class="btn btn-default"              id="'+uid+'_sPage">Page</button>',
  '  </div>',
  '  <label><input type="checkbox" id="'+uid+'_lazy"  '+(this.lazyWave?'checked':'')+'>&#8201;Lazy</label>',
  '  <label><input type="checkbox" id="'+uid+'_async" checked>&#8201;Auto&#8209;sync</label>',
  '  <span class="tlw-sep">|</span>',
  '  <button class="btn btn-default btn-xs" id="'+uid+'_bench" title="Render benchmark">&#9654;</button>',
  '  <div class="tlw-stats">',
  '    <span><strong>From</strong> <span id="'+uid+'_sF">—</span></span>',
  '    <span><strong>To</strong>   <span id="'+uid+'_sT">—</span></span>',
  '    <span><strong>Span</strong> <span id="'+uid+'_sS">—</span></span>',
  '    <span><strong>Items</strong><span id="'+uid+'_sN">—</span></span>',
  '  </div>',
  '</div>',

  '<div id="'+uid+'_benchout" style="display:none;margin-bottom:6px;font-size:11px;background:#fff;border:1px solid #ddd;border-radius:4px;padding:6px 10px"></div>',

  /* ── Detail section ── */
  '<div class="tlw-det-header">',
  '  <button class="tlw-toggle-btn" id="'+uid+'_dtog" title="Collapse detail">',
  '    <span class="tlw-sec">Detail</span>',
  '    <span class="tlw-caret" id="'+uid+'_dcaret">&#9660;</span>',
  '  </button>',
  '  <span class="tlw-hint">click to focus · scroll pan · ← → step</span>',
  '</div>',
  '<div class="tlw-collapsible" id="'+uid+'_dbody" style="max-height:'+(DTH+24)+'px">',
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

/* ─── Destroy ─────────────────────────────────────────────────────────────── */
TimelineWidget.prototype.destroy=function(){
  // Remove document-level listeners
  if(this._onDocMove) document.removeEventListener('mousemove',this._onDocMove);
  if(this._onDocUp)   document.removeEventListener('mouseup',  this._onDocUp);
  if(this._onDocKey)  document.removeEventListener('keydown',  this._onDocKey);
  // Stop any running animation frames
  this._stopEdge();
  if(this._drawRAF){ cancelAnimationFrame(this._drawRAF); this._drawRAF=null; }
  clearTimeout(this._syncTimer);
  // Disconnect ResizeObserver
  if(this._resizeObs){ this._resizeObs.disconnect(); this._resizeObs=null; }
  // Detach Tabulator events
  if(this.tbl){
    try{ this.tbl.off('dataLoaded');   }catch(e){}
    try{ this.tbl.off('dataFiltered'); }catch(e){}
    try{ this.tbl.off('pageLoaded');   }catch(e){}
    try{ this.tbl.off('rowClick');     }catch(e){}
  }
  // Null out large references
  this.rows=[]; this.tbl=null;
  var ctn=document.querySelector(this.ctnId);
  if(ctn) ctn.innerHTML='';
};

/* ─── Table binding ───────────────────────────────────────────────────────── */
TimelineWidget.prototype._waitForTable=function(attempt){
  var self=this; attempt=attempt||0;
  if(attempt>100){
    console.error('TimelineWidget: table not found "'+this.tableId+'"'); return;
  }
  var tables=Tabulator.findTable(this.tableId);
  if(!tables||!tables.length){ setTimeout(function(){ self._waitForTable(attempt+1); },100); return; }
  this.tbl=tables[0];
  var tbl=this.tbl;
  tbl.on('dataLoaded',   function(){ setTimeout(function(){ self._ingest(false); },80); });
  tbl.on('dataFiltered', function(){ setTimeout(function(){ self._ingest(true);  },80); });
  tbl.on('pageLoaded',   function(){ if(self._suppressPageSync) return; self._onPageChange(); });
  tbl.on('rowClick',     function(e,row){ self._onRowClick(row); });
  setTimeout(function(){ self._ingest(false); },120);
};

/* ─── Data ingestion ──────────────────────────────────────────────────────── */
TimelineWidget.prototype._ingest=function(isFilter){
  if(!this.tbl) return;
  var raw=isFilter ? this.tbl.getData('active') : this.tbl.getData();
  if(!raw||!raw.length){ this.rows=[]; this._draw(); return; }

  var oldGMin=this.gMin, oldGSpan=this.gSpan;
  var oldVS=this.vS, oldVE=this.vE;

  this._buildRows(raw);

  if(isFilter && oldGSpan>0 && this.gSpan>0){
    // Preserve viewport in absolute time, re-express in new fractions
    var vsMs=oldGMin+oldVS*oldGSpan, veMs=oldGMin+oldVE*oldGSpan;
    this.vS=clamp((vsMs-this.gMin)/this.gSpan,0,1);
    this.vE=clamp((veMs-this.gMin)/this.gSpan,0,1);
    if(this.vE-this.vS<this.MIN_BRUSH) this.vE=Math.min(1,this.vS+this.MIN_BRUSH);
    this.oS=0; this.oE=1;
  } else {
    this.vS=0; this.vE=1; this.oS=0; this.oE=1;
  }
  this._draw();
};

TimelineWidget.prototype._buildRows=function(raw){
  var self=this;
  var keep=this.f_keep;
  var fs=this.f_start, fe=this.f_end, fw=this.f_wave, fi=this.f_id;

  // Build a field whitelist for raw data trimming
  var keepSet=null;
  if(keep&&keep.length){
    keepSet={};
    keep.forEach(function(k){ keepSet[k]=1; });
    keepSet[fi]=1; keepSet[fs]=1; keepSet[fe]=1;
    if(fw) keepSet[fw]=1;
  }

  this.rows=raw.map(function(r){
    var ts=toMs(r[fs]), te=toMs(r[fe]);
    if(isNaN(ts)||isNaN(te)||te<ts) return null;

    // Trim raw object to only the fields we need
    var trimmed;
    if(keepSet){
      trimmed={};
      var keys=Object.keys(r);
      for(var k=0;k<keys.length;k++){
        if(keepSet[keys[k]]) trimmed[keys[k]]=r[keys[k]];
      }
    } else {
      trimmed=r;
    }

    return {
      id : String(r[fi]),
      ts : ts,
      te : te,
      dur: te-ts,
      svg: fw ? parseSVG(r[fw]||null) : null,  // pre-parsed once
      raw: trimmed
    };
  }).filter(Boolean)
    .sort(function(a,b){ return a.ts-b.ts; });

  if(!this.rows.length) return;
  this.gMin  = this.rows[0].ts;
  this.gMax  = this.rows.reduce(function(m,r){ return r.te>m?r.te:m; }, this.rows[0].te);
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
  return clamp(e.clientX-this._el('ov').getBoundingClientRect().left, 0, this._ovW());
};
TimelineWidget.prototype._gf=function(ts){ return clamp((ts-this.gMin)/this.gSpan,0,1); };

/* ─── View helpers ────────────────────────────────────────────────────────── */
TimelineWidget.prototype._scrollOvToBrush=function(){
  var mid=(this.vS+this.vE)/2, ow=this.oE-this.oS;
  if(mid<this.oS||mid>this.oE){
    this.oS=clamp(mid-ow/2,0,1-ow); this.oE=this.oS+ow;
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
  // Centre overview on item
  var frac=this._gf((r.ts+r.te)/2);
  var ow=this.oE-this.oS;
  this.oS=clamp(frac-ow/2,0,1-ow); this.oE=this.oS+ow;
  if(!this.detOpen) this._toggleSection('det');
  this._draw();
  this._highlightRow(r.id);
  if(this.autoSync) this._scheduleSyncTable();
};

/* ─── Table sync ──────────────────────────────────────────────────────────── */
TimelineWidget.prototype._scheduleSyncTable=function(){
  if(!this.autoSync) return;
  clearTimeout(this._syncTimer);
  var self=this;
  this._syncTimer=setTimeout(function(){ self._syncTableToView(); },250);
};
TimelineWidget.prototype._getPageRows=function(){
  if(!this.tbl) return [];
  try{
    var active=this.tbl.getData('active');
    var ps=this.tbl.getPageSize()||25, pg=Math.max(1,this.tbl.getPage()||1);
    var f=this.f_id, ids={};
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
  if(!this.tbl||!this.autoSync) return;
  var self=this;
  var vs=this.gMin+this.vS*this.gSpan, ve=this.gMin+this.vE*this.gSpan;
  // Find first row in view that exists in the active (filtered) dataset
  var active=this.tbl.getData('active');
  var f=this.f_id, activeIds={};
  active.forEach(function(r){ activeIds[String(r[f])]=1; });
  var hit=this.rows.filter(function(r){ return r.te>vs&&r.ts<ve&&activeIds[r.id]; });
  if(!hit.length) return;

  // If focusId is in view, prefer it for page targeting
  var targetId=hit[0].id;
  if(this.focusId){
    for(var i=0;i<hit.length;i++){ if(hit[i].id===String(this.focusId)){ targetId=hit[i].id; break; } }
  }

  try{
    var ps=this.tbl.getPageSize()||25, idx=-1;
    for(var j=0;j<active.length;j++){ if(String(active[j][f])===targetId){ idx=j; break; } }
    if(idx<0) return;
    var tp=Math.floor(idx/ps)+1, cp=this.tbl.getPage();
    if(tp===cp){
      // Already on right page — scroll focused row to top of table
      self._scrollRowIntoView(targetId);
      return;
    }
    this._suppressPageSync=true;
    this.tbl.setPage(tp).then(function(){
      self._suppressPageSync=false;
      self._scrollRowIntoView(targetId);
    });
  }catch(e){ this._suppressPageSync=false; }
};

TimelineWidget.prototype._scrollRowIntoView=function(id){
  // Scroll the highlighted row to top of Tabulator's scroll area
  if(!this.tbl) return;
  try{
    var f=this.f_id, self=this;
    var rows=this.tbl.getRows();
    for(var i=0;i<rows.length;i++){
      if(String(rows[i].getData()[f])===id){
        rows[i].scrollTo('top',true);
        break;
      }
    }
  }catch(e){}
};

TimelineWidget.prototype._onRowClick=function(row){
  var id=String(row.getData()[this.f_id]);
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

/* ─── Overview draw ───────────────────────────────────────────────────────── */
TimelineWidget.prototype._drawOv=function(){
  if(!this.ovOpen) return;
  var cv=this._el('ovc');
  var W=this._ovW(); cv.width=W; cv.height=this.OV_H;
  var ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,this.OV_H);
  if(!this.rows.length) return;

  var self=this, pad=this.OV_PAD, OVH=this.OV_H;
  var BAR_TOP=this.OV_BAR_TOP, BAR_H=this.OV_BAR_H;
  var RULER_Y=this.OV_RULER, MERGE=this.MERGE_PX;
  var os=this.gMin+this.oS*this.gSpan, oe=this.gMin+this.oE*this.gSpan;

  // ── Anchor labels at top ──
  ctx.save();
  ctx.font='bold 10px sans-serif'; ctx.fillStyle='#222'; ctx.textBaseline='top';
  ctx.textAlign='left';   ctx.fillText(fmtD(os),       pad,   2);
  ctx.textAlign='center'; ctx.fillText(fmtD((os+oe)/2),W/2,   2);
  ctx.textAlign='right';  ctx.fillText(fmtD(oe),       W-pad, 2);
  ctx.restore();

  // ── Coalescing bar intervals ──
  var intervals=[], focId=String(this.focusId);
  this.rows.forEach(function(r){
    var f1=self._gf(r.ts), f2=self._gf(r.te);
    if(f2<self.oS||f1>self.oE) return;
    var x1=clamp(Math.round(self._f2px(f1)),pad,W-pad);
    var x2=clamp(Math.round(self._f2px(f2)),pad,W-pad);
    if(x2<x1+1) x2=x1+1;
    intervals.push({x1:x1,x2:x2,id:r.id});
  });
  // Already sorted by ts (rows are sorted), so intervals are nearly sorted
  // A single insertion-sort pass is cheaper than Array.sort for nearly-sorted data
  for(var ii=1;ii<intervals.length;ii++){
    var iv=intervals[ii], jj=ii-1;
    while(jj>=0&&intervals[jj].x1>iv.x1){ intervals[jj+1]=intervals[jj]; jj--; }
    intervals[jj+1]=iv;
  }
  var merged=[];
  for(var mi=0;mi<intervals.length;mi++){
    var cur=intervals[mi];
    if(!merged.length){ merged.push({x1:cur.x1,x2:cur.x2,ids:[cur.id]}); continue; }
    var last=merged[merged.length-1];
    if(cur.x1-last.x2<=MERGE){ last.x2=Math.max(last.x2,cur.x2); last.ids.push(cur.id); }
    else merged.push({x1:cur.x1,x2:cur.x2,ids:[cur.id]});
  }

  var BAR_CLR=this._cssVar('--tlw-bar','#6baed6');
  var BAR_FOC=this._cssVar('--tlw-bar-focus','#2171b5');
  for(var bi=0;bi<merged.length;bi++){
    var m=merged[bi];
    var isFocus=m.ids.indexOf(focId)>=0, solo=m.ids.length===1;
    var bw=m.x2-m.x1;
    ctx.fillStyle=BAR_CLR+(isFocus?'ff':solo?'cc':'99');
    ctx.fillRect(m.x1,BAR_TOP,bw,BAR_H);
    if(solo){ ctx.fillStyle=BAR_CLR+'ff'; ctx.fillRect(m.x1,BAR_TOP,bw,2); }
    if(isFocus){
      ctx.strokeStyle=BAR_FOC; ctx.lineWidth=1.5;
      ctx.strokeRect(m.x1+0.5,BAR_TOP+0.5,bw-1,BAR_H-1);
    }
  }

  // Median line
  var MID_Y=Math.round(BAR_TOP+BAR_H/2);
  ctx.strokeStyle='rgba(107,174,214,0.35)'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,MID_Y); ctx.lineTo(W-pad,MID_Y); ctx.stroke();

  // ── Tick ruler ──
  this._drawTicks(ctx,W,pad,RULER_Y,os,oe);

  // ── Time brush overlay ──
  var bx1=clamp(Math.round(this._f2px(this.vS)),0,W);
  var bx2=clamp(Math.round(this._f2px(this.vE)),0,W);
  ctx.fillStyle='rgba(51,122,183,0.10)';
  ctx.fillRect(bx1,BAR_TOP-1,bx2-bx1,OVH-BAR_TOP+1);
  ctx.strokeStyle='#337ab7'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(bx1,BAR_TOP-1); ctx.lineTo(bx1,OVH); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(bx2,BAR_TOP-1); ctx.lineTo(bx2,OVH); ctx.stroke();
  this._drawHandle(ctx,bx1,BAR_TOP+BAR_H/2);
  this._drawHandle(ctx,bx2,BAR_TOP+BAR_H/2);

  // DOM brush div
  var br=this._el('br');
  br.style.left=bx1+'px'; br.style.width=Math.max(4,bx2-bx1)+'px';
  br.style.top='0'; br.style.height=OVH+'px';

  // Tooltip
  var vs=this.gMin+this.vS*this.gSpan, ve=this.gMin+this.vE*this.gSpan;
  this._el('tip').textContent=fmtD(vs)+'\u2009\u2192\u2009'+fmtD(ve)+'\u2009('+fmtS(ve-vs)+')';
};

// Helper to read a CSS custom property from the widget's own element
TimelineWidget.prototype._cssVar=function(name,fallback){
  var el=document.getElementById(this.uid);
  if(!el) return fallback;
  var v=getComputedStyle(el).getPropertyValue(name).trim();
  return v||fallback;
};

/* ─── Tick ruler ──────────────────────────────────────────────────────────── */
TimelineWidget.prototype._drawTicks=function(ctx,W,pad,RULER_Y,os,oe){
  var span=oe-os||1;
  var NICE=[100,200,500,1000,2000,5000,10000,15000,30000,
    60000,120000,300000,600000,1800000,3600000,7200000,
    21600000,43200000,86400000,172800000,604800000];
  var interval=NICE[NICE.length-1];
  for(var ni=0;ni<NICE.length;ni++){ if(span/NICE[ni]<=7){ interval=NICE[ni]; break; } }
  var minor=interval/5;
  var MAJ_H=5, LBL_Y=RULER_Y-MAJ_H-12;
  var useW=W-pad*2;

  ctx.save();
  ctx.font='bold 10px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='top';

  // Baseline
  ctx.strokeStyle='rgba(80,80,80,0.5)'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,RULER_Y); ctx.lineTo(W-pad,RULER_Y); ctx.stroke();

  // Minor ticks — batch into one path
  ctx.strokeStyle='rgba(120,120,120,0.5)'; ctx.lineWidth=1;
  ctx.beginPath();
  var t0m=Math.ceil(os/minor)*minor;
  for(var tm=t0m;tm<=oe;tm+=minor){
    var xm=pad+(tm-os)/span*useW;
    if(xm<pad||xm>W-pad) continue;
    ctx.moveTo(xm,RULER_Y); ctx.lineTo(xm,RULER_Y+3);
  }
  ctx.stroke();

  // Major ticks + labels — batch ticks into one path
  ctx.strokeStyle='rgba(40,40,40,0.8)'; ctx.lineWidth=1.5;
  ctx.beginPath();
  var t0=Math.ceil(os/interval)*interval;
  var majXs=[];
  for(var t=t0;t<=oe;t+=interval){
    var x=pad+(t-os)/span*useW;
    if(x<pad||x>W-pad) continue;
    ctx.moveTo(x,RULER_Y-MAJ_H); ctx.lineTo(x,RULER_Y+2);
    majXs.push({x:x,t:t});
  }
  ctx.stroke();

  // Labels (separate loop so we set fillStyle once)
  ctx.fillStyle='#222';
  for(var li=0;li<majXs.length;li++){
    var lx=majXs[li].x;
    if(lx>pad+22&&lx<W-pad-22) ctx.fillText(this._tickLabel(majXs[li].t,interval),lx,LBL_Y);
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
  var vs=this.gMin+this.vS*this.gSpan, ve=this.gMin+this.vE*this.gSpan;
  var sp=ve-vs||1, self=this;
  var focId=String(this.focusId);
  var MWPX=this.MIN_WAVE_PX, PAD_Y=4;
  var itemH=DTH-PAD_Y*2;
  var BAR_CLR=this._cssVar('--tlw-bar','#6baed6');
  var BAR_FOC=this._cssVar('--tlw-bar-focus','#2171b5');

  // Grid
  ctx.strokeStyle=this._cssVar('--tlw-grid','#ececec'); ctx.lineWidth=1;
  ctx.beginPath();
  [1,2,3].forEach(function(i){ ctx.moveTo(W/4*i,0); ctx.lineTo(W/4*i,DTH); });
  ctx.stroke();

  var rows=this._detailRows();
  rows.forEach(function(r){
    var x1=(r.ts-vs)/sp*W, x2=(r.te-vs)/sp*W, bw=x2-x1;
    if(bw<1) return;

    var isFocus=(r.id===focId);
    var clipX1=Math.max(0,x1), clipW=Math.min(W,x2)-clipX1;
    if(clipW<1) return;

    ctx.save();
    ctx.beginPath(); ctx.rect(clipX1,PAD_Y,clipW,itemH); ctx.clip();

    ctx.fillStyle=BAR_CLR+(isFocus?'28':'18');
    ctx.fillRect(x1,PAD_Y,bw,itemH);

    if(x1>=0&&x1<W){
      ctx.fillStyle=BAR_CLR+(isFocus?'ff':'cc');
      ctx.fillRect(x1,PAD_Y,Math.min(2,bw),itemH);
    }

    if(r.svg&&bw>=MWPX){
      ctx.strokeStyle=BAR_CLR+'cc'; ctx.lineWidth=1.3;
      renderSVG(ctx,r.svg,x1,PAD_Y,bw,itemH);
      waveCount++;
    }

    ctx.restore();

    if(isFocus){
      ctx.strokeStyle=BAR_FOC; ctx.lineWidth=2;
      ctx.strokeRect(clipX1+1,PAD_Y+1,clipW-2,itemH-2);
    }
    if(x2>0&&x2<W&&bw>4){
      ctx.strokeStyle='rgba(255,255,255,0.7)'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(x2,PAD_Y); ctx.lineTo(x2,PAD_Y+itemH); ctx.stroke();
    }
  });

  if(this._benchSamples) this._benchSamples.push({ms:performance.now()-t0,waves:waveCount});
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

/* ─── Master draw — throttled to one rAF per call site ───────────────────── */
TimelineWidget.prototype._draw=function(){
  if(this._drawRAF) return; // already queued
  var self=this;
  this._drawRAF=requestAnimationFrame(function(){
    self._drawRAF=null;
    self._drawOv();
    self._drawDet();
    self._drawLabels();
  });
};
// Synchronous draw used during drag (we need immediate feedback)
TimelineWidget.prototype._drawSync=function(){
  this._drawOv(); this._drawDet(); this._drawLabels();
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
    var rows=this.tbl.getRows();
    for(var i=0;i<rows.length;i++){
      if(String(rows[i].getData()[f])===id){
        rows[i].getElement().classList.add('tlw-row-hi');
        this._lastHighlightedRow=rows[i];
        break;
      }
    }
  }catch(e){}
};

/* ─── UI state ────────────────────────────────────────────────────────────── */
TimelineWidget.prototype._toggleSection=function(which){
  if(which==='ov'){
    this.ovOpen=!this.ovOpen;
    this._el('ovbody').style.maxHeight=this.ovOpen?this.OV_H+'px':'0';
    this._el('ovcaret').innerHTML=this.ovOpen?'&#9660;':'&#9654;';
    // Show/hide toolbar only when overview is also hidden
    this._el('tb').style.display=this.ovOpen?'':'none';
  } else {
    this.detOpen=!this.detOpen;
    this._el('dbody').style.maxHeight=this.detOpen?(this.DET_H+24)+'px':'0';
    this._el('dcaret').innerHTML=this.detOpen?'&#9660;':'&#9654;';
  }
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
  var pivot=(this.vS+this.vE)/2;
  var ow=this.oE-this.oS, nw=clamp(ow*f,0.005,1);
  var ratio=(pivot-this.oS)/ow;
  this.oS=clamp(pivot-ratio*nw,0,1-nw); this.oE=this.oS+nw;
  this._draw();
};
TimelineWidget.prototype._ovFull=function(){ this.oS=0;this.oE=1;this._draw(); };
TimelineWidget.prototype._ovSlide=function(v){
  var w=clamp(1/Math.max(1,parseFloat(v)),0.005,1);
  var c=(this.oS+this.oE)/2;
  this.oS=clamp(c-w/2,0,1-w); this.oE=this.oS+w; this._draw();
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
    self.oS=clamp(self.oS+dir*self.ESPD,0,1-w); self.oE=self.oS+w;
    self._drawOv(); self._drawLabels();
    self.edgeRAF=requestAnimationFrame(step);
  })();
};

/* ─── Benchmark ───────────────────────────────────────────────────────────── */
TimelineWidget.prototype._runBenchmark=function(){
  var self=this, btn=this._el('bench'), out=this._el('benchout');
  btn.disabled=true; out.style.display='block'; out.innerHTML='<em>Running…</em>';
  var STEPS=30, origLazy=this.lazyWave, origVS=this.vS, origVE=this.vE, BW=this.vE-this.vS;
  function runPass(lazy,done){
    self.lazyWave=lazy; self._benchSamples=[];
    var step=0;
    (function frame(){
      var pos=step/STEPS;
      self.vS=clamp(pos*(1-BW),0,1-BW); self.vE=self.vS+BW;
      self._drawDet(); step++;
      if(step<=STEPS) requestAnimationFrame(frame); else done(self._benchSamples.slice());
    })();
  }
  function stats(s){
    if(!s.length) return {med:'0',min:'0',max:'0',waves:0};
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
  runPass(false,function(o){
    runPass(true,function(n){
      self.lazyWave=origLazy; self.vS=origVS; self.vE=origVE;
      self._draw(); self._benchSamples=null; btn.disabled=false;
      var off=stats(o), on=stats(n), mx=Math.max(parseFloat(off.med),parseFloat(on.med),1)*1.3;
      var spd=(parseFloat(off.med)/Math.max(parseFloat(on.med),0.01)).toFixed(1);
      out.innerHTML='<strong>Draw benchmark</strong>'
        +'<table style="width:100%;margin-top:4px;font-size:11px;border-collapse:collapse">'
        +'<tr><th></th><th>Median</th><th>Min</th><th>Max</th><th>Waves/frame</th></tr>'
        +'<tr><td><strong>Lazy OFF</strong></td><td>'+off.med+'ms</td><td>'+off.min+'ms</td><td>'+off.max+'ms</td><td>'+off.waves+'</td></tr>'
        +'<tr><td colspan="5">'+bar(off.med,mx)+'</td></tr>'
        +'<tr><td><strong>Lazy ON </strong></td><td>'+on.med+'ms</td><td>'+on.min+'ms</td><td>'+on.max+'ms</td><td>'+on.waves+'</td></tr>'
        +'<tr><td colspan="5">'+bar(on.med,mx)+'</td></tr>'
        +'</table>'
        +'<em>Lazy is '+spd+'x faster</em>';
    });
  });
};

/* ─── Event binding ───────────────────────────────────────────────────────── */
TimelineWidget.prototype._bindDOM=function(){
  var self=this;
  var ov=this._el('ov');

  this._el('ovtog').addEventListener('click',function(){ self._toggleSection('ov');  });
  this._el('dtog') .addEventListener('click',function(){ self._toggleSection('det'); });
  this._el('v2')   .addEventListener('click',function(){ self._viewZoom(2);  });
  this._el('v5')   .addEventListener('click',function(){ self._viewZoom(5);  });
  this._el('v10')  .addEventListener('click',function(){ self._viewZoom(10); });
  this._el('vr')   .addEventListener('click',function(){ self.vS=0;self.vE=1;self._scrollOvToBrush();self._draw(); });
  this._el('sAll') .addEventListener('click',function(){ self._setScope('all');  });
  this._el('sPage').addEventListener('click',function(){ self._setScope('page'); });
  this._el('bench').addEventListener('click',function(){ self._runBenchmark(); });
  this._el('zin')  .addEventListener('click',function(){ self._ovZoom(0.5); });
  this._el('zout') .addEventListener('click',function(){ self._ovZoom(2);   });
  this._el('zall') .addEventListener('click',function(){ self._ovFull();    });
  this._el('zslide').addEventListener('input',function(){ self._ovSlide(this.value); });
  this._el('lazy') .addEventListener('change',function(){ self.lazyWave=this.checked; self._draw(); });
  this._el('async').addEventListener('change',function(){
    self.autoSync=this.checked;
    if(self.autoSync) self._scheduleSyncTable();
  });

  /* ── Overview mouse ── */
  ov.addEventListener('mousemove',function(e){
    if(self.drag) return;
    var px=self._mPx(e), gf=self._px2f(px);
    var bx1=self._f2px(self.vS), bx2=self._f2px(self.vE);
    if(Math.abs(px-bx1)<self.HPIX||Math.abs(px-bx2)<self.HPIX) ov.className='tlw-ov-wrap tlw-cur-ew';
    else if(gf>self.vS&&gf<self.vE)                            ov.className='tlw-ov-wrap tlw-cur-grab';
    else                                                        ov.className='tlw-ov-wrap tlw-cur-pointer';
  });
  ov.addEventListener('mouseleave',function(){ if(!self.drag) ov.className='tlw-ov-wrap tlw-cur-default'; });

  ov.addEventListener('mousedown',function(e){
    var px=self._mPx(e), gf=self._px2f(px);
    var bx1=self._f2px(self.vS), bx2=self._f2px(self.vE), mode;
    if(Math.abs(px-bx1)<self.HPIX)      mode='L';
    else if(Math.abs(px-bx2)<self.HPIX) mode='R';
    else if(gf>self.vS&&gf<self.vE)     mode='M';
    else                                 mode='N';
    self.drag={mode:mode,anchor:gf,anchorPx:px,snapS:self.vS,snapE:self.vE};
    ov.className='tlw-ov-wrap '+((mode==='L'||mode==='R')?'tlw-cur-ew':mode==='M'?'tlw-cur-grabbing':'tlw-cur-crosshair');
    e.preventDefault();
  });

  this._onDocMove=function(e){
    if(!self.drag) return;
    var px=self._mPx(e), gf=self._px2f(px);
    var d=self.drag, w=d.snapE-d.snapS, MIN=self.MIN_BRUSH;
    if(d.mode==='L')       self.vS=clamp(gf,0,self.vE-MIN);
    else if(d.mode==='R')  self.vE=clamp(gf,self.vS+MIN,1);
    else if(d.mode==='M'){ var delta=gf-d.anchor; self.vS=clamp(d.snapS+delta,0,1-w); self.vE=self.vS+w; }
    else if(d.mode==='N'){ var lo=clamp(Math.min(d.anchor,gf),0,1),hi=clamp(Math.max(d.anchor,gf),0,1); if(hi-lo<MIN)hi=lo+MIN; self.vS=lo;self.vE=hi; }
    var raw=px/self._ovW();
    if(raw<self.EDGE&&self.oS>0) self._startEdge(-1);
    else if(raw>(1-self.EDGE)&&self.oE<1) self._startEdge(1);
    else self._stopEdge();
    // During drag: draw overview + labels synchronously (low cost), defer detail
    self._drawOv(); self._drawLabels();
    if(!self.lazyWave) self._drawDet();
  };

  this._onDocUp=function(e){
    if(!self.drag) return;
    var wasMode=self.drag.mode, wasPx=self.drag.anchorPx;
    self.drag=null; self._stopEdge();
    ov.className='tlw-ov-wrap tlw-cur-default';
    self._drawSync();
    if(self.autoSync) self._scheduleSyncTable();

    if(wasMode==='N'&&self.rows.length){
      var curPx=self._mPx(e);
      if(Math.abs(curPx-wasPx)<5){
        var ts2=self.gMin+self._px2f(curPx)*self.gSpan;
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

  document.addEventListener('mousemove',this._onDocMove);
  document.addEventListener('mouseup',  this._onDocUp);

  // Overview wheel — zoom pivoted on brush midpoint
  ov.addEventListener('wheel',function(e){
    e.preventDefault();
    self._ovZoom(e.deltaY>0?1.25:0.8);
  },{passive:false});

  // Detail wheel — pan
  this._el('dw').addEventListener('wheel',function(e){
    e.preventDefault();
    var w=self.vE-self.vS;
    var delta=Math.abs(e.deltaX)>Math.abs(e.deltaY)?e.deltaX:e.deltaY;
    var step=w*0.10*(delta>0?1:-1);
    self.vS=clamp(self.vS+step,0,1-w); self.vE=self.vS+w;
    self._scrollOvToBrush();
    self._drawOv(); self._drawDet(); self._drawLabels();
    clearTimeout(self._syncTimer);
    if(self.autoSync) self._syncTimer=setTimeout(function(){ self._syncTableToView(); },250);
  },{passive:false});

  // Detail click — focus, centre overview, sync table (scroll row to top)
  this._el('dw').addEventListener('click',function(e){
    if(!self.rows.length) return;
    var cv=self._el('detc'), rc=cv.getBoundingClientRect();
    var vs=self.gMin+self.vS*self.gSpan, ve=self.gMin+self.vE*self.gSpan;
    var ts=vs+((e.clientX-rc.left)/rc.width)*(ve-vs);
    var hits=self._detailRows().filter(function(r){ return r.ts<=ts&&r.te>=ts; });
    if(!hits.length) return;
    self._focusRow(hits[0]);
  });

  // Keyboard
  this._kbZone='none';
  ov.addEventListener('mouseenter',function(){ self._kbZone='ov';  });
  ov.addEventListener('mouseleave',function(){ if(self._kbZone==='ov')  self._kbZone='none'; });
  this._el('dw').addEventListener('mouseenter',function(){ self._kbZone='det'; });
  this._el('dw').addEventListener('mouseleave',function(){ if(self._kbZone==='det') self._kbZone='none'; });

  this._onDocKey=function(e){
    if(self._kbZone==='none') return;
    var key=e.key;
    if(self._kbZone==='ov'){
      if(key==='ArrowUp'||key==='ArrowDown'){
        e.preventDefault(); self._ovZoom(key==='ArrowUp'?0.7:1.4); return;
      }
      if(key==='ArrowLeft'||key==='ArrowRight'){
        e.preventDefault();
        var w=self.vE-self.vS, step=w*0.4*(key==='ArrowRight'?1:-1);
        self.vS=clamp(self.vS+step,0,1-w); self.vE=self.vS+w;
        self._scrollOvToBrush(); self._draw();
        if(self.autoSync){ clearTimeout(self._syncTimer); self._syncTimer=setTimeout(function(){ self._syncTableToView(); },250); }
        return;
      }
    }
    if(self._kbZone==='det'&&(key==='ArrowLeft'||key==='ArrowRight')){
      e.preventDefault();
      var dir=key==='ArrowRight'?1:-1;
      if(!self.rows.length) return;
      var idx=-1;
      for(var i=0;i<self.rows.length;i++){ if(self.rows[i].id===String(self.focusId)){idx=i;break;} }
      if(idx<0){
        var vs2=self.gMin+self.vS*self.gSpan, ve2=self.gMin+self.vE*self.gSpan;
        for(var j=0;j<self.rows.length;j++){ if(self.rows[j].te>vs2&&self.rows[j].ts<ve2){idx=j;break;} }
        if(idx<0) return;
      }
      var next=clamp(idx+dir,0,self.rows.length-1);
      if(next!==idx) self._focusRow(self.rows[next]);
    }
  };
  document.addEventListener('keydown',this._onDocKey);

  // ResizeObserver
  if(window.ResizeObserver){
    this._resizeObs=new ResizeObserver(function(){ self._draw(); });
    this._resizeObs.observe(document.querySelector(this.ctnId));
  }
};

g.TimelineWidget=TimelineWidget;
}(window));
