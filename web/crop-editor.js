import {installDialogFocus} from './vendor/promptstudio/dialog-focus.js';
import {moveCrop, resizeCrop, fitCropRatio, cropFromPlan} from './crop-geometry.js';

export function openCropEditor({src, preparation, plan, preview, apply, valid, trigger}) {
  if (document.querySelector('.crop-editor')) return;
  const dialog = document.createElement('dialog');
  dialog.className = 'crop-editor';
  dialog.setAttribute('aria-label', 'Crop & mask');
  dialog.innerHTML = `<header class="crop-heading"><div><h2>Crop & mask</h2><p class="muted">Drag inside the crop to move it. Drag any edge or corner to resize.</p></div><button type="button" data-close aria-label="Close crop editor">✕</button></header>
    <div class="crop-workspace"><div class="crop-main"><div class="crop-toolbar actions" aria-label="Editing tools">
    <button type="button" data-tool="crop" aria-pressed="true">Crop</button><button type="button" data-tool="hide" aria-pressed="false">Hide brush</button><button type="button" data-tool="restore" aria-pressed="false">Restore brush</button><button type="button" data-tool="rectangle" aria-pressed="false">Hide rectangle</button>
    </div><div class="crop-viewport"><div class="crop-surface"><div class="crop-stage"><img alt="Original image" draggable="false"><canvas class="crop-mask" aria-hidden="true"></canvas><div class="crop-selection" tabindex="0" aria-label="Crop selection. Arrow keys move; Shift moves ten pixels."><div class="crop-thirds"></div>${['n','ne','e','se','s','sw','w','nw'].map(side=>`<button type="button" class="crop-handle crop-${side}" data-handle="${side}" aria-label="Resize ${ {n:'top',ne:'top right',e:'right',se:'bottom right',s:'bottom',sw:'bottom left',w:'left',nw:'top left'}[side]} crop edge"></button>`).join('')}</div><div class="crop-brush-cursor" hidden></div></div></div></div>
    <div class="actions crop-view-tools"><button type="button" data-zoom="out" aria-label="Zoom out">−</button><output class="crop-zoom-label">100%</output><button type="button" data-zoom="in" aria-label="Zoom in">+</button><button type="button" data-zoom="fit">Fit view</button><label class="checkbox"><input type="checkbox" data-grid checked>Grid</label><span class="muted">Wheel to zoom at cursor · zoom changes the view only.</span></div></div>
    <aside class="crop-options"><fieldset disabled><legend class="sr-only">Crop and mask options</legend>
    <label class="field">Aspect ratio<select data-ratio><option value="0">Free · resize each side</option><option value="original">Original image</option><option value="current">Lock current ratio</option><option value="1">Square · 1:1</option><option value="1.3333333333">Landscape · 4:3</option><option value="0.75">Portrait · 3:4</option><option value="1.7777777778">Landscape · 16:9</option><option value="0.5625">Portrait · 9:16</option></select></label>
    <div class="actions"><button type="button" data-center>Center crop</button><button type="button" data-full>Full image</button><button type="button" data-reset>Automatic crop</button></div>
    <label class="field">Fit selected area<select data-fit><option value="auto">Use project default</option><option value="crop">Minimal crop · align to model grid</option><option value="pad">Keep entire selection · grey padding</option></select></label>
    <div class="crop-brush-options"><label class="field">Brush diameter <output data-brush-size></output><input type="range" data-brush min="1" value="40" step="1"></label><p class="muted">The circle shows the actual brush footprint. Hidden pixels become neutral grey, including in captions and training. Restore reveals the original.</p><button type="button" data-clear>Clear mask</button></div>
    <div class="actions"><button type="button" data-undo>Undo</button><button type="button" data-redo>Redo</button></div>
    <details><summary>Precise crop · original pixels</summary><div class="crop-coordinates">${['Left','Top','Right','Bottom'].map((label,j)=>`<label class="field">${label}<input type="number" data-coordinate="${j}" min="0" step="1"></label>`).join('')}</div><p class="muted">Coordinate edits use free aspect ratio.</p></details>
    </fieldset><figure class="crop-result"><img alt="Exact training preview" hidden><figcaption role="status">Loading original…</figcaption></figure></aside></div>
    <footer class="crop-footer"><span class="muted">Original preserved. Apply, then save the image review.</span><div class="actions"><button type="button" data-cancel>Cancel</button><button type="button" class="primary" data-apply disabled>Apply crop & mask</button></div></footer>`;
  const $ = selector => dialog.querySelector(selector);
  const picture = $('.crop-stage>img'), stage = $('.crop-stage'), selection = $('.crop-selection');
  const viewport = $('.crop-viewport'), maskCanvas = $('.crop-mask'), cursor = $('.crop-brush-cursor');
  const surface = $('.crop-surface');
  const status = $('.crop-result figcaption'), resultImage = $('.crop-result img');
  let rect = preparation.crop_rect ? [...preparation.crop_rect] : cropFromPlan(plan);
  let custom = Boolean(preparation.crop_rect), strokes = structuredClone(preparation.hide_mask || []);
  let mode = preparation.mode || 'auto', tool = 'crop', ratio = 0, zoom = 1, loaded = false;
  let drag = null, hovering = false, pointer = [.5,.5], history = [], future = [], timer, sequence = 0, controller;
  let laidOut = false;
  const snapshot = () => ({rect:[...rect], custom, strokes:structuredClone(strokes), mode, ratio});
  const remember = () => {history.push(snapshot()); if (history.length > 40) history.shift(); future = [];};
  const restore = value => {({rect, custom, strokes, mode, ratio} = value); $('[data-ratio]').value = ratio ? 'current' : '0'; $('[data-fit]').value = mode; render(); schedule();};
  const options = () => ({...preparation, mode, crop_aspect:ratio*picture.naturalWidth/picture.naturalHeight, ...(custom ? {crop_rect:rect} : {crop_rect:null}), hide_mask:strokes});
  function brushCursor() {
    const sizing = document.activeElement === $('[data-brush]');
    cursor.hidden = !loaded || (!sizing && (!hovering || !['hide','restore'].includes(tool)));
    if (cursor.hidden) return;
    const diameter = Number($('[data-brush]').value) / picture.naturalWidth * stage.clientWidth;
    const p = sizing && !hovering ? [.5,.5] : pointer;
    Object.assign(cursor.style, {width:diameter+'px',height:diameter+'px',left:p[0]*100+'%',top:p[1]*100+'%'});
  }
  function paintMask() {
    const c = maskCanvas.getContext('2d'), w = maskCanvas.width, h = maskCanvas.height;
    c.clearRect(0,0,w,h);
    for (const stroke of strokes) {
      c.globalCompositeOperation = stroke.hide ? 'source-over' : 'destination-out';
      c.strokeStyle = c.fillStyle = '#808080'; c.lineCap = c.lineJoin = 'round';
      const points = stroke.points.map(([x,y])=>[x*w,y*h]);
      if (stroke.tool === 'rectangle') {
        const [a,b] = points; c.fillRect(Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.abs(a[0]-b[0]),Math.abs(a[1]-b[1]));
      } else {
        c.lineWidth = stroke.size * Math.min(w,h);
        c.beginPath();c.arc(points[0][0],points[0][1],c.lineWidth/2,0,Math.PI*2);c.fill();
        if(points.length>1){c.beginPath();c.moveTo(...points[0]);for(const p of points.slice(1))c.lineTo(...p);c.stroke();}
      }
    }
    c.globalCompositeOperation = 'source-over';
  }
  function render() {
    selection.style.left = rect[0]*100+'%'; selection.style.top = rect[1]*100+'%';
    selection.style.width = (rect[2]-rect[0])*100+'%'; selection.style.height = (rect[3]-rect[1])*100+'%';
    selection.classList.toggle('masking',tool!=='crop');
    stage.dataset.tool = tool;
    $('[data-undo]').disabled = !history.length; $('[data-redo]').disabled = !future.length;
    for (const input of dialog.querySelectorAll('[data-coordinate]')) {const j=Number(input.dataset.coordinate);input.max=j%2?picture.naturalHeight:picture.naturalWidth;input.value=Math.round(rect[j]*Number(input.max));}
    $('[data-brush-size]').textContent = `${$('[data-brush]').value} px`;
    for(const button of dialog.querySelectorAll('[data-tool]')) {button.disabled=!loaded;button.setAttribute('aria-pressed',String(button.dataset.tool===tool));}
    if(loaded)paintMask();brushCursor();
  }
  function layout() {
    if(!loaded)return;
    const old=stage.getBoundingClientRect(),view=viewport.getBoundingClientRect();
    const center=laidOut?[(view.left+viewport.clientWidth/2-old.left)/old.width,(view.top+viewport.clientHeight/2-old.top)/old.height]:[.5,.5];
    const scale=Math.min((viewport.clientWidth-40)/picture.naturalWidth,(viewport.clientHeight-40)/picture.naturalHeight);
    const w=Math.max(1,picture.naturalWidth*scale*zoom),h=Math.max(1,picture.naturalHeight*scale*zoom);
    stage.style.width=w+'px';stage.style.height=h+'px';
    // Half a viewport of padding lets cursor anchoring work even while one
    // image dimension is smaller than the viewport (e.g. a tall portrait).
    surface.style.width=w+viewport.clientWidth+'px';
    surface.style.height=h+viewport.clientHeight+'px';
    viewport.scrollLeft=center[0]*w;viewport.scrollTop=center[1]*h;laidOut=true;
    $('.crop-zoom-label').textContent=Math.round(zoom*100)+'%';brushCursor();
  }
  function zoomAt(next, clientX, clientY) {
    if(!loaded||drag)return;
    next=Math.max(1,Math.min(8,next));if(next===zoom)return;
    const before=stage.getBoundingClientRect();
    const x=(clientX-before.left)/before.width,y=(clientY-before.top)/before.height;
    zoom=next;layout();
    const after=stage.getBoundingClientRect();
    viewport.scrollLeft+=after.left+x*after.width-clientX;
    viewport.scrollTop+=after.top+y*after.height-clientY;
    const final=stage.getBoundingClientRect();
    pointer=[Math.max(0,Math.min(1,(clientX-final.left)/final.width)),Math.max(0,Math.min(1,(clientY-final.top)/final.height))];
    brushCursor();
  }
  viewport.addEventListener('wheel',event=>{
    if(!loaded)return;
    event.preventDefault();
    const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?viewport.clientHeight:1);
    zoomAt(zoom*Math.exp(-Math.max(-240,Math.min(240,delta))*.002),event.clientX,event.clientY);
  },{passive:false});
  function schedule() {
    const ticket=++sequence;clearTimeout(timer);controller?.abort();$('[data-apply]').disabled=true;
    status.textContent='Updating training preview…';
    timer=setTimeout(async()=>{
      controller=new AbortController();
      try {
        const result=await preview(options(),controller.signal);
        if(ticket!==sequence||!dialog.open)return;
        if ((custom&&!result.plan.custom_crop)||(strokes.length&&!result.plan.hide_mask)) throw new Error('Restart LoRA Studio to enable crop and mask support, then reopen this editor.');
        resultImage.src='data:image/png;base64,'+result.png;resultImage.hidden=false;
        const p=result.plan;status.textContent=`${p.width} × ${p.height} training pixels · ${p.mode==='pad'?'grey padding':p.removed_percent+'% grid fit adjustment'}`;
        $('[data-apply]').disabled=!valid();
      } catch(error) {if(ticket===sequence&&dialog.open&&error.name!=='AbortError')status.textContent=error.message;}
    },180);
  }
  const point = event => {const b=stage.getBoundingClientRect();return [Math.max(0,Math.min(1,(event.clientX-b.left)/b.width)),Math.max(0,Math.min(1,(event.clientY-b.top)/b.height))];};
  stage.addEventListener('pointerdown',event=>{
    if(!loaded||!valid()||event.button!==0||drag)return;
    if(tool==='crop'&&!event.target.closest('.crop-selection'))return;
    if(tool!=='crop'&&strokes.length>=300){status.textContent='Mask has 300 strokes. Undo or clear the mask before adding more.';return;}
    event.preventDefault();remember();pointer=point(event);
    drag={id:event.pointerId,start:pointer,rect:[...rect],handle:event.target.dataset.handle||'move'};
    stage.setPointerCapture(event.pointerId);
    if(tool==='crop')selection.focus({preventScroll:true});
    else {strokes.push({tool:tool==='rectangle'?'rectangle':'brush',hide:tool!=='restore',size:Number($('[data-brush]').value)/Math.min(picture.naturalWidth,picture.naturalHeight),points:tool==='rectangle'?[pointer,pointer]:[pointer]});paintMask();}
    ++sequence;clearTimeout(timer);controller?.abort();$('[data-apply]').disabled=true;
  });
  stage.addEventListener('pointermove',event=>{
    pointer=point(event);hovering=true;brushCursor();
    if(!drag||event.pointerId!==drag.id)return;
    if(tool==='crop') {
      const dx=pointer[0]-drag.start[0],dy=pointer[1]-drag.start[1];
      rect=drag.handle==='move'?moveCrop(drag.rect,dx,dy):resizeCrop(drag.rect,drag.handle,dx,dy,ratio,1/picture.naturalWidth,1/picture.naturalHeight);custom=true;
    } else {
      const stroke=strokes.at(-1);
      if(stroke.tool==='rectangle')stroke.points[1]=pointer;
      else if(stroke.points.length<1000)stroke.points.push(pointer);
    }
    render();
  });
  function finish(event) {if(!drag||event.pointerId!==drag.id)return;drag=null;if(stage.hasPointerCapture(event.pointerId))stage.releasePointerCapture(event.pointerId);render();schedule();}
  stage.addEventListener('pointerup',finish);
  stage.addEventListener('pointercancel',event=>{if(!drag)return;drag=null;restore(history.pop());future=[];render();});
  stage.addEventListener('lostpointercapture',finish);
  stage.addEventListener('pointerleave',()=>{hovering=false;brushCursor();});
  selection.addEventListener('keydown',event=>{
    if(tool!=='crop'||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;
    event.preventDefault();remember();const step=event.shiftKey?10:1,dx=(event.key==='ArrowLeft'?-step:event.key==='ArrowRight'?step:0)/picture.naturalWidth,dy=(event.key==='ArrowUp'?-step:event.key==='ArrowDown'?step:0)/picture.naturalHeight;
    rect=event.target.dataset.handle?resizeCrop(rect,event.target.dataset.handle,dx,dy,ratio,1/picture.naturalWidth,1/picture.naturalHeight):moveCrop(rect,dx,dy);custom=true;render();schedule();
  });
  for(const button of dialog.querySelectorAll('[data-tool]'))button.addEventListener('click',()=>{tool=button.dataset.tool;render();});
  $('[data-brush]').addEventListener('input',render);$('[data-brush]').addEventListener('focus',brushCursor);$('[data-brush]').addEventListener('blur',brushCursor);
  $('[data-fit]').value=mode;
  $('[data-fit]').addEventListener('change',()=>{remember();mode=$('[data-fit]').value;schedule();render();});
  $('[data-ratio]').addEventListener('change',()=>{remember();const value=$('[data-ratio]').value;ratio=value==='original'?1:value==='current'?(rect[2]-rect[0])/(rect[3]-rect[1]):Number(value)*picture.naturalHeight/picture.naturalWidth;rect=fitCropRatio(rect,ratio);custom=true;render();schedule();});
  $('[data-center]').addEventListener('click',()=>{remember();rect=moveCrop(rect,.5-(rect[0]+rect[2])/2,.5-(rect[1]+rect[3])/2);custom=true;render();schedule();});
  $('[data-full]').addEventListener('click',()=>{remember();rect=[0,0,1,1];custom=true;ratio=0;$('[data-ratio]').value='0';render();schedule();});
  $('[data-reset]').addEventListener('click',async()=>{
    remember();custom=false;ratio=0;rect=[0,0,1,1];$('[data-ratio]').value='0';
    clearTimeout(timer);controller?.abort();$('[data-apply]').disabled=true;render();
    // Automatic object crops come from the backend; show their bounds too.
    try {const ticket=++sequence;const result=await preview({...options(),crop_rect:null});if(ticket!==sequence||!dialog.open)return;rect=cropFromPlan(result.plan);render();schedule();}catch(error){status.textContent=error.message;}
  });
  $('[data-clear]').addEventListener('click',()=>{remember();strokes=[];render();schedule();});
  $('[data-undo]').addEventListener('click',()=>{if(history.length){future.push(snapshot());restore(history.pop());}});
  $('[data-redo]').addEventListener('click',()=>{if(future.length){history.push(snapshot());restore(future.pop());}});
  for(const input of dialog.querySelectorAll('[data-coordinate]'))input.addEventListener('change',()=>{
    if(input.value===''||!Number.isFinite(input.valueAsNumber)){render();return;}
    remember();const j=Number(input.dataset.coordinate),size=j%2?picture.naturalHeight:picture.naturalWidth;
    const low=j<2?0:rect[j-2]+1/size,high=j<2?rect[j+2]-1/size:1;
    rect[j]=Math.max(low,Math.min(high,input.valueAsNumber/size));custom=true;ratio=0;$('[data-ratio]').value='0';render();schedule();
  });
  $('[data-grid]').addEventListener('change',()=>{$('.crop-thirds').hidden=!$('[data-grid]').checked;});
  for(const button of dialog.querySelectorAll('[data-zoom]'))button.addEventListener('click',()=>{
    const box=viewport.getBoundingClientRect();
    zoomAt(button.dataset.zoom==='fit'?1:zoom*(button.dataset.zoom==='in'?1.5:1/1.5),box.left+box.width/2,box.top+box.height/2);
    if(button.dataset.zoom==='fit'){viewport.scrollLeft=stage.clientWidth/2;viewport.scrollTop=stage.clientHeight/2;}
  });
  dialog.addEventListener('keydown',event=>{
    if(!(event.ctrlKey||event.metaKey)||event.target.matches('input,textarea,select'))return;
    if(event.key.toLowerCase()==='z'||event.key.toLowerCase()==='y'){
      event.preventDefault();$(event.shiftKey||event.key.toLowerCase()==='y'?'[data-redo]':'[data-undo]').click();
    }
  });
  for(const button of dialog.querySelectorAll('[data-close],[data-cancel]'))button.addEventListener('click',()=>dialog.close());
  $('[data-apply]').addEventListener('click',()=>{if(!valid()){status.textContent='Preparation is locked. Copy to a new run before editing.';return;}apply(options());dialog.close();});
  const observer=new ResizeObserver(layout);
  picture.onload=()=>{
    loaded=true;$('fieldset').disabled=false;
    if(preparation.crop_aspect){
      ratio=preparation.crop_aspect*picture.naturalHeight/picture.naturalWidth;
      const preset=[...$('[data-ratio]').options].find(option=>Number(option.value)>0&&Math.abs(Number(option.value)-preparation.crop_aspect)<1e-6);
      $('[data-ratio]').value=preset?.value||(Math.abs(ratio-1)<1e-6?'original':'current');
    }
    const small=Math.min(picture.naturalWidth,picture.naturalHeight);
    $('[data-brush]').min=Math.max(1,Math.ceil(small*.001));$('[data-brush]').max=Math.max(1,Math.floor(small*.5));$('[data-brush]').value=Math.max(1,Math.round(small*.05));
    const scale=Math.min(1,1600/Math.max(picture.naturalWidth,picture.naturalHeight));maskCanvas.width=Math.round(picture.naturalWidth*scale);maskCanvas.height=Math.round(picture.naturalHeight*scale);
    layout();render();schedule();
  };
  picture.onerror=()=>{status.textContent='Unable to load the original image. Close and reopen the editor to retry.';};
  document.body.append(dialog);installDialogFocus(dialog);
  dialog.addEventListener('close',()=>{++sequence;clearTimeout(timer);controller?.abort();observer.disconnect();dialog.remove();picture.removeAttribute('src');if(trigger?.isConnected)trigger.focus({preventScroll:true});});
  dialog.showModal();observer.observe(viewport);render();picture.src=src;
}
