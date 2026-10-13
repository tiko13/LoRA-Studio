import {escapeHTML as h} from './domain.js';

export function objectControls(image,project) {
  if(project.kind!=='object')return '';
  const o=image.object_target||{},s=image.object_selection||{status:'pending'},b=o.box||s.box||[0,0,1,1];
  return `<section class="card compact-prep"><h3>Training object · ${h(s.status)}</h3><p class="muted">${h(s.evidence||'Prepare dataset locates the object using your Base info description. Refine individual images here.')}</p><label class="field">Target selection<select name="object_mode"><option value="auto" ${!o.mode||o.mode==='auto'?'selected':''}>Automatic · locate the described object</option><option value="manual" ${o.mode==='manual'?'selected':''}>Manual · selected rectangle</option><option value="full" ${o.mode==='full'?'selected':''}>Keep entire image · target confirmed by me</option></select></label><label class="field">Which object in this image? <small>Optional, e.g. the striped teapot on the left. Distinguish similar objects.</small><input name="object_query" maxlength="1000" value="${h(o.query||'')}"></label><div class="actions"><button type="button" data-object-tool="box">Draw target rectangle</button><button type="button" data-object-tool="polygon">Draw mask polygon</button><button type="button" data-object-tool="clear">Clear mask</button></div><p id="object-tool-status" class="muted">Draw on the original image above. Save, then Prepare dataset to refresh the caption.</p><input type="hidden" name="object_box" value="${h(JSON.stringify(b))}"><input type="hidden" name="object_polygon" value="${h(JSON.stringify(o.polygon||[]))}"><label class="field">Space around the target<input name="object_margin" type="range" min="0" max="0.5" step="0.01" value="${o.margin??.08}"><small>Default: 8% on each side. Preserve handles, spouts and defining parts.</small></label><details><summary>Loss mask & precise coordinates</summary><p class="muted">Masks reduce loss outside the selected area. They do not remove visible context or guarantee separation. White means full loss weight.</p><div class="form-grid">${['Left','Top','Right','Bottom'].map((label,j)=>`<label class="field">${label} (%)<input type="number" data-object-coordinate="${j}" min="0" max="100" step="0.1" value="${Math.round(b[j]*1000)/10}"></label>`).join('')}</div><label class="field">Loss mask<select name="object_mask">${[['none','None · ordinary training'],['box','Target rectangle · includes background inside'],['polygon','Drawn polygon · follow the visible object'],['alpha','Original transparency · alpha channel']].map(([v,l])=>`<option value="${v}" ${(o.mask||'none')===v?'selected':''}>${l}</option>`).join('')}</select></label><label class="field">Background loss weight<input name="object_background" type="number" min="0" max="1" step="0.05" value="${o.background_weight??.1}"><small>0 excludes background loss; 0.1 keeps some context learning; 1 disables its reduction.</small></label><figure id="object-mask-figure" hidden><button type="button" class="image-preview-trigger" data-image-preview aria-label="Enlarge loss mask"><img id="object-mask-preview" alt="Aligned training loss mask" style="max-width:100%;max-height:240px;object-fit:contain"></button><figcaption>Loss mask · exact exported size</figcaption></figure></details></section>`;
}

export function readObjectForm(form) {
  if(!form.elements.object_mode)return null;
  const e=form.elements;
  return {mode:e.object_mode.value,query:e.object_query.value,box:JSON.parse(e.object_box.value),polygon:JSON.parse(e.object_polygon.value),margin:Number(e.object_margin.value),mask:e.object_mask.value,background_weight:Number(e.object_background.value)};
}

export function restoreObjectForm(form,o) {
  if(!o||!form.elements.object_mode)return;
  for(const [key,name] of Object.entries({mode:'mode',query:'query',margin:'margin',mask:'mask',background_weight:'background'}))if(o[key]!=null)form.elements['object_'+name].value=o[key];
  form.elements.object_box.value=JSON.stringify(o.box||[0,0,1,1]);
  form.elements.object_polygon.value=JSON.stringify(o.polygon||[]);
  syncCoordinates(form);
}

function syncCoordinates(form) {
  const b=JSON.parse(form.elements.object_box.value);
  form.querySelectorAll('[data-object-coordinate]').forEach(input=>input.value=Math.round(b[Number(input.dataset.objectCoordinate)]*1000)/10);
}

export function drawObjectMask(canvas,image,form) {
  if(!form.elements.object_mode)return;
  const o=readObjectForm(form),ctx=canvas.getContext('2d'),s=Math.min(400/image.naturalWidth,400/image.naturalHeight),w=image.naturalWidth*s,h=image.naturalHeight*s,x=(400-w)/2,y=(400-h)/2;
  if(o.polygon.length){ctx.beginPath();o.polygon.forEach(([px,py],i)=>ctx[i?'lineTo':'moveTo'](x+px*w,y+py*h));if(o.polygon.length>=3)ctx.closePath();ctx.fillStyle='rgba(100,220,160,.2)';ctx.fill();ctx.strokeStyle='#6cdaa1';ctx.lineWidth=2;ctx.stroke();}
}

export function bindObjectEditor(form,image) {
  if(!form.elements.object_mode)return;
  const canvas=form.querySelector('#crop-overlay');let tool='box',start=null;
  const changed=()=>form.elements.object_mode.dispatchEvent(new Event('input',{bubbles:true}));
  const point=event=>{const r=canvas.getBoundingClientRect(),s=Math.min(400/image.width,400/image.height),w=image.width*s,h=image.height*s;return [Math.max(0,Math.min(1,((event.clientX-r.left)*400/r.width-(400-w)/2)/w)),Math.max(0,Math.min(1,((event.clientY-r.top)*400/r.height-(400-h)/2)/h))];};
  canvas.style.touchAction='none';canvas.style.cursor='crosshair';
  form.querySelectorAll('[data-object-tool]').forEach(button=>button.addEventListener('click',()=>{
    tool=button.dataset.objectTool;
    if(tool==='polygon'||tool==='clear'){form.elements.object_polygon.value='[]';form.elements.object_mask.value='none';changed();}
    form.querySelector('#object-tool-status').textContent=tool==='polygon'?'Click around the visible object. Three or more points form the loss mask.':tool==='box'?'Drag a rectangle around every visible part of the target.':'Mask cleared.';
  }));
  canvas.addEventListener('pointerdown',event=>{delete canvas.dataset.previewGesture;if(form.elements.object_mode.disabled)return;if(tool==='box'){start=point(event);canvas.setPointerCapture(event.pointerId);}});
  canvas.addEventListener('pointerup',event=>{
    if(form.elements.object_mode.disabled){start=null;return;}
    const end=point(event);
    if(tool==='box'&&start){const b=[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];start=null;if(b[2]-b[0]<.01||b[3]-b[1]<.01)return;canvas.dataset.previewGesture='drawing';form.elements.object_box.value=JSON.stringify(b);form.elements.object_mode.value='manual';syncCoordinates(form);changed();}
    else if(tool==='polygon'){canvas.dataset.previewGesture='drawing';const p=JSON.parse(form.elements.object_polygon.value);if(p.length>=200)return;p.push(end);form.elements.object_polygon.value=JSON.stringify(p);if(p.length>=3)form.elements.object_mask.value='polygon';changed();}
  });
  form.querySelectorAll('[data-object-coordinate]').forEach(input=>input.addEventListener('change',()=>{const b=JSON.parse(form.elements.object_box.value);b[Number(input.dataset.objectCoordinate)]=Number(input.value)/100;form.elements.object_box.value=JSON.stringify(b);form.elements.object_mode.value='manual';changed();}));
}
