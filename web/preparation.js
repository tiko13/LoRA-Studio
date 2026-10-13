import {conceptSummary} from './concept-preparation.js';
import {styleSummary} from './style-preparation.js';
import {objectControls} from './object-preparation.js';
import {escapeHTML as h} from './domain.js';

export function categoryProfile(project,catalog) {
  return catalog.types?.find(t=>t.id===project.kind)||{coverage:catalog.coverage||{}};
}

function categoryCoverage(images,project,catalog) {
  const profile=categoryProfile(project,catalog), included=images.filter(i=>i.included);
  return `<section class="card compact-prep"><details><summary>${h(profile.label)} coverage</summary><p class="muted">${h(profile.guidance)}</p><div class="coverage">${Object.entries(profile.coverage).map(([key,values])=>{
    const counts={};let suggestions=0;
    for(const i of included){const manual=i.effective_coverage_override?.[key],tag=manual||i.effective_coverage?.[key]||'uncertain';counts[tag]=(counts[tag]||0)+1;if(!manual&&tag!=='uncertain')suggestions++;}
    return `<div><strong>${h(key.replaceAll('_',' '))}</strong><br>${values.filter(v=>counts[v]).map(v=>`${h(v)}: ${counts[v]}`).join(' · ')||'No images'}${suggestions?`<br><small>${suggestions} unreviewed model suggestions</small>`:''}</div>`;
  }).join('')}</div><p class="muted"><small>Unknown and stale suggestions are not verified coverage. These tags do not establish semantic diversity or training quality. Correct tags in image review; no fixed quotas are imposed.</small></p></details></section>`;
}



export function datasetSummary(images,kind='character') {

  const included=images.filter(i=>i.included);

  const result={included:included.length,portrait:0,landscape:0,square:0,unchecked:0,uncertain:0,multiple:0,flags:0};

  for(const i of included) {

    if(i.width&&i.height) result[i.width/i.height>.95&&i.width/i.height<1.05?'square':i.width>i.height?'landscape':'portrait']++;

    if(kind==='character') {
    const count=i.person_count??(i.detection?.status==='complete'?i.detection.count:null);

    if(count==null) { if(i.detection?.status==='complete') result.uncertain++; else result.unchecked++; }

    else if(count>1) result.multiple++;
    }

    if(i.advisories?.length&&!i.advisories_accepted) result.flags++;

  }

  return result;

}



export function prepSettings(project,catalog) {

  const profile=catalog.models.find(m=>m.id===project.model).preparation;

  return `<section class="card compact-prep"><details><summary>Training image preparation · ${project.training.resolution} target · ${project.preparation?.mode==='pad'?'fit with padding':'preserve aspect ratio'}</summary><p class="muted">Keep the original shape. The target is a pixel budget, not a square crop or a maximum side length.</p><div class="form-grid"><label class="field">Resolution target<input type="number" data-field="training.resolution" min="256" max="${profile.maximum}" step="${profile.multiple}" value="${project.training.resolution}"><small>1024 ≈ 1 megapixel. Width and height align to ${profile.multiple} pixels.</small></label><label class="field">Default fit<select data-preparation="mode"><option value="crop" ${(project.preparation?.mode||'crop')==='crop'?'selected':''}>Minimal crop · preserve proportions</option><option value="pad" ${project.preparation?.mode==='pad'?'selected':''}>Fit entire image · grey padding</option></select><small>Open an original image to crop and mask it. Padding is visible training content.</small></label></div><p class="muted"><small>${h(profile.note)} Existing resolution choices are preserved.${project.kind!=='character'?' '+h(catalog.models.find(m=>m.id===project.model).caption_guidance||''):''}</small></p><div class="actions"><button data-action="apply-preparation">Save & refresh preparation</button><button data-action="recommended-resolution">Use recommended ${profile.default} target</button></div></details></section>`;

}



export function prepSummary(images,project={kind:'character'},catalog={}) {

  const s=datasetSummary(images,project.kind);
  if(project.kind!=='character') return `<section class="card">${project.kind==='object'?`<div class="actions"><button data-action="objects" ${images.length?'':'disabled'}>Find objects</button><span>${images.filter(i=>i.included&&i.object_selection?.status!=='ready').length} need target selection</span></div><p class="muted">Prepare dataset locates the target, crops, captions and exports in one job. Open an image to refine its selection. Ambiguous or absent targets require review.</p>`:''}${styleSummary(project)}${conceptSummary(project,catalog)}<h3>Dataset checks · ${h(categoryProfile(project,catalog).label)}</h3><p class="muted">${h(categoryProfile(project,catalog).guidance)}</p><div class="coverage"><div>${s.portrait} portrait · ${s.landscape} landscape · ${s.square} near-square</div><div>${s.flags} with unacknowledged advisories</div></div><p class="muted">Open an image for category-specific visual checks and coverage corrections. Advisories never exclude images or block training.</p></section>`;

  return `<section class="card"><div class="card-head"><div><h3>Dataset checks</h3><p>Advisories never exclude images or block training. Open an image to correct counts or keep it anyway.</p></div><button data-action="detect" ${images.length?'':'disabled'}>Check person counts</button></div><div class="coverage"><div>${s.portrait} portrait · ${s.landscape} landscape · ${s.square} near-square</div><div>${s.flags} with unacknowledged advisories</div><div>${s.multiple} with multiple people reported</div><div>${s.uncertain} ambiguous person counts</div><div>${s.unchecked} person counts unchecked</div></div><details><summary>How person checks work</summary><p class="muted"><small>Identity matching is disabled. Optional CPU detection uses the configured detector/AI Toolkit Python and downloads ~14 MB of public weights on first use. Images stay local. Counts may miss close-ups, occlusions and unusual poses; one detection does not prove one person. Nested boxes are shown as an uncertain count range because a head and its own body may be detected separately. Framing, pose, expression and lighting coverage still need human review.</small></p></details></section>`;

}



export function coverageSummary(images,project,catalog) {

  if(project.kind!=='character')return categoryCoverage(images,project,catalog);

  const included=images.filter(i=>i.included), rows=Object.entries(catalog.coverage).map(([key,values])=>{

    const counts={};let unreviewed=0;

    for(const i of included){const manual=i.coverage_override?.[key],tag=manual||(!i.preparation_stale&&(i.coverage_kind||'character')==='character'?i.coverage?.[key]:null)||'uncertain';counts[tag]=(counts[tag]||0)+1;if(!manual&&tag!=='uncertain')unreviewed++;}

    return `<div><strong>${h(key)}</strong><br>${values.filter(v=>counts[v]).map(v=>`${h(v)}: ${counts[v]}`).join(' · ')||'No images'}${unreviewed?`<br><small>${unreviewed} vision suggestions</small>`:''}</div>`;

  });

  return `<section class="card compact-prep"><details><summary>Person coverage · framing, viewpoint & lighting</summary><p class="muted">Coverage tags describe the training image. Vision suggestions can be corrected in each review. Unknowns and stale crop tags are not counted as verified coverage.</p><div class="coverage">${rows.join('')}</div><p class="muted"><small>Aim for useful face detail plus varied viewpoints, expressions, clothing and settings where those should remain changeable. No fixed quota or identity score is imposed. Review actual examples; counts alone do not establish diversity.</small></p></details></section>`;

}



export function coverageControls(i,catalog,project={kind:'character'}) {
  if(project.kind!=='character') return `<details><summary>Coverage tags · optional corrections</summary><p class="muted"><small>Tags describe the selected category. Suggestions can be wrong; corrections are preserved separately for each category.</small></p>${Object.entries(categoryProfile(project,catalog).coverage).map(([key,values])=>`<label class="field">${h(key.replaceAll('_',' '))}<select name="coverage_${key}"><option value="">Use model suggestion (${h(i.coverage_kind===project.kind&&!i.preparation_stale?i.coverage?.[key]||'unchecked':'unchecked')})</option>${values.map(v=>`<option value="${v}" ${i.effective_coverage_override?.[key]===v?'selected':''}>${h(v)}</option>`).join('')}</select></label>`).join('')}</details>`;

  return `<details><summary>Coverage tags · optional corrections</summary><p class="muted"><small>Suggestions describe visible framing, viewpoint and lighting, not identity. Manual tags remain until you change them.</small></p>${Object.entries(catalog.coverage).map(([key,values])=>`<label class="field">${h(key)}<select name="coverage_${key}"><option value="">Use vision suggestion (${h((i.coverage_kind||'character')==='character'?i.coverage?.[key]||'unchecked':'unchecked')})</option>${values.map(v=>`<option value="${v}" ${i.coverage_override?.[key]===v?'selected':''}>${h(v)}</option>`).join('')}</select></label>`).join('')}</details>`;

}



export function imagePrepInfo(i) {

  return i.plan?`<small>${i.width} × ${i.height} → ${i.plan.width} × ${i.plan.height}</small>${i.advisories?.length?`<span class="badge ${i.advisories_accepted?'':'warn'}">${i.advisories_accepted?'Kept despite flags':i.advisories.length+' advisories'}</span>`:''}`:`<small>${h(i.preparation_error||'Dimensions unavailable')}</small>`;

}



export function reviewPreparation(i,project={kind:'character'}) {

  const p=i.preparation||{mode:'auto',x:.5,y:.5},d=project.kind==='character'?i.detection:null;

  return `<div class="preview-pair"><figure><button type="button" class="image-preview-trigger" data-crop-editor aria-label="Open crop and mask editor"><canvas id="crop-overlay" width="400" height="400" aria-label="Original image with training crop outline"></canvas></button><figcaption>Original · click to crop & mask</figcaption></figure><figure><button type="button" class="image-preview-trigger" data-image-preview aria-label="Enlarge training image"><img id="prepared-preview" alt="Exact prepared training image"></button><figcaption>Training image</figcaption></figure></div>${d?.status==='complete'?'<label class="checkbox"><input name="show_boxes" type="checkbox" checked>Show detector boxes · solid ≥50%, dashed uncertain</label>':''}<p id="preparation-result" class="muted">Loading training preview…</p>${objectControls(i,project)}<label class="field">Fit for this image<select name="prep_mode">${[['auto','Use project default'],['crop','Minimal crop'],['pad','Fit with grey padding']].map(([v,l])=>`<option value="${v}" ${p.mode===v?'selected':''}>${l}</option>`).join('')}</select></label><input name="prep_x" type="hidden" value="${p.x??.5}"><input name="prep_y" type="hidden" value="${p.y??.5}"><input name="prep_aspect" type="hidden" value="${p.crop_aspect||0}"><input name="prep_rect" type="hidden" value="${h(JSON.stringify(p.crop_rect||null))}"><input name="prep_mask" type="hidden" value="${h(JSON.stringify(p.hide_mask||[]))}"><button type="button" data-crop-editor>Edit crop & mask</button><p class="muted"><small>Open the original to drag crop edges, choose an aspect ratio or hide unwanted content. The original is preserved.</small></p>${project.kind==='character'?`<label class="field">Person count override<input name="person_count" type="number" min="0" max="100" step="1" value="${i.person_count??''}" placeholder="Unknown / use detector"><small>${d?.status==='complete'?`Original-image detector: ${d.count??`${d.count_min}–${d.count_max} (overlapping boxes)`} at ≥ ${Math.round(d.threshold*100)}% score${d.uncertain?' · additional uncertain detection(s)':''}. This is a suggestion.`:d?.reason||'Person detector has not checked this image.'} Clear to use detector again.</small></label>`:''}${i.advisories?.length?`<div class="note warning">${i.advisories.map(f=>`<p>${h(f)}</p>`).join('')}</div>`:''}<label class="checkbox"><input name="accept_advisories" type="checkbox" ${i.advisories_accepted?'checked':''}>Keep anyway · acknowledge preparation flags</label><p class="muted"><small>Flags are advisory even without acknowledgement. Inclusion and caption review remain your choices.</small></p>`;

}



export function drawCrop(canvas,image,plan,detection) {

  const ctx=canvas.getContext('2d'),s=Math.min(400/image.naturalWidth,400/image.naturalHeight),w=image.naturalWidth*s,ht=image.naturalHeight*s,x=(400-w)/2,y=(400-ht)/2;

  ctx.clearRect(0,0,400,400);ctx.drawImage(image,x,y,w,ht);

  if(plan.mode==='crop') {

    const r=plan.source_region||[0,0,plan.source_width,plan.source_height],rw=(r[2]-r[0])*s,rh=(r[3]-r[1])*s,sx=rw/plan.resize_width,sy=rh/plan.resize_height,l=x+r[0]*s+plan.left*sx,t=y+r[1]*s+plan.top*sy,cw=plan.width*sx,ch=plan.height*sy;

    ctx.fillStyle='rgba(0,0,0,.5)';ctx.fillRect(x,y,w,t-y);ctx.fillRect(x,t+ch,w,y+ht-t-ch);ctx.fillRect(x,t,l-x,ch);ctx.fillRect(l+cw,t,x+w-l-cw,ch);

    ctx.strokeStyle='#b9a7ff';ctx.lineWidth=2;ctx.strokeRect(l,t,cw,ch);

  }

  if(plan.object_box){const b=plan.object_box;ctx.strokeStyle='#6cdaa1';ctx.lineWidth=2;ctx.strokeRect(x+b[0]*w,y+b[1]*ht,(b[2]-b[0])*w,(b[3]-b[1])*ht);}
  for(const [index,{box,score}] of (detection?.boxes||[]).entries()) {

    const certain=score>=.5&&!detection.overlap_candidates?.includes(index);

    ctx.strokeStyle=certain?'#6cdaa1':'#e9bb63';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=2;ctx.setLineDash(certain?[]:[4,4]);

    ctx.strokeRect(x+box[0]*w,y+box[1]*ht,(box[2]-box[0])*w,(box[3]-box[1])*ht);

    ctx.font='12px sans-serif';ctx.fillText(Math.round(score*100)+'%',x+box[0]*w+3,y+box[1]*ht+14);

  }

  ctx.setLineDash([]);

}
