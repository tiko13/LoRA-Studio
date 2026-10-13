import test from 'node:test';
import assert from 'node:assert/strict';
import {escapeHTML,imageState,resolvedAttributes} from '../web/domain.js';
import {datasetSummary,prepSummary,coverageSummary,coverageControls,reviewPreparation} from '../web/preparation.js';
import {captionEvidence} from '../web/review.js';
import {presetPanel, previewPanel, promptMode} from '../web/training.js';
import {objectControls} from '../web/object-preparation.js';
import {styleSettings,styleSummary} from '../web/style-preparation.js';
import {conceptSettings,conceptSummary} from '../web/concept-preparation.js';
import {llmSettings,captureLlm} from '../web/llm-settings.js';
import {comparisonGrid,comparisonList} from '../web/comparisons.js';

test('comparison grid keeps seed rows and checkpoint columns and escapes saved content',()=>{
  const data={id:'a'.repeat(32),job:'synthetic',created:0,workflow_name:'<workflow>',status:'failed',error:'<error>',
    settings:{strength:1,prompt:'<prompt>',models:{},seeds:[42,43]},columns:[{label:'Step <25>'},{label:'Final'}],
    cells:[42,43].flatMap(seed=>[0,1].map(column=>({id:`${seed}-${column}`,seed,column,status:'failed',images:[],error:'<cell error>'})))};
  const html=comparisonGrid(data);
  assert.match(html,/&lt;workflow&gt;/);assert.match(html,/&lt;prompt&gt;/);assert.match(html,/&lt;cell error&gt;/);
  assert.doesNotMatch(html,/<workflow>|<prompt>|<cell error>/);
  assert.equal((html.match(/scope="row"/g)||[]).length,2);
  assert.equal((html.match(/data-cell=/g)||[]).length,4);
  assert.match(comparisonList([data]),/0\/4/);
});

test('backend controls preserve missing selections and escape provider/profile values',()=>{
  const html=llmSettings({integration:'standalone',llm_provider:'ollama',ollama_model:'<saved-model>',ollama_url:'http://localhost:11434',
    llm_profile_library:{synthetic:{name:'<profile>'}},llm_profile_selection:{ollama:'synthetic'}},{models:['available']});
  assert.match(html,/&lt;saved-model&gt;/);assert.match(html,/&lt;profile&gt;/);
  assert.doesNotMatch(html,/<saved-model>|<profile>/);
  assert.match(html,/Use Prompt Studio backend \(recommended\)/);
  assert.match(html,/Thinking sampler settings/);
  assert.match(html,/Available instruct modes/);
});

test('changing providers captures the previous provider sampler without overwriting the new one',()=>{
  const draft={llm_provider:'ollama',llm_profiles:{ollama:{temperature:.4},koboldcpp:{temperature:.8}},llamacpp_config_profile:'preserved.json'};
  const input=(name,value,type='text',modes=false)=>({name,value,type,hasAttribute:()=>modes});
  const container={querySelector:selector=>selector==='[data-llm-profile-provider]'?{dataset:{llmProfileProvider:'ollama'}}:null,
    querySelectorAll:selector=>selector==='[data-llm-field]'?[input('llm_provider','koboldcpp')]:[
      input('temperature','.25','number'),input('thinking_modes','Low, High','text',true),input('instruct_modes','Disabled','text',true)]};
  captureLlm(container,draft);
  assert.equal(draft.llm_provider,'koboldcpp');assert.equal(draft.llm_profiles.ollama.temperature,.25);
  assert.equal(draft.llm_profiles.koboldcpp.temperature,.8);assert.deepEqual(draft.llm_profiles.ollama.thinking_modes,['Low','High']);
  assert.equal(draft.llamacpp_config_profile,'preserved.json');
});

test('concept presets resolve Auto while preserving overrides and hidden choices',()=>{
  const catalog={types:[{id:'concept',attributes:[{id:'action',default:'keep'},{id:'expression',default:'change'},{id:'viewpoint',default:'change'}]}],concept_families:{expression:{label:'Expression',keep:['expression'],note:'<unsafe>'},camera_angle:{label:'Angle',keep:['viewpoint'],note:''}}};
  const p={kind:'concept',concept_family:'expression',choices:{action:'change',hair_color:'keep'}};
  assert.deepEqual(resolvedAttributes(p,catalog).map(a=>a.resolved),['change','keep','change']);
  p.concept_family='camera_angle';
  assert.deepEqual(resolvedAttributes(p,catalog).map(a=>a.resolved),['change','change','keep']);
  assert.equal(p.choices.hair_color,'keep');
  assert.ok(conceptSettings(p,catalog).includes('value="camera_angle" selected'));
  p.concept_family='expression';
  assert.ok(conceptSettings(p,catalog).includes('&lt;unsafe&gt;'));
  assert.ok(conceptSummary(p,catalog).includes('&lt;unsafe&gt;'));
  assert.equal(conceptSettings({kind:'style'},catalog),'');
});

test('style family controls are optional, escaped and preserve selected values',()=>{
  const catalog={style_families:{general:{label:'Any',note:''},photo:{label:'Photography',note:'<script>bad</script>'}}};
  assert.equal(styleSettings({kind:'object'},catalog),'');
  assert.equal(styleSummary({kind:'character'}),'');
  const html=styleSettings({kind:'style',style_family:'photo'},catalog);
  assert.ok(html.includes('value="photo" selected'));
  assert.ok(html.includes('&lt;script&gt;'));
  assert.ok(!html.includes('<script>'));
  assert.ok(html.includes('Optional'));
  assert.ok(styleSummary({kind:'style'}).includes('object colours remain captionable'));
});

test('object refinement controls stay category-specific and escape target evidence',()=>{
  const image={object_target:{query:'<script>target</script>'},object_selection:{status:'ambiguous',evidence:'<img src=x>'}};
  for(const kind of ['character','style','concept'])assert.equal(objectControls(image,{kind}),'');
  const html=objectControls(image,{kind:'object'});
  assert.ok(html.includes('object_mask'));assert.ok(html.includes('object_mode'));
  assert.ok(!html.includes('<script>'));assert.ok(!html.includes('<img src=x>'));
  assert.equal(imageState({included:true,object_selection:{status:'ambiguous'},caption:'caption',reviewed:true},'x').label,'Object selection needed');
});

test('preview controls retain custom text and escape model and preset text',()=>{
  const p={model:'krea2',sample_prompts:'<script>custom</script>',sample_prompt_mode:'automatic'};
  assert.equal(promptMode(p),'automatic');
  const html=previewPanel(p,{prompts:[{role:'familiar',text:'<script>preview</script>'}]});
  assert.ok(html.includes('&lt;script&gt;preview'));assert.ok(!html.includes('<script>'));
  const manual=previewPanel({...p,sample_prompt_mode:'manual',sample_prompt_history:[{text:'old',at:1}]});
  assert.ok(manual.includes('&lt;script&gt;custom'));assert.ok(manual.includes('restore-preview-prompts'));
  assert.ok(presetPanel(p,[{model:'krea2',id:'test',name:'<img src=x>'}]).includes('&lt;img'));
});

test('caption evidence distinguishes extraction from vision accuracy and escapes model text',()=>{
  const html=captionEvidence({caption_facts:[{attribute:'pose',text:'standing',sentence:1,source_sentence:'<script>bad</script>'}],caption_omitted_facts:[{text:'<img src=x>',sentence:2}]});
  assert.ok(html.includes('not whether the vision model'));
  assert.ok(html.includes('1 omitted'));assert.ok(html.includes('Omitted text-editor claims'));
  assert.ok(html.includes('&lt;script&gt;'));assert.ok(!html.includes('<script>'));
  assert.ok(!html.includes('<img src=x>'));assert.equal(captionEvidence({}),'');
});

test('changed preparation needs caption review even for a manual caption',()=>{
  assert.equal(imageState({included:true,caption:'x',intent_digest:'x',manual:true,preparation_stale:true},'x').label,'Crop review needed');
});
test('manual counts supersede detector and unchecked is distinct from zero',()=>{
  const summary=datasetSummary([{included:true,width:600,height:900,person_count:1,detection:{status:'complete',count:3}},
    {included:true,width:900,height:600},{included:true,width:512,height:512,person_count:0},
    {included:false,person_count:5}]);
  assert.equal(summary.unchecked,1);assert.equal(summary.multiple,0);assert.equal(summary.included,3);
});

test('ambiguous overlap counts are separate from unchecked and can be overridden',()=>{
  const i={included:true,detection:{status:'complete',count:null,count_min:1,count_max:2}};
  assert.equal(datasetSummary([i]).uncertain,1);
  assert.equal(datasetSummary([i]).multiple,0);
  assert.equal(datasetSummary([i]).unchecked,0);
  assert.equal(datasetSummary([{...i,person_count:1}]).uncertain,0);
});
test('user supplied text cannot become markup',()=>{
  assert.equal(escapeHTML('<img src=x onerror="alert(1)">'), '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;');
});
test('stale captions cannot appear ready even after previous review',()=>{
  assert.equal(imageState({included:true,caption:'target',reviewed:true,intent_digest:'old'},'new').label,'Rules changed');
});
test('excluded and warning captions have distinct states',()=>{
  assert.equal(imageState({included:false},'x').label,'Excluded');
  assert.equal(imageState({included:true,caption:'x',intent_digest:'x',warnings:['unclear'],reviewed:false},'x').label,'Review needed');
});

test('an automatic caption without model warnings is still an unreviewed draft',()=>{
  const i={included:true,caption:'target sitting',intent_digest:'x',warnings:[],manual:false};
  assert.equal(imageState(i,'x').label,'Caption draft');
  assert.equal(imageState({...i,reviewed:true},'x').label,'Ready');
});
test('Auto follows current type without removing hidden choices',()=>{
  const p={kind:'style',choices:{hair:'keep',medium:'auto'}};
  const catalog={types:[{id:'style',attributes:[{id:'medium',default:'keep'}]}]};
  assert.equal(resolvedAttributes(p,catalog)[0].resolved,'keep');
  assert.equal(p.choices.hair,'keep');
});
import {filterImages,reviewPanel,workflowControl} from '../web/review.js';
import {ema,chartSVG,metricsCSV,sampleIdentity} from '../web/monitoring.js';
import {trainingLabel,visibleTraining,activityCards,projectTrainingBadge} from '../web/activity.js';
import {metadataPanel,setMetadataField} from '../web/lora-metadata.js';

test('public metadata edits preserve unrelated state and blank versus zero strength',()=>{
  const p={description:'Private target',trigger:'training_token'};
  setMetadataField(p,'author','Example');
  setMetadataField(p,'trigger_words','first phrase\nsecond');
  setMetadataField(p,'recommended_strength','0');
  assert.equal(p.lora_metadata.recommended_strength,0);
  assert.deepEqual(p.lora_metadata.trigger_words,['first phrase','second']);
  assert.equal(p.lora_metadata.author,'Example');
  setMetadataField(p,'recommended_strength','');
  setMetadataField(p,'trigger_words','  ');
  assert.equal(p.lora_metadata.recommended_strength,null);
  assert.equal(p.lora_metadata.trigger_words,null);
  assert.equal(p.description,'Private target');
  assert.equal(p.trigger,'training_token');
  assert.equal(setMetadataField(p,'__proto__','x'),false);
});

test('public metadata panel escapes authored text and never fills from private description',()=>{
  const html=metadataPanel({description:'PRIVATE_TARGET',trigger:'<img onerror=x>',
    lora_metadata:{description:'</textarea><script>x</script>',author:'" onfocus="x',trigger_words:['<b>'],recommended_strength:0}});
  assert.ok(!html.includes('PRIVATE_TARGET'));
  assert.ok(!html.includes('<script>'));
  assert.ok(!html.includes('<img'));
  assert.ok(!html.includes('<b>'));
  assert.ok(html.includes('value="0"'));
  assert.ok(metadataPanel({}).includes('LoRA file metadata'));
});

test('sidebar distinguishes active training, preparation and paused runs',()=>{
  const run={id:'job',project:'project',kind:'train',status:'running',phase:'Training',pid:123,created:1};
  assert.equal(trainingLabel(run),'LoRA in training');
  assert.equal(trainingLabel({...run,pid:null}),'Preparing for training');
  assert.equal(trainingLabel({...run,status:'paused'}),'Training paused');
  assert.equal(visibleTraining([run,{...run,status:'complete'},{...run,kind:'caption'}]).length,1);
  const html=activityCards([run],[{id:'project',name:'<img onerror="x">'}]);
  assert.ok(html.includes('data-dashboard-job="job"'));assert.ok(!html.includes('<img'));
});

test('project completion comes from its latest saved training run and updates on continuation',()=>{
  const run={id:'run',project:'project',kind:'train',status:'complete',created:1};
  assert.match(projectTrainingBadge(JSON.parse(JSON.stringify([run])),'project'),/complete.*✓ Training complete/);
  assert.equal(projectTrainingBadge([run],'copy'),'');
  assert.equal(projectTrainingBadge([{...run,kind:'export'}],'project'),'');
  assert.equal(projectTrainingBadge([{...run,kind:'caption'}],'project'),'');
  for(const status of ['failed','cancelled','interrupted']) {
    assert.equal(projectTrainingBadge([{...run,id:'new',created:2,status},run],'project'),'');
  }
  for(const status of ['queued','running','stopping','paused']) {
    const badge=projectTrainingBadge([{...run,status}],'project');
    assert.ok(badge.includes(trainingLabel({...run,status})));
    assert.ok(!badge.includes('Training complete'));
  }
  assert.equal(activityCards([run],[{id:'project'}]),'');
});

test('monitoring preserves missing metrics, smooths measured values and escapes labels',()=>{
  const points=[{step:1,loss:1},{step:2,loss:null},{step:3,loss:0}];
  assert.deepEqual(ema(points,3).map(p=>p.loss),[1,.5]);
  assert.ok(chartSVG([], 'loss').includes('Waiting for measurements'));
  const svg=chartSVG(points,'loss',{label:'<img onerror="alert(1)">',smooth:3});
  assert.ok(!svg.includes('<img'));assert.ok(!svg.includes('NaN'));
  assert.equal(metricsCSV(points).split('\n')[2],'2,,,,');
  assert.deepEqual(sampleIdentity('output/samples/123__000000250_2.jpg'),{step:250,index:2});
});

test('visual findings remain distinct from human review and can be filtered after override',()=>{
  const i={included:true,caption:'shape',intent_digest:'current',reviewed:true,visual_review:{status:'concerns'}};
  assert.equal(imageState(i,'current').label,'Visual concerns');
  assert.equal(filterImages([i],'concerns').length,1);
  assert.equal(filterImages([{...i,visual_review:{status:'overridden'}}],'concerns').length,0);
});
test('visual model text and questions cannot become markup',()=>{
  const output=reviewPanel({caption:'shape',visual_review:{status:'concerns'},verification:{at:0,source:'connected',focus:'<script>x</script>',findings:[{category:'pose',status:'uncertain',observation:'<img onerror=x>'}]}});
  assert.ok(!output.includes('<script>'));assert.ok(!output.includes('<img onerror'));
});
test('automatic mode clearly distinguishes advisory use from manual review',()=>{
  assert.ok(workflowControl({training_mode:'automatic'}).includes('Advisory findings do not require approval'));
  assert.ok(workflowControl({training_mode:'review'}).includes('resolve or override'));
});

test('non-person review and dataset controls use category coverage and exclude person checks',()=>{
  for(const [kind,coverage,checks] of [
    ['object',{target_visibility:['whole','absent','uncertain']},['target','material','placement']],
    ['style',{content:['object','abstract','uncertain']},['content','style','composition']],
    ['concept',{relationship:['clear','unclear','uncertain']},['action','participants','props']],
  ]) {
    const p={kind},cat={types:[{id:kind,label:kind,guidance:'Review visible evidence.',coverage,checks}]};
    const i={included:true,caption:'synthetic',width:512,height:384,person_count:0,detection:{status:'complete',count:2,threshold:.5}};
    const html=prepSummary([i],p,cat)+coverageSummary([i],p,cat)+reviewPreparation(i,p)+coverageControls(i,cat,p)+reviewPanel(i,p,cat);
    for(const irrelevant of ['Person count','person counts','verify_people','verify_exposure','coverage_framing','show_boxes'])assert.ok(!html.includes(irrelevant),kind+': '+irrelevant);
    assert.ok(html.includes('coverage_'+Object.keys(coverage)[0]));
    assert.ok(html.includes('verify_'+checks[0]));
  }
});

test('category coverage restores selected corrections and escapes model tag text',()=>{
  const p={kind:'object'},cat={types:[{id:'object',label:'Object',guidance:'',coverage:{viewpoint:['front','top','uncertain']}}]};
  const i={coverage_kind:'style',coverage:{viewpoint:'<script>'},effective_coverage_override:{viewpoint:'top'}};
  const html=coverageControls(i,cat,p);
  assert.ok(html.includes('value="top" selected'));
  assert.ok(html.includes('unchecked'));
  assert.ok(!html.includes('<script>'));
});

test('person review controls remain available',()=>{
  const html=reviewPreparation({})+reviewPanel({caption:'synthetic'});
  assert.ok(html.includes('name="person_count"'));
  assert.ok(html.includes('name="verify_pose"'));
  assert.ok(html.includes('name="verify_exposure"'));
});

import {completedRun,resultsView,resultURL} from '../web/results.js';
test('completion navigation follows the viewed training run without interrupting editing',()=>{
  const previous=[{id:'run',kind:'train',status:'running'}],current=[{...previous[0],status:'complete'}];
  assert.equal(completedRun(previous,current,'dashboard','run'),'run');
  assert.equal(completedRun(previous,current,'training',null),'run');
  for(const tab of ['setup','dataset','results','runs'])assert.equal(completedRun(previous,current,tab,'run'),null);
  assert.equal(completedRun(previous,current,'dashboard','other'),null);
  assert.equal(completedRun(current,current,'dashboard','run'),null);
  assert.equal(completedRun(previous,[{...current[0],status:'failed'}],'dashboard','run'),null);
});
test('results escape file names and prompts and show unavailable previews and incomplete saves',()=>{
  const job={id:'run',project:'project',kind:'train',status:'complete',created:1};
  const data={job,total_steps:100,output_available:true,other_previews:[],sampling_disabled:false,checkpoints:[
    {name:'<img onerror=bad>.safetensors',path:'output/a & b.safetensors',step:100,final:true,ready:true,size:100,modified:1,previews:[{path:'output/samples/a.png',name:'a.png',index:0,step:100,prompt:'<script>bad</script>'}]},
    {name:'broken.safetensors',path:'output/broken.safetensors',step:25,final:false,ready:false,size:3,modified:1,previews:[]},
  ]};
  const html=resultsView([job],'run',data);
  assert.match(html,/&lt;img onerror=bad&gt;/);assert.match(html,/&lt;script&gt;/);
  assert.match(html,/No previews saved at this exact step/);assert.match(html,/Incomplete or unreadable/);
  assert.equal((html.match(/Download LoRA/g)||[]).length,1);
  assert.equal((html.match(/data-delete-checkpoint=/g)||[]).length,2);
  assert.match(html,/data-delete-checkpoint="output\/a &amp; b.safetensors"/);
  assert.doesNotMatch(html,/data-continue-checkpoint/);
  const active=resultsView([{...job,status:'running'}],'run',{...data,job:{...job,status:'running'}});
  assert.equal((active.match(/data-delete-checkpoint="[^"]*" disabled/g)||[]).length,2);
  assert.match(resultURL('run','output/a & b.safetensors'),/a%20%26%20b/);
  assert.match(resultsView([],null,null),/after training/);
});
