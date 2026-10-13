import {escapeHTML as h} from './domain.js';
export const finite = v => typeof v==='number'&&Number.isFinite(v);
export function ema(points, window=30, key='loss') {
  let value; const alpha=2/(Math.max(1,window)+1);
  return points.filter(p=>finite(p[key])).map(p=>{value=value===undefined?p[key]:alpha*p[key]+(1-alpha)*value;return {...p,[key]:value};});
}
export function duration(s) {if(!finite(s))return '—';s=Math.max(0,Math.round(s));return s>=3600?`${Math.floor(s/3600)}h ${Math.floor(s%3600/60)}m`:s>=60?`${Math.floor(s/60)}m ${s%60}s`:`${s}s`;}
export function number(v,digits=3) {return finite(v)?Math.abs(v)>0&&Math.abs(v)<.001?v.toExponential(2):v.toLocaleString(undefined,{maximumFractionDigits:digits}):'—';}
export function bytes(v){if(!finite(v))return '—';const i=Math.min(3,Math.floor(Math.log2(Math.max(1,v))/10));return `${number(v/1024**i,1)} ${['B','KiB','MiB','GiB'][i]}`;}
export function chartSVG(points,key,{smooth=1,time=false,label=key}={}) {
  const data=points.filter(p=>finite(p[key])&&finite(p.x??p.step));
  if(!data.length)return '<div class="chart-empty">Waiting for measurements</div>';
  const W=800,H=240,L=65,R=20,T=16,B=38;
  let min=Math.min(...data.map(p=>p[key])),max=Math.max(...data.map(p=>p[key]));
  if(min===max){const pad=Math.abs(min)*.1||.01;min-=pad;max+=pad;}
  const lo=data[0].x??data[0].step,hi=data.at(-1).x??data.at(-1).step;
  const sx=x=>L+(x-lo)/(hi-lo||1)*(W-L-R),sy=y=>H-B-(y-min)/(max-min)*(H-T-B);
  // Keep extrema within each horizontal bin so large histories retain spikes.
  function path(rows){let reduced=rows;if(rows.length>1600){reduced=[];const size=Math.ceil(rows.length/800);for(let i=0;i<rows.length;i+=size){const block=rows.slice(i,i+size);const sorted=[block.reduce((a,b)=>a[key]<b[key]?a:b),block.reduce((a,b)=>a[key]>b[key]?a:b)].sort((a,b)=>(a.x??a.step)-(b.x??b.step));reduced.push(...sorted);}}return reduced.map((p,i)=>`${i?'L':'M'}${sx(p.x??p.step).toFixed(2)},${sy(p[key]).toFixed(2)}`).join(' ');}
  const grid=Array.from({length:5},(_,i)=>{const y=min+(max-min)*i/4;return `<line x1="${L}" x2="${W-R}" y1="${sy(y)}" y2="${sy(y)}"/><text x="${L-9}" y="${sy(y)+4}" text-anchor="end">${h(number(y))}</text>`;}).join('');
  const ticks=Array.from({length:5},(_,i)=>{const x=lo+(hi-lo)*i/4;return `<text x="${sx(x)}" y="${H-10}" text-anchor="middle">${h(time?new Date(x*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}):Math.round(x).toLocaleString())}</text>`;}).join('');
  return `<svg class="metric-chart" viewBox="0 0 ${W} ${H}" role="img" tabindex="0" aria-label="${h(label)}; ${data.length} measurements, latest ${h(number(data.at(-1)[key]))}" data-key="${h(key)}" data-min-x="${lo}" data-max-x="${hi}"><g class="chart-gridlines">${grid}${ticks}</g><path class="chart-line ${smooth>1?'raw':''}" d="${path(data)}"/>${smooth>1?`<path class="chart-line" d="${path(ema(data,smooth,key))}"/>`:''}<circle cx="${sx(hi)}" cy="${sy(data.at(-1)[key])}" r="3" class="chart-dot"/></svg>`;
}
export function metricsCSV(points){return 'step,loss,learning_rate,seconds_per_step,trainer_elapsed_seconds\n'+points.map(p=>[p.step,p.loss,p.lr,p.seconds_per_step,p.elapsed].map(v=>finite(v)?v:'').join(',')).join('\n');}
export function sampleIdentity(path){const name=path.split('/').at(-1);const match=name.match(/_(\d+)_(\d+)\.(?:jpg|png)$/i);return match?{step:Number(match[1]),index:Number(match[2])}:{step:null,index:name};}
