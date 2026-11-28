// 앱 틀(헤더 + 사이드바) 씌우기: <body data-on="고객 심사" data-ver="1안 ..."> 안의 <main> 내용을 감싼다
const ICON={'고객 심사':'ti-user-check','XAI 분석':'ti-align-left','공정성':'ti-scale','모델 성능':'ti-chart-line','통계 검증':'ti-layout-board'};
function navA(x,on){return `<a class="${x===on?'on':''}"><i class="ti ${ICON[x]}"></i>${x}</a>`}
(function(){
  const lk=document.createElement('link');lk.rel='stylesheet';lk.href='https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/tabler-icons.min.css';document.head.appendChild(lk);
  const st=document.createElement('style');st.textContent='.sb a{display:flex;align-items:center;gap:12px}.sb a i{font-size:21px;color:#6B7684;width:22px}.sb a.on i{color:#3182F6}';document.head.appendChild(st);
  const b=document.body, on=b.dataset.on||'고객 심사', ver=b.dataset.ver||'';
  const items=[['심사 업무',['고객 심사','XAI 분석']],['모델 검증',['공정성','모델 성능','통계 검증']]];
  const sb=items.map(([h,l],i)=>`<h6 ${i?'style="margin-top:14px"':''}>${h}</h6>`+l.map(x=>`<a class="${x===on?'on':''}">${x}</a>`).join('')).join('');
  const screens=[...document.querySelectorAll('main')];
  screens.forEach(m=>{
    const label=m.dataset.label||'';
    const on2=m.dataset.on||on;
    const sb2=items.map(([h,l],i)=>`<h6 ${i?'style="margin-top:14px"':''}>${h}</h6>`+l.map(x=>navA(x,on2)).join('')).join('');
    const shell=document.createElement('section');shell.style.position='relative';
    shell.innerHTML=`<div class="hd"><div class="logo"><img src="logos/namgung.png" alt="" style="width:34px;height:34px"><b style="color:#183062">남궁은행</b><span style="width:1px;height:22px;background:var(--line)"></span><span style="font-size:15px;font-weight:500;color:var(--ink2)">대안신용평가 심사 시스템</span></div><div class="hr"><span class="pill">교육용</span>${ver||label?`<span class="ver" style="position:static">${[ver,label].filter(Boolean).join(' · ')}</span>`:''}</div></div><div class="wrap"><div class="sb">${sb2}</div><div class="main"></div></div>`;
    m.replaceWith(shell);
    const main=shell.querySelector('.main');while(m.firstChild)main.appendChild(m.firstChild);
    main.insertAdjacentHTML('beforeend','<p class="foot">본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.</p>');
  });
  if(screens.length>1){document.querySelectorAll('section').forEach((s,i)=>{if(i)s.style.marginTop='48px'});document.body.style.background='#DDE1E6'}
})();

// ---------- 작은 차트 도우미 (SVG) ----------
const NS='http://www.w3.org/2000/svg';
function svg(w,h,inner){return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" xmlns="${NS}" font-family="Noto Sans KR, sans-serif">${inner}</svg>`}
const ROC_F=[0,.0056,.0172,.0317,.048,.0657,.0844,.1033,.1231,.1431,.1632,.1837,.2041,.2244,.2453,.2661,.2874,.3085,.3297,.3507,.3718,.3932,.4146,.4359,.4576,.4791,.5003,.5218,.5434,.565,.5865,.6082,.6297,.6515,.6731,.6947,.7166,.7383,.76,.7817,.8034,.8253,.8471,.8689,.8907,.9126,.9345,.9563,.9781,1];
const ROC_T=[0,.2274,.371,.4734,.5519,.6097,.6529,.6948,.7234,.7507,.7753,.7939,.8145,.8358,.8491,.865,.8723,.8823,.8923,.9036,.9149,.9209,.9282,.9355,.9382,.9435,.9528,.9588,.9614,.9654,.9701,.9741,.9787,.9807,.9834,.9874,.9874,.99,.9927,.9947,.996,.9967,.998,.9987,.9987,.9987,.9987,.9993,1,1];
function axes(w,h,p,xl,yl){let s='';for(let i=0;i<=5;i++){const v=i/5,x=p+v*(w-p-10),y=h-p-v*(h-p-10);s+=`<line x1="${p}" x2="${w-10}" y1="${y}" y2="${y}" stroke="#EEF0F2"/><text x="${p-6}" y="${y+4}" font-size="10" fill="#8B95A1" text-anchor="end">${v.toFixed(1)}</text><text x="${x}" y="${h-p+15}" font-size="10" fill="#8B95A1" text-anchor="middle">${v.toFixed(1)}</text>`}
  s+=`<text x="${(w+p)/2}" y="${h-4}" font-size="11" fill="#4E5968" text-anchor="middle">${xl}</text><text transform="translate(11 ${(h-p)/2}) rotate(-90)" font-size="11" fill="#4E5968" text-anchor="middle">${yl}</text>`;return s}
function path(xs,ys,w,h,p){return xs.map((x,i)=>`${i?'L':'M'}${(p+x*(w-p-10)).toFixed(1)} ${(h-p-ys[i]*(h-p-10)).toFixed(1)}`).join('')}
function roc(w,h){const p=40;const lr=ROC_T.map(t=>Math.pow(t,1.35));return svg(w,h,axes(w,h,p,'잘못 거절한 정상 고객 비율 (FPR)','걸러낸 부도 고객 비율 (TPR)')+
  `<path d="${path([0,1],[0,1],w,h,p)}" stroke="#C9CED4" stroke-dasharray="4 4" fill="none"/><path d="${path(ROC_F,lr,w,h,p)}" stroke="#B0B8C1" stroke-width="2" fill="none"/><path d="${path(ROC_F,ROC_T,w,h,p)}" stroke="#3182F6" stroke-width="3" fill="none"/>`+
  `<circle cx="${p+.1638*(w-p-10)}" cy="${h-p-.7753*(h-p-10)}" r="5" fill="#191F28" stroke="#FFF" stroke-width="2"/>`)}
function prc(w,h){const p=40,P=1504,N=20996;const r=[],pr=[];ROC_T.forEach((t,i)=>{if(i===0)return;r.push(t);pr.push(t*P/(t*P+ROC_F[i]*N))});r.unshift(0);pr.unshift(1);
  return svg(w,h,axes(w,h,p,'걸러낸 부도 고객 비율 (Recall)','거절 중 실제 부도 비율 (Precision)')+`<path d="${path([0,1],[.067,.067],w,h,p)}" stroke="#C9CED4" stroke-dasharray="4 4" fill="none"/><path d="${path(r,pr,w,h,p)}" stroke="#3182F6" stroke-width="3" fill="none"/><circle cx="${p+.7753*(w-p-10)}" cy="${h-p-.2532*(h-p-10)}" r="5" fill="#191F28" stroke="#FFF" stroke-width="2"/>`)}
function cm(big){const c=[[17557,3439,'맞게 승인','잘못 거절'],[338,1166,'놓친 부도','맞게 거절']];const tot=[20996,1504];const sz=big?1:.82;
  const cell=(v,t,i,ok)=>{const a=v/tot[i];const bg=ok?`rgba(49,130,246,${.12+a*.75})`:`rgba(240,68,82,${.10+a*.6})`;const fg=a>.5?'#FFF':'#191F28';return `<div style="background:${bg};color:${fg};border-radius:12px;padding:${14*sz}px 16px"><div style="font-size:${22*sz}px;font-weight:700">${v.toLocaleString()}</div><div style="font-size:12px;opacity:.85">${t} ${(a*100).toFixed(1)}%</div></div>`};
  return `<div style="display:grid;grid-template-columns:88px 1fr 1fr;gap:8px;align-items:center;font-size:12px;color:#8B95A1"><span></span><span>예측: 승인</span><span>예측: 거절</span><span>실제 정상<br><b style="color:#191F28">20,996명</b></span>${cell(17557,'맞게 승인',0,1)}${cell(3439,'잘못 거절',0,0)}<span>실제 부도<br><b style="color:#191F28">1,504명</b></span>${cell(338,'놓친 부도',1,0)}${cell(1166,'맞게 거절',1,1)}</div>`}
// 집단별 승인율 막대 (보정 전/후)
const FAIR={b:{'남성':79.7,'여성':79.4,'20-34세':64.0,'35-54세':73.8,'55세 이상':89.8},a:{'남성':79.5,'여성':79.0,'20-34세':79.5,'35-54세':79.7,'55세 이상':78.7}};
function abars(w,which,grp,rh){const keys=grp?grp:Object.keys(FAIR.a);const rowH=rh||34;let s='';const x0=80,x1=w-50;keys.forEach((k,i)=>{const y=i*rowH+8;
  const both=which==='both';const vb=FAIR.b[k],va=FAIR.a[k];const X=v=>x0+(v/100)*(x1-x0);
  s+=`<text x="0" y="${y+15}" font-size="12" fill="#4E5968">${k}</text>`;
  if(both){s+=`<rect x="${x0}" y="${y}" width="${X(vb)-x0}" height="9" rx="4.5" fill="#C9CED4"/><rect x="${x0}" y="${y+12}" width="${X(va)-x0}" height="9" rx="4.5" fill="#3182F6"/><text x="${X(Math.max(vb,va))+6}" y="${y+18}" font-size="11" fill="#4E5968">${vb.toFixed(1)} → <tspan font-weight="700" fill="#191F28">${va.toFixed(1)}%</tspan></text>`}
  else{const v=which==='b'?vb:va;s+=`<rect x="${x0}" y="${y+3}" width="${X(v)-x0}" height="14" rx="7" fill="${which==='b'?'#B0B8C1':'#3182F6'}"/><text x="${X(v)+6}" y="${y+15}" font-size="12" font-weight="700" fill="#191F28">${v.toFixed(1)}%</text>`}});
  return svg(w,keys.length*rowH+10,s)}
// DI 범위 막대: 0.5~1.5 축, 기준 0.8~1.25
function diRange(w,before,after,label){const x=v=>20+(v-.5)/(1.5-.5)*(w-40);let s=`<rect x="${x(.8)}" y="22" width="${x(1.25)-x(.8)}" height="14" rx="7" fill="#E5F8EF"/><line x1="20" x2="${w-20}" y1="29" y2="29" stroke="#E5E8EB" stroke-width="2"/><rect x="${x(.8)}" y="22" width="${x(1.25)-x(.8)}" height="14" rx="7" fill="#DDF5E9"/>`;
  [.5,.8,1,1.25,1.5].forEach(v=>s+=`<text x="${x(v)}" y="54" font-size="10" fill="#8B95A1" text-anchor="middle">${v}</text>`);
  if(before!=null)s+=`<circle cx="${x(before)}" cy="29" r="7" fill="#FFF" stroke="#E42939" stroke-width="2.5"/><text x="${x(before)}" y="14" font-size="11" fill="#E42939" text-anchor="middle" font-weight="700">보정 전 ${before}</text>`;
  s+=`<circle cx="${x(after)}" cy="29" r="7" fill="#3182F6" stroke="#FFF" stroke-width="2"/><text x="${x(after)}" y="${before!=null&&Math.abs(before-after)<.15?72:14}" font-size="11" fill="#3182F6" text-anchor="middle" font-weight="700">${before!=null?'보정 후 ':''}${after}</text>`;
  if(before!=null)s+=`<path d="M${x(before)-10} 29 L${x(after)+12} 29" stroke="#3182F6" stroke-width="1.5" stroke-dasharray="3 3" marker-end="url(#ar)"/>`;
  return svg(w,76,`<defs><marker id="ar" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6z" fill="#3182F6"/></marker></defs>`+s)}
// ANOVA 평균 + 95% CI 점그래프
const AN={t1:{g:['전통 데이터만','대안 데이터만','통합 (전통 + 대안)'],m:[.828,.838,.873],ci:[[.816,.840],[.836,.840],[.868,.879]],lo:.80,hi:.89,best:2},t2:{g:['Logistic Regression','XGBoost','LightGBM'],m:[.855,.882,.884],ci:[[.852,.857],[.876,.888],[.877,.890]],lo:.84,hi:.90,best:2}};
function dots(w,t){const d=AN[t];const x=v=>150+(v-d.lo)/(d.hi-d.lo)*(w-180);let s='';const ticks=[];for(let v=d.lo;v<=d.hi+1e-9;v+=.02)ticks.push(+v.toFixed(2));
  ticks.forEach(v=>s+=`<line x1="${x(v)}" x2="${x(v)}" y1="0" y2="${d.g.length*36}" stroke="#F0F2F4"/><text x="${x(v)}" y="${d.g.length*36+14}" font-size="10" fill="#8B95A1" text-anchor="middle">${v.toFixed(2)}</text>`);
  d.g.forEach((g,i)=>{const y=i*36+18;const c=i===d.best?'#3182F6':'#8B95A1';s+=`<text x="0" y="${y+4}" font-size="12" fill="#4E5968">${g}</text><line x1="${x(d.ci[i][0])}" x2="${x(d.ci[i][1])}" y1="${y}" y2="${y}" stroke="${c}" stroke-width="3" stroke-linecap="round"/><circle cx="${x(d.m[i])}" cy="${y}" r="6" fill="${c}" stroke="#FFF" stroke-width="2"/><text x="${x(d.ci[i][1])+10}" y="${y+4}" font-size="12" font-weight="700" fill="#191F28">${d.m[i].toFixed(3)}</text>`});
  return svg(w,d.g.length*36+20,s)}
const TK={t1:[['전통 vs 대안','-0.010','0.15',0],['전통 vs 통합','-0.046','< 0.001',1],['대안 vs 통합','-0.035','< 0.001',1]],t2:[['LR vs XGBoost','-0.028','< 0.001',1],['LR vs LightGBM','-0.029','< 0.001',1],['XGBoost vs LightGBM','-0.001','0.90',0]]};
function tukey(t){return `<table><tr><th>비교 (Tukey HSD)</th><th class="n">평균 차이</th><th class="n">조정 p값</th><th class="n">결과</th></tr>`+TK[t].map(r=>`<tr><td>${r[0]}</td><td class="n">${r[1]}</td><td class="n">${r[2]}</td><td class="n"><span class="badge ${r[3]?'b-ok':'b-g'}">${r[3]?'유의함':'차이 없음'}</span></td></tr>`).join('')+`</table>`}
function anovaTable(){return `<table><tr><th>검증</th><th class="n">F값</th><th class="n">p값</th><th class="n">η² (효과 크기)</th><th class="n">결과 (α = 0.05)</th></tr><tr><td><b>검증 1</b> 피처 구성별 <span class="muted">One-way ANOVA</span></td><td class="n">39.2</td><td class="n">&lt; 0.001</td><td class="n">0.47 Large</td><td class="n"><span class="badge b-ok">유의함</span></td></tr><tr><td><b>검증 2</b> 알고리즘별 <span class="muted">One-way ANOVA</span></td><td class="n">51.2</td><td class="n">&lt; 0.001</td><td class="n">0.79 Large</td><td class="n"><span class="badge b-ok">유의함</span></td></tr></table>`}
function fill(){document.querySelectorAll('[data-chart]').forEach(el=>{el.innerHTML=new Function('return '+el.dataset.chart)()})}
