"""Curvas por presupuesto real, dispersión entre semillas y resultados verificables."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    out=Path('informe/figuras'); out.mkdir(parents=True,exist_ok=True)
    evaluations=[]
    grouped={}
    for root in [Path('subtareas'),Path('global')]:
        for path in root.rglob('episodes.monitor.csv'):
            frame=pd.read_csv(path,comment='#')
            if frame.empty:
                continue
            group=path.parent.parent
            grouped.setdefault(group,[]).append(frame)
        for path in root.rglob('evaluation.json'):
            data=json.loads(path.read_text())
            evaluations.append(dict(path=str(path),**{k:v for k,v in data.items() if k not in ('rows','wilson95')}))
    for group,frames in grouped.items():
        # Align by real completed-episode interaction count, no endpoint extrapolation.
        end=min(f['l'].sum() for f in frames)
        start=max(f['l'].iloc[0] for f in frames)
        if end<=start:
            continue
        grid=np.linspace(start,end,100)
        values=[]
        for frame in frames:
            metric='base_return' if 'base_return' in frame else 'r'
            values.append(np.interp(grid,frame['l'].cumsum(),frame[metric].rolling(20,min_periods=1).mean()))
        values=np.array(values)
        plt.figure(figsize=(8,4))
        plt.plot(grid,values.mean(0),label=f'media ({len(frames)} semillas)')
        if len(frames)>=3:
            plt.fill_between(grid,values.mean(0)-values.std(0),values.mean(0)+values.std(0),alpha=.2,label='±1 sigma entre semillas')
        plt.xlabel('Decisiones de entrenamiento completadas'); plt.ylabel('Retorno base suavizado')
        plt.title(str(group)); plt.legend(); plt.tight_layout()
        plt.savefig(out/(str(group).replace('/','_').replace('\\','_')+'.png'),dpi=160); plt.close()
    if evaluations:
        pd.DataFrame(evaluations).to_csv('informe/evaluaciones.csv',index=False)
    print(f'{len(evaluations)} evaluaciones, {len(grouped)} grupos de curvas')


if __name__=='__main__':
    main()
