from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt

def main():
    df=pd.read_parquet('data/processed/traffic.parquet')
    Path('figures').mkdir(exist_ok=True); Path('results').mkdir(exist_ok=True)
    daily=df.set_index('timestamp').groupby('cell_id')['internet'].resample('1D').sum().reset_index()
    agg=daily.groupby('timestamp')['internet'].mean()
    fig,ax=plt.subplots(figsize=(10,4)); agg.plot(ax=ax); ax.set_title('Mean daily internet activity across selected cells'); fig.tight_layout(); fig.savefig('figures/daily_seasonality.png',dpi=160); plt.close(fig)
    hourly=df.assign(hour=df.timestamp.dt.hour).groupby('hour')['internet'].mean()
    fig,ax=plt.subplots(figsize=(8,4)); hourly.plot(ax=ax); ax.set_title('Hourly seasonality'); fig.tight_layout(); fig.savefig('figures/hourly_seasonality.png',dpi=160); plt.close(fig)
    dow=df.assign(dow=df.timestamp.dt.dayofweek).groupby('dow')['internet'].mean()
    fig,ax=plt.subplots(figsize=(8,4)); dow.plot(ax=ax); ax.set_title('Weekly seasonality'); fig.tight_layout(); fig.savefig('figures/weekly_seasonality.png',dpi=160); plt.close(fig)
    df.groupby('cell_id')['internet'].agg(['count','mean','std','min','median','max']).to_csv('results/cell_distribution.csv')
    print({'rows':len(df),'cells':df.cell_id.nunique(),'start':str(df.timestamp.min()),'end':str(df.timestamp.max())})
if __name__=='__main__': main()
