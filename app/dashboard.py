import streamlit as st, pandas as pd
st.set_page_config(page_title='Network Traffic Capacity Planning',layout='wide')
st.title('Network Traffic Forecasting & Anomaly Detection')
try: df=pd.read_parquet('data/processed/traffic.parquet')
except Exception: st.warning('Run `make data` with the real Telecom Italia files first.'); st.stop()
cell=st.selectbox('Cell',sorted(df.cell_id.unique())); g=df[df.cell_id==cell].set_index('timestamp')
st.line_chart(g[['internet']]); st.caption('Capacity proxy and model forecasts appear after evaluation outputs are generated.')
