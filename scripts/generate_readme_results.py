from pathlib import Path
import pandas as pd
p=Path('README.md'); text=p.read_text()
start='<!-- AUTO_RESULTS_START -->'; end='<!-- AUTO_RESULTS_END -->'
parts=["## Results\n\nResults are generated only after real-data execution.\n"]
for f in ['results/forecast_metrics.csv','results/anomaly_metrics.csv','results/capacity_metrics.csv']:
    fp=Path(f)
    if fp.exists(): parts.append(f"### {fp.stem}\n\n"+pd.read_csv(fp).to_markdown(index=False)+"\n")
block=start+'\n'+'\n'.join(parts)+end
if start in text and end in text: text=text.split(start)[0]+block+text.split(end)[1]
p.write_text(text)
