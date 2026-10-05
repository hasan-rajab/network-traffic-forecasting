"""Exercise the served dashboard with the actual immutable data bundle."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.serve import validate_artifacts


def main() -> None:
    root = Path(os.environ.get("NETWORK_ARTIFACT_DIR", "deployment/artifacts")).resolve()
    scope = validate_artifacts(root)
    os.environ["NETWORK_ARTIFACT_DIR"] = str(root)
    dashboard = Path(__file__).resolve().parents[1] / "app/dashboard.py"
    app = AppTest.from_file(str(dashboard)).run(timeout=45)
    assert not app.exception, [error.value for error in app.exception]
    assert len(app.selectbox) == 1
    cell_options = list(app.selectbox[0].options)
    assert len(cell_options) == scope["selected_cells"]
    required = {
        "Forecast comparison and uncertainty — all evaluated cells",
        "Anomaly event denominators — held-out injected labels",
        "SQL daily load, segmentation and coverage",
        "SQL cell trends — prior-only 24-hour history",
        "History",
        "+24h rolling-origin forecast",
        "Injected-anomaly evaluation view",
        "Capacity-proxy warnings",
    }
    tested_cells = []
    for index in sorted({0, len(cell_options) - 1}):
        app.selectbox[0].select_index(index).run(timeout=45)
        assert not app.exception, [error.value for error in app.exception]
        assert required.issubset({section.value for section in app.subheader})
        assert len(app.dataframe) == 6
        trends = app.dataframe[3].value
        assert not trends.empty
        assert trends["cell_id"].nunique() == 1
        assert int(trends["cell_id"].iloc[0]) == int(cell_options[index])
        tested_cells.append(cell_options[index])
    print(json.dumps({"dashboard_rendered": True, "exceptions": 0,
                      "selected_cells": len(cell_options), "tested_cells": tested_cells,
                      "sql_trends_follow_selection": True, "dataframes": len(app.dataframe)}))


if __name__ == "__main__":
    main()
