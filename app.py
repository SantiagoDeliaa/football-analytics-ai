from __future__ import annotations

import importlib
import sys

LEGACY_APP_MODULE = "legacy.streamlit.app"

if LEGACY_APP_MODULE in sys.modules:
    legacy_app = importlib.reload(sys.modules[LEGACY_APP_MODULE])
else:
    legacy_app = importlib.import_module(LEGACY_APP_MODULE)

legacy_app.main()
