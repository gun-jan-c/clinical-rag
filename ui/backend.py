"""The backend every page calls: services.mock when USE_MOCK=1 (the default, free), else services.api.
Pages import only this and clinical_rag.schemas (ProjectSpec.md section 6)."""

import os

from dotenv import load_dotenv

load_dotenv()

USING_MOCK = os.environ.get("USE_MOCK", "1").strip() == "1"

if USING_MOCK:
    from clinical_rag.services import mock as backend
else:
    from clinical_rag.services import api as backend

__all__ = ["USING_MOCK", "backend"]
