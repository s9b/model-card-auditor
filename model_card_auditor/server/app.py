from openenv.core.env_server import create_fastapi_app
from .environment import ModelCardAuditEnvironment
from ..models import ModelCardAction, ModelCardObservation

# create_fastapi_app auto-generates endpoints: /ws, /reset, /step, /state, /health, /docs
app = create_fastapi_app(ModelCardAuditEnvironment, ModelCardAction, ModelCardObservation)
