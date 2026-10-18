"""ComfyUI discovers this entry point; Forge discovers scripts/ instead."""
from .neo_sigma.comfy import SigmaRescaleNeo

NODE_CLASS_MAPPINGS = {"SigmaRescaleNeo": SigmaRescaleNeo}
NODE_DISPLAY_NAME_MAPPINGS = {"SigmaRescaleNeo": "Sigma Rescale Neo"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
