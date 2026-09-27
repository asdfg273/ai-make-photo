from core.identity.base import IdentityProvider, IdentityProviderBase, IdentitySpec
from core.identity.passthrough import PassthroughProvider
from core.identity.registry import (
    available, compliance_matrix, current_spec, get, ipa_input,
    register, register_defaults, resolver, set_provider,
)

__all__ = [
    "IdentityProvider", "IdentityProviderBase", "IdentitySpec",
    "PassthroughProvider",
    "register", "register_defaults", "get", "available",
    "ipa_input", "resolver", "set_provider", "current_spec",
    "compliance_matrix",
]