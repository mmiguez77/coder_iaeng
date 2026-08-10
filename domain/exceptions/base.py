class LLMException(Exception):
    """Excepción base para todos los errores de la capa de clientes de LLM."""
    def __init__(self, message: str, provider: str = None):
        super().__init__(message)
        self.message = message
        self.provider = provider

    def __str__(self) -> str:
        provider_prefix = f"[{self.provider.upper()}] " if self.provider else ""
        return f"{provider_prefix}{self.message}"


class LLMTimeoutError(LLMException):
    """Excepción lanzada cuando una petición al LLM excede el tiempo límite establecido."""
    pass


class LLMConnectionError(LLMException):
    """Excepción lanzada por fallos de conectividad o de red con la API del proveedor."""
    pass


class LLMAPIError(LLMException):
    """Excepción lanzada cuando la API del proveedor retorna un estado de error (4xx/5xx)."""
    def __init__(self, message: str, status_code: int = None, provider: str = None):
        super().__init__(message, provider=provider)
        self.status_code = status_code

    def __str__(self) -> str:
        status_suffix = f" (Status Code: {self.status_code})" if self.status_code else ""
        return f"{super().__str__()}{status_suffix}"


class LLMValidationError(LLMException):
    """Excepción lanzada por fallos de validación en los parámetros de entrada o en la respuesta."""
    pass
