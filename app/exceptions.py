class AppError(Exception):
    """Base class for expected application errors (mapped to HTTP responses)."""
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ConfigurationError(AppError):
    status_code = 503
    code = "configuration_error"


class ProviderError(AppError):
    status_code = 502
    code = "provider_error"


class RepositoryError(AppError):
    status_code = 503
    code = "database_error"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class IngestionError(AppError):
    status_code = 400
    code = "ingestion_error"
