class PromptNotFoundError(Exception):
    """Raised when a prompt slug does not exist (404)."""

    def __init__(self, slug: str) -> None:
        self.slug = slug
        super().__init__(f'Prompt "{slug}" not found')
