
import regex as re
def _clean_text(text: str) -> str:
    """Same preprocessing as your training pipeline."""
    return re.sub(
        r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*',
        'LINK', str(text), flags=re.MULTILINE
    )
