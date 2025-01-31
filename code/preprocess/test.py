database = [
    {"book_name": "A", "author_name": "author_a", "year": 2011},
    {"book_name": "AA", "author_name": "author_aa", "year": 2022},
    {"book_name": "B", "author_name": "author_b", "year": 2021},
]


def search_for_book(book_name: str, author_name: str, year: int) -> str:
    """
    Search for a book in the database.
    """
    return f"Searching for book: {book_name} by {author_name}"
