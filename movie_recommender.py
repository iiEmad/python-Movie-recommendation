"""
Simple Movie Recommendation System (content-based filtering)
-------------------------------------------------------------
No external libraries needed. Run:  python movie_recommender.py

How it works:
1. Each movie is described by genres + a few keywords.
2. Text is converted into TF-IDF vectors (rare words get more weight).
3. Movies are compared with cosine similarity.
4. Two ways to get recommendations:
   - "Similar to a movie I like"
   - "Based on my ratings" (builds a taste profile from your ratings)
"""

import math
import re
from collections import Counter

# ---------------------------------------------------------------- Dataset
MOVIES = [
    ("The Dark Knight", "action crime thriller", "batman joker gotham vigilante hero"),
    ("Inception", "action sci-fi thriller", "dreams heist mind subconscious"),
    ("Interstellar", "sci-fi drama adventure", "space wormhole time father astronaut"),
    ("The Matrix", "action sci-fi", "simulation hacker reality machines chosen"),
    ("Titanic", "romance drama", "ship love disaster ocean tragedy"),
    ("The Notebook", "romance drama", "love memory couple letters"),
    ("Toy Story", "animation comedy family", "toys friendship adventure cowboy"),
    ("Finding Nemo", "animation family adventure", "fish ocean father journey"),
    ("The Lion King", "animation family drama", "lion kingdom father revenge"),
    ("Avengers: Endgame", "action sci-fi adventure", "superhero time travel team battle"),
    ("Iron Man", "action sci-fi", "superhero billionaire suit technology"),
    ("The Godfather", "crime drama", "mafia family power loyalty"),
    ("Pulp Fiction", "crime thriller comedy", "hitmen gangsters violence nonlinear"),
    ("The Shawshank Redemption", "drama crime", "prison hope friendship escape"),
    ("Forrest Gump", "drama romance comedy", "journey history love simple"),
    ("Jurassic Park", "sci-fi adventure thriller", "dinosaurs island park survival"),
    ("Star Wars", "sci-fi adventure action", "space rebels empire force galaxy"),
    ("The Conjuring", "horror thriller", "haunted house demons paranormal"),
    ("A Quiet Place", "horror thriller sci-fi", "silence monsters family survival"),
    ("Superbad", "comedy", "friends high school party teenagers"),
    ("The Hangover", "comedy", "friends vegas wedding party chaos"),
    ("3 Idiots", "comedy drama", "college friends engineering education"),
    ("Dangal", "drama sports", "wrestling father daughters india"),
    ("Mad Max: Fury Road", "action adventure sci-fi", "desert chase survival post-apocalyptic"),
]

STOPWORDS = {"a", "an", "the", "of", "and", "in", "to"}


# ---------------------------------------------------------------- Helpers
def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9\-]+", text.lower()) if w not in STOPWORDS]


def build_tfidf(movies):
    """Return {title: {term: tfidf_weight}} for every movie."""
    docs = {}
    for title, genres, keywords in movies:
        # Genres are repeated so they count more than keywords.
        docs[title] = tokenize(genres) * 2 + tokenize(keywords)

    n_docs = len(docs)
    df = Counter(term for tokens in docs.values() for term in set(tokens))
    idf = {t: math.log((1 + n_docs) / (1 + c)) + 1 for t, c in df.items()}

    vectors = {}
    for title, tokens in docs.items():
        tf = Counter(tokens)
        vectors[title] = {t: (c / len(tokens)) * idf[t] for t, c in tf.items()}
    return vectors


def cosine(a, b):
    dot = sum(a[t] * b[t] for t in a if t in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------- Recommender
class MovieRecommender:
    def __init__(self, movies):
        self.titles = [m[0] for m in movies]
        self.vectors = build_tfidf(movies)
        self.lookup = {t.lower(): t for t in self.titles}

    def find_title(self, query):
        """Case-insensitive exact or partial title match."""
        q = query.strip().lower()
        if q in self.lookup:
            return self.lookup[q]
        for low, real in self.lookup.items():
            if q and q in low:
                return real
        return None

    def similar_to(self, title, n=5):
        base = self.vectors[title]
        scores = [
            (other, cosine(base, vec))
            for other, vec in self.vectors.items()
            if other != title
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:n]

    def for_user(self, ratings, n=5):
        """ratings: {title: 1-5}. Builds a taste profile and scores unseen movies."""
        if not ratings:
            return []
        avg = sum(ratings.values()) / len(ratings)
        profile = Counter()
        for title, rating in ratings.items():
            weight = rating - avg if len(ratings) > 1 else rating
            for term, val in self.vectors[title].items():
                profile[term] += weight * val

        scores = [
            (t, cosine(profile, vec))
            for t, vec in self.vectors.items()
            if t not in ratings
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:n]


# ---------------------------------------------------------------- CLI
def show(results):
    if not results:
        print("  No recommendations found.")
    for i, (title, score) in enumerate(results, 1):
        print(f"  {i}. {title}  (match: {score:.0%})")


def main():
    rec = MovieRecommender(MOVIES)
    ratings = {}

    while True:
        print("\n=== Movie Recommender ===")
        print("1. List movies")
        print("2. Find movies similar to one I like")
        print("3. Rate a movie")
        print("4. Recommend based on my ratings")
        print("5. Quit")
        choice = input("Choose (1-5): ").strip()

        if choice == "1":
            for t in sorted(rec.titles):
                print("  -", t)

        elif choice == "2":
            title = rec.find_title(input("Movie title: "))
            if title:
                print(f"\nBecause you like '{title}':")
                show(rec.similar_to(title))
            else:
                print("  Movie not found. Use option 1 to see the list.")

        elif choice == "3":
            title = rec.find_title(input("Movie title: "))
            if not title:
                print("  Movie not found.")
                continue
            try:
                r = int(input(f"Your rating for '{title}' (1-5): "))
                if 1 <= r <= 5:
                    ratings[title] = r
                    print("  Saved.")
                else:
                    print("  Rating must be between 1 and 5.")
            except ValueError:
                print("  Please enter a number.")

        elif choice == "4":
            if len(ratings) < 2:
                print("  Please rate at least 2 movies first (option 3).")
            else:
                print("\nRecommended for you:")
                show(rec.for_user(ratings))

        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
