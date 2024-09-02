import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # Initialize probabilities dictionary
    probs = dict.fromkeys(corpus, 0)  # Each page in the corpus will have an initial probability of 0

    # Get the links from the current page
    links = corpus[page]

    # Calculate the probabilities for each page
    for p in probs:
        # Probability of choosing a link from the current page
        if p in links:
            probs[p] += damping_factor / len(links)
        # Probability of choosing a random page
        probs[p] += (1 - damping_factor) / len(corpus)

    return probs

    raise NotImplementedError



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Initialize PageRank values
    pagerank = dict.fromkeys(corpus, 0)

    # Choose a random page as the starting page
    page = random.choice(list(corpus))

    # Loop for n iterations
    for i in range(n):
        # Increment the PageRank value for the current page
        pagerank[page] += 1

        # Choose the next page based on the transition model
        model = transition_model(corpus, page, damping_factor)
        page = random.choices(list(model), weights=list(model.values()))[0]

    # Normalize the PageRank values
    for page in pagerank:
        pagerank[page] /= n

    # Return the PageRank dictionary
    return pagerank

    raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Initialize PageRank values
    n = len(corpus)
    pagerank = {page: 1 / n for page in corpus}

    # Repeat until convergence
    while True:
        # Keep track of previous values
        old_pagerank = pagerank.copy()

        # Update PageRank values based on the formula
        for page in corpus:
            # Calculate the weighted sum of incoming links
            total = 0
            for p in corpus:
                if page in corpus[p]:
                    total += pagerank[p] / len(corpus[p])

            # Apply the damping factor and the random jump factor
            pagerank[page] = (1 - damping_factor) / n + damping_factor * total

        # Check if the values converged within the threshold
        if all(abs(old_pagerank[page] - pagerank[page]) < 0.0001 for page in corpus):
            break

    # Return the final PageRank dictionary
    return pagerank

    raise NotImplementedError


if __name__ == "__main__":
    main()