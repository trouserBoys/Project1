from collections import defaultdict
from nltk import Tree
import nltk

# Ensure nltk resources are available
nltk.download('punkt')

# ---------------------------
# CYK PARSER (Non-Probabilistic)
# ---------------------------
def cyk_parser(grammar, tokens):
    n = len(tokens)
    table = [[set() for _ in range(n)] for _ in range(n)]
    back = [[defaultdict(list) for _ in range(n)] for _ in range(n)]

    # Inverse grammar mapping: RHS -> LHS
    rhs_to_lhs = defaultdict(set)
    for lhs, rules in grammar.items():
        for rhs in rules:
            rhs_to_lhs[tuple(rhs)].add(lhs)

    # Fill diagonals
    for i, token in enumerate(tokens):
        for lhs in rhs_to_lhs.get((token,), []):
            table[i][i].add(lhs)

    # Fill upper triangle
    for l in range(2, n+1):
        for i in range(n - l + 1):
            j = i + l - 1
            for k in range(i, j):
                for B in table[i][k]:
                    for C in table[k+1][j]:
                        for A in rhs_to_lhs.get((B, C), []):
                            table[i][j].add(A)
                            back[i][j][A].append((k, B, C))

    # Recursive tree building
    def build_tree(i, j, symbol):
        if i == j:
            return (symbol, tokens[i])
        for k, B, C in back[i][j].get(symbol, []):
            left = build_tree(i, k, B)
            right = build_tree(k+1, j, C)
            return (symbol, left, right)

    if 'S' in table[0][n-1]:
        return build_tree(0, n-1, 'S')
    else:
        return None

# ---------------------------
# EXAMPLE USAGE
# ---------------------------
tokens = ['the', 'cat', 'chased', 'the', 'mouse']

# CYK grammar
grammar = {
    'S': [['NP', 'VP']],
    'VP': [['V', 'NP']],
    'NP': [['Det', 'N']],
    'Det': [['the']],
    'N': [['cat'], ['mouse']],
    'V': [['chased']]
}

# Run CYK
cyk_tree = cyk_parser(grammar, tokens)
if cyk_tree:
    print("CYK Parse Tree (tuple):")
    print(cyk_tree)
    nltk_tree = tuple_to_nltk_tree(cyk_tree)
    nltk_tree.pretty_print()
else:
    print("No parse tree found.Invalid input sentence")