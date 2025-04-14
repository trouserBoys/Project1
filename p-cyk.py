# ---------------------------
# PROBABILISTIC CYK PARSER
# ---------------------------
def pcyk_parser(grammar, tokens):
    n = len(tokens)
    table = [[defaultdict(float) for _ in range(n)] for _ in range(n)]
    back = [[{} for _ in range(n)] for _ in range(n)]

    for i, token in enumerate(tokens):
        for lhs in grammar:
            for rhs, prob in grammar[lhs]:
                if len(rhs) == 1 and rhs[0] == token:
                    if prob > table[i][i][lhs]:
                        table[i][i][lhs] = prob
                        back[i][i][lhs] = token

    for l in range(2, n+1):
        for i in range(n - l + 1):
            j = i + l - 1
            for k in range(i, j):
                for A in grammar:
                    for (rhs, prob) in grammar[A]:
                        if len(rhs) == 2:
                            B, C = rhs
                            prob_B = table[i][k].get(B, 0)
                            prob_C = table[k+1][j].get(C, 0)
                            if prob_B and prob_C:
                                new_prob = prob * prob_B * prob_C
                                if new_prob > table[i][j].get(A, 0):
                                    table[i][j][A] = new_prob
                                    back[i][j][A] = (k, B, C)

    def build_tree(i, j, symbol):
        if i == j:
            return (symbol, back[i][j][symbol])
        k, B, C = back[i][j][symbol]
        return (symbol, build_tree(i, k, B), build_tree(k+1, j, C))

    if 'S' in table[0][n-1]:
        return build_tree(0, n-1, 'S')
    else:
        return None

# ---------------------------
# CONVERT TO nltk.Tree & DISPLAY
# ---------------------------
def tuple_to_nltk_tree(tree_tuple):
    if isinstance(tree_tuple, tuple):
        label = tree_tuple[0]
        children = [tuple_to_nltk_tree(child) for child in tree_tuple[1:]]
        return Tree(label, children)
    else:
        return tree_tuple

# PCYK grammar
p_grammar = {
    'S': [ (['NP', 'VP'], 1.0) ],
    'VP': [ (['V', 'NP'], 1.0) ],
    'NP': [ (['Det', 'N'], 1.0) ],
    'Det': [ (['the'], 1.0) ],
    'N': [ (['cat'], 0.5), (['mouse'], 0.5) ],
    'V': [ (['chased'], 1.0) ]
}

# Run PCYK
pcyk_tree = pcyk_parser(p_grammar, tokens)
if pcyk_tree:
    print("\nPCYK Parse Tree (tuple)- returns the most probable parse tree:")
    print(pcyk_tree)
    nltk_tree = tuple_to_nltk_tree(pcyk_tree)
    nltk_tree.pretty_print()
else:
    print("No parse tree found.Invalid input sentence")

from collections import defaultdict


def extract_rules_from_tree(tree):
    rules = []
    if len(tree) == 2 and isinstance(tree[1], str):  # Terminal rule
        lhs, word = tree
        rules.append((lhs, (word,)))
    elif len(tree) == 3:
        lhs, left, right = tree
        rules.append((lhs, (left[0], right[0])))
        rules += extract_rules_from_tree(left)
        rules += extract_rules_from_tree(right)
    return rules

def compute_rule_probabilities(trees):
    rule_counts = defaultdict(int)
    lhs_counts = defaultdict(int)

    for tree in trees:
        rules = extract_rules_from_tree(tree)
        for lhs, rhs in rules:
            rule_counts[(lhs, rhs)] += 1
            lhs_counts[lhs] += 1

    rule_probs = {
        (lhs, rhs): count / lhs_counts[lhs]
        for (lhs, rhs), count in rule_counts.items()
    }

    return rule_probs

def collect_leaves(tree):
    if len(tree) == 2 and isinstance(tree[1], str):
        return [tree[1]]
    elif len(tree) == 3:
        return collect_leaves(tree[1]) + collect_leaves(tree[2])
    return []

def validate_and_score_tree(tree, sentence, rule_probs):
    leaves = collect_leaves(tree)
    matches = (leaves == sentence)

    return matches

def compute_tree_probability(tree, rule_probs):
    if len(tree) == 2 and isinstance(tree[1], str):
        lhs, word = tree
        rule = (lhs, (word,))
        if rule not in rule_probs:
            raise ValueError(f"Unknown rule: {lhs} -> '{word}'")
        return rule_probs[rule]
    elif len(tree) == 3:
        lhs, left, right = tree
        rule = (lhs, (left[0], right[0]))
        if rule not in rule_probs:
            raise ValueError(f"Unknown rule: {lhs} -> {left[0]} {right[0]}")
        left_prob = compute_tree_probability(left, rule_probs)
        right_prob = compute_tree_probability(right, rule_probs)
        return rule_probs[rule] * left_prob * right_prob
    else:
        raise ValueError("Invalid tree format")

training_trees = [
    ('S', ('NP', 'she'), ('VP', ('V', 'eats'), ('NP', 'fish'))),
    ('S', ('NP', 'fish'), ('VP', 'eats')),
    ('S', ('NP', 'she'), ('VP', 'eats')),
    ('S', ('NP', 'fish'), ('VP', ('V', 'eats'), ('NP', 'fish')))
]

rule_probs = compute_rule_probabilities(training_trees)

sentence = ['she', 'eats', 'fish']
test_tree = (
    'S',
    ('NP', 'she'),
    ('VP',
        ('V', 'eats'),
        ('NP', 'fish')
    )
)

is_valid = validate_and_score_tree(test_tree, sentence, rule_probs)

print("Sentence Valid for Tree?", is_valid)
print("\nLearned Grammar Rules with Probabilities:")
for (lhs, rhs), p in rule_probs.items():
    print(f"{lhs} -> {' '.join(rhs)} [{p:.3f}]")