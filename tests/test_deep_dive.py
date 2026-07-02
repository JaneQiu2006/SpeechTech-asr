from scripts.analyze_deep_dive_errors import align_words


def test_word_alignment_counts_substitution_deletion_and_insertion():
    substitution = align_words("the cat".split(), "the dog".split())
    deletion = align_words("the small cat".split(), "the cat".split())
    insertion = align_words("the cat".split(), "the small cat".split())
    assert [item[0] for item in substitution].count("substitution") == 1
    assert [item[0] for item in deletion].count("deletion") == 1
    assert [item[0] for item in insertion].count("insertion") == 1
