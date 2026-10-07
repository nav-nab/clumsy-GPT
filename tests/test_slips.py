import torch

from clumsygpt.slips import SlipMaker
from clumsygpt.vocab import LOWER, TITLE, VocabInfo

(SALT, PEPPER, SUGAR, DOCTOR, NURSE, HOSPITAL, COMMA, THE, SALT_CAP, SALTS, PEPPER_CAP) = range(11)
NAMES = ["salt", "pepper", "sugar", "doctor", "nurse", "hospital", ",", "the", "Salt", "salts", "Pepper"]

# Two clusters: kitchen words and medical words.
EMBEDDINGS = torch.tensor(
    [
        [1.0, 0.1, 0.0],  # salt
        [0.9, 0.2, 0.0],  # pepper
        [0.8, 0.0, 0.1],  # sugar
        [0.1, 1.0, 0.0],  # doctor
        [0.2, 0.9, 0.1],  # nurse
        [0.0, 0.8, 0.2],  # hospital
        [0.0, 0.0, 0.5],  # ,
        [0.1, 0.1, 1.0],  # the
        [1.0, 0.1, 0.0],  # Salt   (case variant of salt)
        [1.0, 0.15, 0.0],  # salts  (shares a stem with salt)
        [0.9, 0.2, 0.0],  # Pepper (title case)
    ]
)


def make_maker():
    n = len(NAMES)
    word_start = torch.ones(n, dtype=torch.bool)
    word_start[COMMA] = False
    content = word_start.clone()
    content[THE] = False
    word_index = torch.tensor([0, 1, 2, 3, 4, 5, -1, 6, 0, 7, 1])
    case = torch.full((n,), LOWER)
    case[COMMA] = -1
    case[SALT_CAP] = case[PEPPER_CAP] = TITLE
    words = ["" if i == COMMA else NAMES[i].lower() for i in range(n)]
    pool = word_start.clone()
    vocab = VocabInfo(word_start, pool, content, word_index, case, words)
    return SlipMaker(EMBEDDINGS, vocab)


def best(maker, discarded, previous, weights):
    """Brute-force highest-scoring allowed candidate, to compare against choose()."""
    allowed = maker.allowed(discarded, previous).nonzero().squeeze(1).tolist()
    anchors = {discarded: weights[0], previous: weights[1]}

    def score(i):
        return sum(w * float(maker.emb[i] @ maker.emb[a]) for a, w in anchors.items())

    return max(allowed, key=score)


def test_eligibility_rules():
    m = make_maker()
    # Substitution (s = 0): only the discarded word must be a content word.
    assert m.is_eligible(SALT, DOCTOR, 0.0)
    assert m.is_eligible(SALT, COMMA, 0.0)
    assert not m.is_eligible(THE, DOCTOR, 0.0)
    assert not m.is_eligible(COMMA, DOCTOR, 0.0)
    # Intrusion (s = 1): only the previous word must be a content word.
    assert m.is_eligible(THE, DOCTOR, 1.0)
    assert not m.is_eligible(THE, COMMA, 1.0)
    assert not m.is_eligible(THE, THE, 1.0)
    # The token must start a word whatever the slip type.
    assert not m.is_eligible(COMMA, DOCTOR, 1.0)
    # In between, both words must be content words.
    assert m.is_eligible(SALT, DOCTOR, 0.5)
    assert not m.is_eligible(THE, DOCTOR, 0.5)
    assert not m.is_eligible(SALT, THE, 0.5)


def test_allowed_candidates():
    m = make_maker()
    allowed = set(m.allowed(SALT, DOCTOR).nonzero().squeeze(1).tolist())
    # Excluded: both anchors, the case and stem variants of salt, title case, the function word, the comma.
    assert allowed == {PEPPER, SUGAR, NURSE, HOSPITAL}


def test_title_case_is_kept():
    m = make_maker()
    allowed = set(m.allowed(SALT_CAP, DOCTOR).nonzero().squeeze(1).tolist())
    assert allowed == {PEPPER_CAP}


def test_substitution_picks_nearest_to_the_discarded_word():
    m = make_maker()
    logits = torch.zeros(len(NAMES))
    expected = best(m, SALT, DOCTOR, (1.0, 0.0))
    assert NAMES[expected] == "pepper"
    for _ in range(20):
        assert m.choose(logits, SALT, DOCTOR, 0.0, 0.001, 100) == expected


def test_intrusion_picks_nearest_to_the_previous_word():
    m = make_maker()
    logits = torch.zeros(len(NAMES))
    expected = best(m, SALT, DOCTOR, (0.0, 1.0))
    assert expected in (NURSE, HOSPITAL)
    for _ in range(20):
        assert m.choose(logits, SALT, DOCTOR, 1.0, 0.001, 100) == expected


def test_blend_uses_both_words():
    m = make_maker()
    logits = torch.zeros(len(NAMES))
    expected = best(m, SALT, DOCTOR, (0.5, 0.5))
    for _ in range(20):
        assert m.choose(logits, SALT, DOCTOR, 0.5, 0.001, 100) == expected


def test_slip_only_comes_from_the_allowed_words():
    m = make_maker()
    logits = torch.zeros(len(NAMES))
    for s in (0.0, 0.5, 1.0):
        allowed = set(m.allowed(SALT, DOCTOR).nonzero().squeeze(1).tolist())
        for _ in range(100):
            assert m.choose(logits, SALT, DOCTOR, s, 1.0, 100) in allowed


def test_slip_comes_from_the_models_top_k():
    m = make_maker()
    logits = torch.zeros(len(NAMES))
    logits[SUGAR] = 5.0
    logits[HOSPITAL] = 4.0
    # With k = 2 the shortlist is {sugar, hospital}, whatever the similarities say.
    for _ in range(50):
        assert m.choose(logits, SALT, DOCTOR, 0.0, 1.0, 2) in (SUGAR, HOSPITAL)


def test_low_looseness_is_sharper_than_high_looseness():
    m = make_maker()
    logits = torch.zeros(len(NAMES))
    torch.manual_seed(0)
    low = {m.choose(logits, SALT, DOCTOR, 0.0, 0.001, 100) for _ in range(50)}
    high = {m.choose(logits, SALT, DOCTOR, 0.0, 5.0, 100) for _ in range(200)}
    assert len(low) == 1
    assert len(high) > 2


def test_empty_shortlist_returns_the_discarded_token():
    m = make_maker()
    m.vocab.content[:] = False
    assert m.choose(torch.zeros(len(NAMES)), SALT, DOCTOR, 0.0, 1.0, 10) == SALT


def test_irregular_inflections_are_excluded():
    # "knew" and "tell"/"told" are different spellings of the same word. They must not be slips.
    m = make_maker()
    m.vocab.words[PEPPER] = "know"
    m.vocab.words[SUGAR] = "knew"
    m.vocab.words[NURSE] = "told"
    m.vocab.words[HOSPITAL] = "tell"
    allowed = set(m.allowed(PEPPER, DOCTOR).nonzero().squeeze(1).tolist())
    assert SUGAR not in allowed
    allowed = set(m.allowed(NURSE, DOCTOR).nonzero().squeeze(1).tolist())
    assert HOSPITAL not in allowed
