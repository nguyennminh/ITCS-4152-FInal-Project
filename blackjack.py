"""Pure hand scoring: no camera or third-party dependencies."""
from dataclasses import dataclass

RANKS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K')


def normalize(rank):
    rank = str(rank).strip().upper()
    if rank == 'T':
        rank = '10'
    if rank not in RANKS:
        raise ValueError(f'Invalid rank: {rank!r}. Use A, 2-10, J, Q, K.')
    return rank


def card_value(rank):
    rank = normalize(rank)
    return 11 if rank == 'A' else 10 if rank in ('J', 'Q', 'K') else int(rank)


@dataclass(frozen=True)
class Hand:
    total: int
    soft: bool
    bust: bool
    blackjack: bool


def evaluate_hand(cards):
    ranks = [normalize(card) for card in cards]
    total = sum(card_value(card) for card in ranks)
    high_aces = ranks.count('A')
    while total > 21 and high_aces:
        total -= 10
        high_aces -= 1
    return Hand(total, high_aces > 0, total > 21,
                len(ranks) == 2 and total == 21)
