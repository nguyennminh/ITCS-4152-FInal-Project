"""Total-dependent basic strategy for six decks, S17, DAS, no surrender.

Assume American peek rules; dealer blackjack has been ruled out before play.
Double on any initial two cards. Split recommendation only; no split-hand play.
Reference: https://wizardofodds.com/games/blackjack/strategy/4-decks/
"""
from blackjack import card_value, evaluate_hand, normalize


def recommend(cards, dealer):
    cards = [normalize(c) for c in cards]
    d = card_value(dealer)
    if len(cards) < 2:
        return 'WAIT'
    hand = evaluate_hand(cards)
    if hand.bust:
        return 'BUST'
    if hand.blackjack:
        return 'BLACKJACK'

    # Pair decisions precede total-dependent decisions. Never split tens/fives.
    if len(cards) == 2 and cards[0] == cards[1]:
        r = cards[0]
        split = (r in ('A', '8') or
                 (r in ('2', '3', '7') and 2 <= d <= 7) or
                 (r == '4' and d in (5, 6)) or
                 (r == '6' and 2 <= d <= 6) or
                 (r == '9' and d in (2, 3, 4, 5, 6, 8, 9)))
        if split:
            return 'SPLIT'

    t = hand.total
    can_double = len(cards) == 2
    if hand.soft:
        double = ((t in (13, 14) and d in (5, 6)) or
                  (t in (15, 16) and d in (4, 5, 6)) or
                  (t in (17, 18) and d in (3, 4, 5, 6)))
        if can_double and double:
            return 'DOUBLE'
        if t >= 19 or (t == 18 and d in (2, 3, 4, 5, 6, 7, 8)):
            return 'STAND'
        return 'HIT'

    double = ((t == 9 and 3 <= d <= 6) or
              (t == 10 and 2 <= d <= 9) or
              (t == 11 and 2 <= d <= 10))
    if can_double and double:
        return 'DOUBLE'
    if t >= 17 or (t == 12 and 4 <= d <= 6) or (13 <= t <= 16 and 2 <= d <= 6):
        return 'STAND'
    return 'HIT'
