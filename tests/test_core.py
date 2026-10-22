import unittest
from blackjack import evaluate_hand
from strategy import recommend
from regions import Detection, Stabilizer, read_table


class HandTests(unittest.TestCase):
    def test_aces(self):
        for ranks, total, soft in [(['A','7'],18,True), (['A','7','K'],18,False),
                                   (['A','A','9'],21,True), (['A','A','9','K'],21,False),
                                   (['A','A','A'],13,True)]:
            with self.subTest(ranks=ranks):
                hand = evaluate_hand(ranks)
                self.assertEqual((hand.total, hand.soft), (total, soft))

    def test_terminal_and_invalid(self):
        self.assertTrue(evaluate_hand(['A','K']).blackjack)
        self.assertFalse(evaluate_hand(['7','7','7']).blackjack)
        self.assertTrue(evaluate_hand(['K','Q','2']).bust)
        self.assertEqual(evaluate_hand([]).total, 0)
        with self.assertRaises(ValueError):
            evaluate_hand(['joker'])


class StrategyTests(unittest.TestCase):
    def test_reference_cases(self):
        cases = [(['8','8'],'A','SPLIT'), (['A','A'],'10','SPLIT'),
                 (['4','4'],'5','SPLIT'), (['4','4'],'4','HIT'),
                 (['9','9'],'7','STAND'), (['9','9'],'9','SPLIT'),
                 (['5','5'],'9','DOUBLE'), (['5','5'],'10','HIT'),
                 (['5','6'],'A','HIT'), (['5','6'],'10','DOUBLE'),
                 (['A','7'],'6','DOUBLE'), (['A','7'],'2','STAND'),
                 (['A','7'],'A','HIT'), (['A','8'],'6','STAND'),
                 (['A','2','5'],'6','STAND'), (['2','3','6'],'6','HIT'),
                 (['10','2'],'3','HIT'), (['10','2'],'4','STAND'),
                 (['10','6'],'10','HIT'), (['10','7'],'A','STAND'),
                 (['A','K'],'A','BLACKJACK'), (['K','Q','5'],'6','BUST')]
        for hand, dealer, expected in cases:
            with self.subTest(hand=hand, dealer=dealer):
                self.assertEqual(recommend(hand, dealer), expected)

    def test_no_double_after_hit(self):
        from blackjack import RANKS
        from itertools import product
        for hand in product(RANKS, repeat=3):
            for dealer in ('2','6','10','A'):
                self.assertNotIn(recommend(hand, dealer), ('DOUBLE','SPLIT'))


class TableTests(unittest.TestCase):
    def test_stability_resets_immediately(self):
        s = Stabilizer(2)
        a, b = (('8','8'), ('6',)), (('8','8','2'), ('6',))
        self.assertIsNone(s.update(a))
        self.assertEqual(s.update(a), a)
        self.assertIsNone(s.update(b))
        self.assertIsNone(s.update(None))
        self.assertIsNone(s.update(b))
        self.assertEqual(s.update(b), b)

    def test_duplicates_and_unknown(self):
        config = {'confidence':0.75, 'regions':{'dealer':[0,0,1,0.4], 'player':[0,0.5,1,1]}}
        cards = [Detection('8',0.9,(10,60,20,80)), Detection('8',0.9,(30,60,40,80)),
                 Detection('6',0.9,(10,10,20,30))]
        self.assertEqual(read_table(cards, (100,100,3), config), (('8','8'), ('6',)))
        self.assertIsNone(read_table(cards + [Detection(None,0.3,(50,60,60,80))], (100,100,3), config))
        self.assertIsNone(read_table(cards + [Detection('9',0.9,(40,10,50,30))], (100,100,3), config))


if __name__ == '__main__':
    unittest.main()
