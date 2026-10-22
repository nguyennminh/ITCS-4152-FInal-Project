import tempfile
import unittest
from pathlib import Path
from scripts.prepare_data import label_rank, convert_labels


class DataTests(unittest.TestCase):
    def test_rank_mapping(self):
        for source, expected in [('AS','A'),('10H','10'),('queen_of_hearts','Q'),('KD','K')]:
            self.assertEqual(label_rank(source), expected)
        with self.assertRaises(ValueError):
            label_rank('joker')

    def test_invalid_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'test.txt'
            p.write_text('0 0.5 0.5 0.2 0.3\n')
            self.assertTrue(convert_labels(p, [12]).startswith('12 '))
            p.write_text('0 0.9 0.5 0.8 0.3\n')
            with self.assertRaises(ValueError):
                convert_labels(p, [12])
            p.write_text('-1 0.5 0.5 0.2 0.3\n')
            with self.assertRaises(ValueError):
                convert_labels(p, [12])


if __name__ == '__main__':
    unittest.main()
