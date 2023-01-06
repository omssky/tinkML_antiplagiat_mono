import argparse
import ast
from typing import List, Tuple

class Normalizer:
    def normalize(self, text: str) -> str:
        """
        Normalize the formatting of a Python program by simplifying docstrings and
        function annotations, and normalizing white space and line breaks.
        """
        # Parse the text into an AST
        tree = ast.parse(text)

        # Simplify docstrings and function annotations
        self.simplify_docstrings_and_annotations(tree)

        # Normalize formatting and convert back to text
        text = ast.unparse(tree).strip()

        # Normalize white space and line breaks
        text = self.normalize_whitespace_and_linebreaks(text)

        return text

    def simplify_docstrings_and_annotations(self, tree):
        """Simplify docstrings and function annotations in the given AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                node.value.s = 'DOCSTRING'
            if isinstance(node, ast.FunctionDef):
                node.returns = None
                node.args.args = []
                node.args.vararg = None
                node.args.kwarg = None
                node.args.kwonlyargs = []
                node.args.defaults = []
                node.args.kw_defaults = []

    def normalize_whitespace_and_linebreaks(self, text: str) -> str:
        """Normalize white space and line breaks in the given text."""
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        return '\n'.join(lines)


class Levenshtein:
    def distance(self, s1: str, s2: str) -> int:
        """
        Calculate the Levenshtein distance between two strings.
        """
        if s1 == s2:
            return 0
        if len(s1) == 0:
            return len(s2)
        if len(s2) == 0:
            return len(s1)

        # Initialize the distance matrix
        matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        for i in range(len(s1) + 1):
            matrix[i][0] = i
        for j in range(len(s2) + 1):
            matrix[0][j] = j

        # Calculate the distance
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i - 1] == s2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)

        # Return the distance
        return matrix[len(s1)][len(s2)]


class FileComparer:
    def __init__(self, normalizer: Normalizer, distance_calculator: Levenshtein):
        self.normalizer = normalizer
        self.distance_calculator = distance_calculator

    def compare(self, file1: str, file2: str) -> float:
        """
        Calculate the similarity of two files by comparing their contents.
        """
        with open(file1, 'r', encoding="utf8") as f1, open(file2, 'r', encoding="utf8") as f2:
            text1 = f1.read()
            text2 = f2.read()
            text1 = self.normalizer.normalize(text1)
            text2 = self.normalizer.normalize(text2)
            return 1 - self.distance_calculator.distance(text1, text2) / max(len(text1), len(text2))

class InputParser:
  def parse(self, input_file: str) -> List[Tuple[str, str]]:
    """
    Parse the input file containing pairs of file paths.
    """
    pairs = []
    with open(input_file, 'r') as f:
      for line in f:
        pair = tuple(line.strip().split())
        pairs.append(pair)
    return pairs

class Runner:
  def __init__(self, file_comparer: FileComparer, input_parser: InputParser):
    self.file_comparer = file_comparer
    self.input_parser = input_parser

  def compare(self, input_file: str, output_file: str):
    try:
        pairs = self.input_parser.parse(input_file)
        scores = []
        for pair in pairs:
            scores.append(self.file_comparer.compare(*pair))
        with open(output_file, 'w') as f:
            for score in scores:
                f.write(str(score) + '\n')
    except (FileNotFoundError, OSError) as e:
        print(f"Error: {e}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='input file containing pairs of file paths')
    parser.add_argument('output_file', help='output file where the results will be written')
    args = parser.parse_args()

    # Create the anti-plagiarism utility
    normalizer = Normalizer()
    distance_calculator = Levenshtein()
    file_comparer = FileComparer(normalizer, distance_calculator)
    input_parser = InputParser()
    utility = Runner(file_comparer, input_parser)

    # Compare the pairs of files and write the results to the output file
    utility.compare(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
