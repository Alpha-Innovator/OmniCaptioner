import re
import json
from math import isclose
import sympy as sp
from sympy import simplify, Eq, sympify, evalf, Pow
from sympy.parsing.latex import parse_latex
import antlr4
from decimal import Decimal, getcontext
from fractions import Fraction
import sys
import math


chinese_answer_type_dict = {
    'Numerical': '数值',
    'Expression': '表达式',
    'Equation': '方程',
    'Interval': '区间'
}
english_answer_type_dict = {
    'Numerical': 'a numerical value',
    'Expression': 'an expression',
    'Equation': 'an equation',
    'Interval': 'an interval'
}


def get_single_answer_type_text(answer_type, is_chinese):
    if '-' in answer_type:  # No need now
        answer_type = answer_type[:answer_type.find('-')]
    for t in ['Numerical', 'Expression', 'Equation', 'Interval']:
        if t in answer_type:
            if is_chinese:
                return chinese_answer_type_dict[t]
            else:
                return english_answer_type_dict[t]
    exit(f'Error parsing answer type {answer_type}!')


def get_answer_type_text(answer_type, is_chinese, multiple_answer):
    # 'Tuple' has various meanings in different context, such as position or values of a series of variable,
    # so it may lead to confusion to directly use 'tuple' in the prompt.
    if ('Need_human_evaluate' in answer_type) or ('Tuple' in answer_type):
        full_answer_text = ''
    else:
        if not multiple_answer:
            answer_text = get_single_answer_type_text(answer_type, is_chinese)
            if is_chinese:
                full_answer_text = f'，答案类型为{answer_text}'
            else:
                full_answer_text = f"The answer of The problem should be {answer_text}. "
        else:
            if ',' not in answer_type:  # Same answer type for all answers
                answer_text = get_single_answer_type_text(answer_type, is_chinese)
                if is_chinese:
                    full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
                else:
                    full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
            else:
                answer_types = answer_type.split(',')
                answer_types = [get_single_answer_type_text(t, is_chinese) for t in answer_types]
                if len(set(answer_types)) == 1:
                    answer_text = answer_types[0]
                    if is_chinese:
                        full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
                    else:
                        full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
                else:
                    if is_chinese:
                        answer_text = '、'.join(answer_types)
                        full_answer_text = f'，题目有多个答案，答案类型分别为{answer_text}'
                    else:
                        answer_text = ', '.join(answer_types)
                        full_answer_text = (
                            f'The problem has multiple answers, with the answers in order being {answer_text}. '
                        )
    return full_answer_text


def make_input(prompt, question_content):
    # diversified based on the vllm, which is not implemented temporarily
    input = prompt + '\n' + question_content
    return input


sys.set_int_max_str_digits(1000000)
# Set the precision of decimal
getcontext().prec = 50


class MathJudger:
    def __init__(self):
        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.pi = parse_latex("\\pi")
        self.precision = 1e-8

    def split_by_comma(self, expr: str):
        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char == "(" or char == "[":
                in_bracket_num += 1
            elif char == ")" or char == "]":
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())

        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list):
        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)

        return new_expr_list

    def judge(self, expression1, expression2, precision=1e-8):
        # expression1:  Ground_Truth
        precision = precision if isinstance(precision, list) else [precision]

        try:
            expression1, expression2 = self.preprocess(expression1, expression2)
        except:
            return False
        if expression1 == expression2:
            return True

        expression1 = re.sub(r'[\u4e00-\u9fff]+', '', expression1)
        expression2 = re.sub(r'[\u4e00-\u9fff]+', '', expression2)

        expression1 = self.split_by_comma(expression1)
        expression2 = self.split_by_comma(expression2)

        temp_list1 = self.trans_plus_minus_sign(expression1)
        temp_list2 = self.trans_plus_minus_sign(expression2)

        # List of Error Values
        if len(precision) <= 1:
            precision = precision * len(temp_list1)

        if len(temp_list1) != len(temp_list2):
            return False

        # Check if the elements of two lists can be paired off in pairs, and the pairs are equal, thereby supporting the comparison of multiple answers
        idx = -1
        while len(temp_list1) != 0:
            idx = (idx + 1) % len(temp_list1)

            item1 = temp_list1[idx]
            self.precision = precision[idx]

            for item2 in temp_list2:
                if self.is_equal(item1, item2):
                    temp_list1.remove(item1)
                    temp_list2.remove(item2)
                    precision.remove(self.precision)
                    break
            else:
                # If we didn't break from the inner loop, it means no match was found
                return False

        # If all elements are matched and removed, the lists can be paired
        return True

    def is_interval(self, epr):
        return epr.startswith(("(", "[")) and epr.endswith((")", "]"))


    # expression1: ground_truth
    def is_equal(self, expression1, expression2):
        if expression1 == expression2 and expression1 != "" and expression2 != "":
            return True

        # Firstly, determine if it is two intervals. If so, check for equality; if not equal, return False.
        if self.is_interval(expression1) and self.is_interval(expression2):
            try:
                if self.interval_equal(expression1, expression2):
                    return True
            except:
                return False

        # Check for equal number
        try:
            if self.numerical_equal(expression1, expression2):
                # print("数值等价")
                return True
        except:
            pass

        # Check for equal expression
        try:
            if self.expression_equal(expression1, expression2) and not ("=" in expression1 and "=" in expression2):
                return True
        except:
            pass

        # Check for equal equation
        try:
            if self.equation_equal(expression1, expression2):
                return True
        except:
            pass

        return False

    # Determine whether two numerical values are equal within the allowed error range.
    def numerical_equal(self, expression1: str, expression2: str, include_percentage: bool = True):
        """
        (expression1: Ground_Truth)
        """
        reference = float(expression1)
        prediction = float(expression2)

        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]

        for item in gt_result:
            # if isclose(item, prediction, abs_tol=self.precision, rel_tol=0):
            if abs(item - prediction) <= self.precision * 1.01:
                return True
        return False

    # Determine whether two expressions are mathematically equivalent
    def expression_equal(self, exp1, exp2):
        """
        (expression1: Ground_Truth)
        Function: Determine if two expressions are mathematically equivalent
        Step 1: Extract the expression, to prevent some models from giving "x=1" instead of "1"
        Step 2: Use the sympy library for equivalence judgment
        """

        def extract_expression(expression):
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()

        exp1 = extract_expression(exp1)
        exp2 = extract_expression(exp2)

        exp_too_long = len(exp1) > 300 or len(exp2) > 300

        expr1_sym = sympify(parse_latex(exp1))
        expr2_sym = sympify(parse_latex(exp2))

        if expr1_sym == expr2_sym:
            return True
        else:
            expr1_sym = self.sympy_sub_pi(expr1_sym)
            expr2_sym = self.sympy_sub_pi(expr2_sym)

            if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (
                    not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)):
                return False
            elif not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
                try:
                    if not (self.can_compute_power(expr1_sym) and self.can_compute_power(expr2_sym)):
                        print(
                            "These two number can not be calculated by current computer for: "
                            f"\"{str(expr1_sym)}\" and \"{str(expr2_sym)}\""
                        )
                        return False
                    if exp_too_long:
                        print(f'Expression {exp1} or {exp2} is too long to compute. ')
                        return False

                    if abs(expr1_sym.evalf() - expr2_sym.evalf()) <= self.precision * 1.01:
                        return True
                    else:
                        return False
                except:
                    return False
            elif exp_too_long:
                print(f'Expression {exp1} or {exp2} is too long to compute. ')
                return False
            else:
                try:
                    simplified_expr = simplify(expr1_sym - expr2_sym)

                    num_value = simplified_expr.evalf()

                    return abs(num_value) < 1e-3
                except:
                    return False

    def equation_equal(self, expression1, expression2):
        """
        (expression1: Ground_Truth)
        Function: Determine if two equations are mathematically equivalent
        Step 1: Simplify one equation/equality to a standard equation, i.e., the right side of the equation strictly equals 0. Next, only the left sides of the two equations need to be judged for "equivalence"
        Step 2: Use the sympy library to calculate the quotient of the left sides of the two equations. If this quotient or its reciprocal is an integer, then mathematically we can deduce that these two equations are equivalent
        """

        def simplify_equation(latex_eq):
            lhs, rhs = latex_eq.split('=')

            lhs_expr = parse_latex(lhs)
            rhs_expr = parse_latex(rhs)

            equation = Eq(lhs_expr, rhs_expr)

            simplified_eq = simplify(equation.lhs - equation.rhs)

            return simplified_eq

        expr1_sym = simplify_equation(expression1)
        expr2_sym = simplify_equation(expression2)

        division_result_1 = simplify(expr1_sym / expr2_sym)
        division_result_2 = simplify(expr2_sym / expr1_sym)

        if (division_result_1.is_Integer and division_result_1 != 0) or (
                division_result_2.is_Integer and division_result_2 != 0):
            return True
        else:
            return False

    def interval_equal(self, expression1, expression2):
        # Function: Determine if two intervals are mathematically equivalent
        # Step 1: Simplify the expression of the interval, remove irrelevant symbols such as "\left", "\right", and delete the possible "x \in"
        # Step 2: Compare if the left and right symbols, and the mathematical expressions in the middle of the two intervals are consistent

        def compare_two_interval(inter1, inter2):

            if inter1[0] != inter2[0] or inter1[-1] != inter2[-1]:
                return False

            inter1 = inter1.strip('[]()')
            inter2 = inter2.strip('[]()')

            items_1 = inter1.split(',')
            items_2 = inter2.split(',')

            for item_1, item_2 in zip(items_1, items_2):
                if not self.expression_equal(item_1, item_2):
                    return False
            return True

        interval1 = expression1
        interval2 = expression2

        if interval1 == interval2:
            return True
        else:
            inter_list1 = interval1.split("\\cup")
            inter_list2 = interval2.split("\\cup")

            if len(inter_list1) != len(inter_list2):
                return False
            else:
                for inter1, inter2 in zip(inter_list1, inter_list2):
                    if not compare_two_interval(inter1, inter2):
                        return False
                return True

    def preprocess(self, expression1, expression2):

        def extract_boxed_content(latex_str):
            boxed_matches = re.finditer(r'\\boxed{', latex_str)
            results = ""

            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1

                while stack > 0 and end_index < len(latex_str):
                    if latex_str[end_index] == '{':
                        stack += 1
                    elif latex_str[end_index] == '}':
                        stack -= 1
                    end_index += 1

                if stack == 0:
                    content = latex_str[start_index:end_index - 1]
                    results += content + ","
                else:
                    raise ValueError("Mismatched braces in LaTeX string.")

            if results == "":
                last_line_ans = latex_str.strip().split("\n")[-1]
                dollar_pattern = r"\$(.*?)\$"
                answers = re.findall(dollar_pattern, last_line_ans)

                if answers:
                    for ans in answers:
                        results += ans + ","
                else:
                    results = latex_str

            return results

        def sepcial_symbol_replace(expression):
            if "\\in " in expression:
                expression = expression.split("\\in ")[1]

            for signal in self.special_signal_map:
                expression = expression.replace(signal, self.special_signal_map[signal])

            expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")

            pattern = r'\\(?:mathrm|mathbf)\{~?([^}]*)\}'
            expression = re.sub(pattern, r'\1', expression)

            return expression

        exp1, exp2 = extract_boxed_content(expression1), extract_boxed_content(expression2)
        exp1, exp2 = sepcial_symbol_replace(exp1), sepcial_symbol_replace(exp2)

        return exp1, exp2

    def can_compute_power(self, expr):
        """
        Check if the power expression can be computed.

        Parameters:
        expr (sympy expression): The expression to check.

        Returns:
        bool: True if the expression can be computed, False otherwise.
        """
        # Check if the expression is a power expression
        if isinstance(expr, Pow):
            # Extract the base and the exponent
            base, exp = expr.as_base_exp()

            # Check if the base and the exponent are numbers
            if base.is_number and exp.is_number:
                # Set a threshold for the maximum size of the exponent
                MAX_EXP = 1000  # This threshold can be adjusted based on the computing environment

                # Check if the exponent is greater than the threshold
                if abs(exp.evalf()) > MAX_EXP:
                    return False
                else:
                    return True
            else:
                # If the base or the exponent is not a number, we cannot compute the power
                return False
        else:
            # If the expression is not a power expression, return True as it is not the case we are checking for
            return True


def extract_answer(is_chinese, model_output, is_deepseek=False):
    # deepseekmath has special answering format
    if str(model_output) == 'nan':
        model_output = 'nan'

    if is_deepseek:
        if is_chinese:
            matches = re.findall('## 解题答案(.*)', model_output)
        else:
            matches = re.findall('The answer is: (.*)', model_output)

        if matches:
            model_answer = matches[-1].strip()
            return model_answer
        else:
            return model_output

    if is_chinese:
        matches = re.findall('所以最终答案是(.*)', model_output)
    else:
        matches = re.findall('So the final answer is (.*)', model_output)

    if matches:
        model_answer = matches[-1].strip()
        return model_answer
    else:
        return model_output


def calculate_merged_accuracy(reference_dir, text_only):
    pass
