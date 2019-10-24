#!/usr/bin/env python3

from pyparsing import (Literal, CaselessLiteral, Word, Combine, Group, Optional,
                       ZeroOrMore, Forward, nums, alphas, oneOf)
import math
import operator

# Mostly copied from
# https://rosettacode.org/wiki/Decimal_floating_point_number_to_binary
hex2bin = dict('{:x} {:04b}'.format(x,x).split() for x in range(16))
def float_dec2bin(d):
    neg = False
    if d < 0:
        d = -d
        neg = True
    hx = float(d).hex()
    p = hx.index('p')
    bn = ''.join(hex2bin.get(char, char) for char in hx[2:p])
    return ('-' if neg else '') + bn.strip('0') + hx[p:p+2] + hx[p+2:]

# Mostly copied from https://stackoverflow.com/a/2371789/5135869
class NumericStringParser(object):
    '''
    Most of this code comes from the fourFn.py pyparsing example

    '''

    def pushFirst(self, strg, loc, toks):
        self.exprStack.append(toks[0])

    def pushUMinus(self, strg, loc, toks):
        if toks and toks[0] == '-':
            self.exprStack.append('unary -')

    def __init__(self):
        self.config = {
            "base": 2
        }

        """
        expop   :: '^'
        multop  :: '*' | '/'
        addop   :: '+' | '-'
        integer :: ['+' | '-'] '0'..'9'+
        atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
        factor  :: atom [ expop factor ]*
        term    :: factor [ multop factor ]*
        expr    :: term [ addop term ]*
        """

        point = Literal(".")
        e = CaselessLiteral("E")
        # fnumber = Combine(Word("+-" + nums, nums) +
        #                   Optional(point + Optional(Word(nums))) +
        #                   Optional(e + Word("+-" + nums, nums)))
        fnumber = Word("+-." + nums + alphas)
        ident = Word(alphas, alphas + nums + "_$")
        plus = Literal("+")
        minus = Literal("-")
        mult = Literal("*")
        div = Literal("/")
        lpar = Literal("(").suppress()
        rpar = Literal(")").suppress()
        addop = plus | minus
        multop = mult | div
        expop = Literal("^")
        pi = CaselessLiteral("PI")
        expr = Forward()
        atom = ((Optional(oneOf("- +")) +
                 (ident + lpar + expr + rpar | pi | e | fnumber).setParseAction(self.pushFirst))
                | Optional(oneOf("- +")) + Group(lpar + expr + rpar)
                ).setParseAction(self.pushUMinus)
        # by defining exponentiation as "atom [ ^ factor ]..." instead of
        # "atom [ ^ atom ]...", we get right-to-left exponents, instead of left-to-right
        # that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        factor = Forward()
        factor << atom + \
            ZeroOrMore((expop + factor).setParseAction(self.pushFirst))
        term = factor + \
            ZeroOrMore((multop + factor).setParseAction(self.pushFirst))
        expr << term + \
            ZeroOrMore((addop + term).setParseAction(self.pushFirst))
        # addop_term = ( addop + term ).setParseAction( self.pushFirst )
        # general_term = term + ZeroOrMore( addop_term ) | OneOrMore( addop_term)
        # expr <<  general_term
        self.bnf = expr
        # map operator symbols to corresponding arithmetic operations
        epsilon = 1e-12
        self.opn = {"+": operator.add,
                    "-": operator.sub,
                    "*": operator.mul,
                    "/": operator.truediv,
                    "^": operator.pow}
        self.fn = {"sin": math.sin,
                   "cos": math.cos,
                   "tan": math.tan,
                   "exp": math.exp,
                   "abs": abs,
                   "trunc": lambda a: int(a),
                   "round": round,
                   "sgn": lambda a: abs(a) > epsilon and cmp(a, 0) or 0}

    def evaluateStack(self, s):
        op = s.pop()
        if op == 'unary -':
            return -self.evaluateStack(s)
        if op in "+-*/^":
            op2 = self.evaluateStack(s)
            op1 = self.evaluateStack(s)
            return self.opn[op](op1, op2)
        elif op == "PI":
            return math.pi  # 3.1415926535
        elif op == "E":
            return math.e  # 2.718281828
        elif op in self.fn:
            return self.fn[op](self.evaluateStack(s))
        # elif op[0].isalpha():
        #     return 0
        else:
            return self.parse_num(op)

    def eval(self, num_string, parseAll=True):
        if num_string[0] == '/':
            # Command
            cmd = num_string[1:].split(' ')
            if cmd[0] == "set":
                self.config[cmd[1]] = int(cmd[2])
            else:
                print("Invalid command")
                return None
        else:
            # Expression
            self.exprStack = []
            results = self.bnf.parseString(num_string, parseAll)
            val = self.evaluateStack(self.exprStack[:])
            return val

    # Custom method for parsing numbers in other bases
    def parse_num(self, op):
        result = 0
        base = self.config["base"]

        prefix = op[0:2]
        suffix_val = op[2:]
        if prefix == "0b":
            base = 2
            op = suffix_val
        elif prefix == "0d":
            base = 10
            op = suffix_val
        elif prefix == "0h" or prefix == "0x":
            base = 16
            op = suffix_val

        if '.' in op:
            [ipart, fpart] = op.split('.')
            ival = 0 if len(ipart) == 0 else int(ipart, base)
            fval = int(fpart, base)
            result = ival + fval / base**len(fpart)
        else:
            result = int(op, base)

        return result


nsp = NumericStringParser()

# REPL loop
while True:
    # Read
    c = input(f"B{nsp.config['base']} > ")

    # Evaluate
    result = nsp.eval(c)
    if result is None:
        continue

    # Print
    print(f"\t{result}\t{float_dec2bin(result)}")
