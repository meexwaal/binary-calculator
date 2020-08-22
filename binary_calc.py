#!/usr/bin/env python3

from pyparsing import (Literal, CaselessLiteral, Word, Combine, Group, Optional,
                       ZeroOrMore, Forward, nums, alphas, oneOf)
import math
import operator
import struct

# Returns the tuple (sign, mag) where sign is either '-' or '' and mag is abs(v)
def split_sign(v):
    return ('-' if v < 0 else '', abs(v))

# Get the IEEE bit representation of a float
# Copied from https://stackoverflow.com/a/14431225/5135869
def float_to_bits(f):
    s = struct.pack('>f', f)
    return struct.unpack('>L', s)[0]

# Convert a float to its IEEE representation as a binary string
def ieee_bin(f):
    bits = float_to_bits(f)
    bitstr = bin(bits)[2:].zfill(32)

    return f"{bitstr[0]}_{bitstr[1:9]}_{bitstr[9:32]}"

# Convert a float to its IEEE representation as a hex string
def ieee_hex(f):
    bits = float_to_bits(f)

    return hex(bits)

# float_to_bin - convert a float to a binary string representation
# float_to_hex - convert a float to a hexidecimal string representation
# Mostly copied from
# https://rosettacode.org/wiki/Decimal_floating_point_number_to_binary
hex2bin = dict('{:x} {:04b}'.format(x,x).split() for x in range(16))
def float_to_bin(d):
    (sign, mag) = split_sign(d)
    hx = float(mag).hex()
    p = hx.index('p')
    bn = ''.join(hex2bin.get(char, char) for char in hx[2:p])
    return sign + '0b' + bn.strip('0') + hx[p:p+2] + hx[p+2:]

def float_to_hex(d):
    (sign, mag) = split_sign(d)
    hx = float(mag).hex()
    p = hx.index('p')
    bn = hx[2:p]
    return sign + '0x' + bn.strip('0') + hx[p:p+2] + hx[p+2:]

# Custom exception
class FloatToTwosComp(Exception):
    pass

# Class for parsing and evaluating a numeric expression
# Mostly copied from https://stackoverflow.com/a/2371789/5135869
class NumericStringParser(object):
    '''
    Most of this code comes from the fourFn.py pyparsing example

    '''

    # Clamp a value to two's complement for the current bit width
    def clamp_tc(self, v):
        if self.config["width"] < float("inf"):
            if int(v) != v:
                raise FloatToTwosComp
            return v % (2**self.config["width"])
        else:
            return v

    # Same, man
    def get_help(self):
        help_text = f"""
Binary Calculator

Press C-d to to quit
Default base is currently base {self.config['base']}
All numbers with no base prefix will be interpreted in that base
Set width to some constant to perform two's complement arithmetic in that width
Set width to inf to perform arithmetic in arbitrary (Python) precision

Prefix a number with:
  0b for base 2
  0o for base 8
  0d for base 10
  0x or 0h for base 16

Available commands:
  /set <parameter> <value> : Set configuration <parameter> to <value>
  /help or /?              : Display this message
  /quit or /exit           : Quit or exit

Current parameters:
"""
        for key in self.config:
            help_text += f"  {key}\t: {self.config[key]}\n"
        return help_text

    def pushFirst(self, strg, loc, toks):
        self.exprStack.append(toks[0])

    def pushUMinus(self, strg, loc, toks):
        if toks and toks[0] == '-':
            self.exprStack.append('unary -')

    def __init__(self):
        self.config = {
            "base" : 2,
            "width" : float("inf")
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
        fnumber = Word("+-." + nums + alphas, "." + nums + alphas)
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
            return self.clamp_tc(-self.evaluateStack(s))

        elif op in "+-*":
            # Perform 2's comp clamping after the op
            op2 = self.evaluateStack(s)
            op1 = self.evaluateStack(s)
            return self.clamp_tc(self.opn[op](op1, op2))

        elif op in "/":
            # Perform 2's comp clamping before and after the op,
            # and round down after doing the division
            op2 = self.clamp_tc(self.evaluateStack(s))
            op1 = self.clamp_tc(self.evaluateStack(s))
            result = self.opn[op](op1, op2)
            if self.config["width"] < float("inf"):
                return self.clamp_tc(int(result))
            else:
                return result

        elif op in "^":
            op2 = self.evaluateStack(s) # Don't make the exponent 2's comp
            # (but if it's not a literal, the op will force it to 2's comp)
            op1 = self.clamp_tc(self.evaluateStack(s))
            return self.clamp_tc(self.opn[op](op1, op2))

        elif op == "PI":
            return math.pi  # 3.1415926535

        elif op == "E":
            return math.e  # 2.718281828

        elif op in self.fn:
            return self.fn[op](self.evaluateStack(s))

        # elif op[0].isalpha():
        #     return 0

        else:
            # Don't convert literals to 2's comp right away because there are
            # some cases where we want the non-Two's comp (like exponents)
            # This means we do have to clamp at the Very end
            return self.parse_num(op)

    def eval(self, num_string, parseAll=True):
        if len(num_string) == 0:
            return None
        elif num_string[0] == '/':
            # Command
            cmd = num_string[1:].split(' ')
            if cmd[0] == "set":
                if cmd[2] == "inf":
                    self.config[cmd[1]] = float("inf")
                else:
                    self.config[cmd[1]] = int(cmd[2])
                return None
            elif cmd[0] == "help" or cmd[0] == '?':
                print(self.get_help())
                return None
            elif cmd[0] == "quit" or cmd[0] == "exit":
                quit() or exit() # :-P
            else:
                print("Invalid command")
                return None
        else:
            # Expression
            self.exprStack = []
            results = self.bnf.parseString(num_string, parseAll)
            val = self.clamp_tc(self.evaluateStack(self.exprStack[:]))
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
        if prefix == "0o":
            base = 8
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
            fval = 0 if len(fpart) == 0 else int(fpart, base)
            result = ival + fval / base**len(fpart)
        else:
            result = int(op, base)

        return result


print("Type /help for usage")
nsp = NumericStringParser()

# REPL loop
while True:
    # Read
    try:
        c = input(f"B{nsp.config['base']} > ")
    except EOFError:
        # C-d quits
        print("")
        break
    except KeyboardInterrupt:
        # C-c cancels the current command and opens a new line
        print("")
        continue

    # Evaluate
    try:
        result = nsp.eval(c)
    except ValueError as err:
        print(err)
        continue

    if result is None:
        continue

    # Print
    # For ints:   print in binary, decimal, and hex
    # For floats: print in bin frac, dec frac, and hex float
    if int(result) == result:
        result = int(result)
        # Print ints
        (sign, mag) = split_sign(result)
        print(f"\t{bin(result)}\t{sign}0d{mag}\t{hex(result)}")

        # And print negatives when we're dealing with fixed precision
        if (nsp.config["width"] < float("inf")):
            neg = nsp.clamp_tc(-result)
            print(f"\t-{bin(neg)}\t-0d{neg}\t-{hex(neg)}")
    else:
        # Print floats
        (sign, mag) = split_sign(result)
        print(f"\t{float_to_bin(result)}\t{sign}0d{mag}\t{float_to_hex(result)}\tIEEE:{ieee_bin(result)}, {ieee_hex(result)}")
