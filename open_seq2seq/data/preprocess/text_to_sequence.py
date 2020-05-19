import re
import nltk
from random import random
from .symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_arpabet = nltk.corpus.cmudict.dict()

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

def separate_words(txt):
    out = []
    seq = ""
    cont = False
    for i in txt:
        if( i.isalpha() ):
            cont = True
            seq+=i
        elif( cont ):
            out.append(seq)
            seq=""
            out.append(i)
            cont = False
        else:
            out.append(i)
    if seq!="":
        out.append(seq)
    return out

def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if s in _symbol_to_id]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _maybe_get_arpabet(word, p):
    try:
        phonemes = _arpabet[word][0]
        phonemes = " ".join(phonemes)
    except KeyError:
        return word

    return '{%s}' % phonemes if random() < p else word


# Update this with a better word extractor in text
def mix_pronunciation(text, p):
    # text = ' '.join(_maybe_get_arpabet(word, p) for word in text.split(' '))
    text = "".join( _maybe_get_arpabet(word, p) for word in separate_words(text) )
    return text


def text_to_sequence( text, p=0.0 ):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      The text can optionally have ARPAbet sequences enclosed in curly braces embedded
      in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through

      Returns:
        List of integers corresponding to the symbols in the text
    '''
    text = text.lower()
    text = text.replace("'", "")
    if p >= 0:
        text = mix_pronunciation(text, p)
    
    # print(text)
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence( text )
            break
        sequence += _symbols_to_sequence( m.group(1) )
        sequence += _arpabet_to_sequence( m.group(2) )
        text = m.group(3)

    # Append EOS token
    # sequence.append(_symbol_to_id['~'])
    return sequence