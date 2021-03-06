#source https://github.com/eyaler/hebrew_tokenizer
import re
from unidecode import unidecode


def cc(s):
    return '[' + s + ']'

def ncc(s):
    return cc('^' + s)

def ncg(s):
    return '(?:' + s + ')'

def nla(s):
    return '(?!' + s + ')'

def to_nonfinal(text):
    return text.replace('ך', 'כ').replace('ם', 'מ').replace('ן', 'נ').replace('ף', 'פ').replace('ץ', 'צ')

def to_final(text):
    return text.replace('כ', 'ך').replace('מ', 'ם').replace('נ', 'ן').replace('פ', 'ף').replace('צ', 'ץ')

class HebTokenizer:
    '''
    Nikud and Teamim are ignored.
    Punctuation is normalized to ASCII (using unidecode).
    Correct usage of final letters (ךםןףץ) is enforced. Final פ is allowed.
    Same letter repetition (שולטתתתת), which is a common form of slang writing, is limited to a maximum of max_letter_repetition (default=2),
        and at the end of words a maximum max_end_of_word_letter_repetition (default=2). Use 0 or None for no limit.
        Note that these will throw away a very small number of legitimate repetitions, most notably 'מממ' as in 'מממן', 'מממשלת'.
        allow_mmm (default=True) will specifically allow 'מממ' for the case max_letter_repetition==2.
    Acronyms (צה"ל) and abbreviations ('וכו) are excluded. TBD: option to allow these.
    MWE refers to multi-word expression *candidates*, which are tokenized based on hyphen/makaf or surrounding punctuation.
    Hyphen-based MWE's are discarded if they contain more than max_mwe_hyphens (default=1). Use 0 not allowing hyphens or None for unlimited hyphens.
    Line opening hyphens as used in conversation and enumeration, can be ignored by allow_line_opening_hyphens (default=True)
    Strict mode can enforce the absence of extraneous hebrew letters in the same "clause" (strict=HebTokenizer.CLAUSE),
        sentence (strict=HebTokenizer.SENTENCE) or line (strict=HebTokenizer.LINE) of the MWE. Use 0 or None to not be strict (default=None).
    '''

    hebrew_diacritics = '\u0591-\u05bd\u05bf-\u05c2\u05c4\u05c5\u05c7' # all nikud and teamim except makaf, sof-pasuk, nun-hafukha
    hebrew_diacritics_regex = re.compile(cc(hebrew_diacritics))

    hebrew_letters = 'א-ת'
    nonfinal_letters = 'אבגדהוזחטיכלמנסעפצקרשת'
    final_letters = to_final(nonfinal_letters) + 'פ'
    nonfinal_letters_allowing_geresh = 'גזצ'
    final_letters_allowing_geresh = to_final(nonfinal_letters_allowing_geresh)
    geresh = '\''
    nonfinal_letter_geresh_pattern = ncg(cc(nonfinal_letters_allowing_geresh) + geresh + '|' + cc(nonfinal_letters))
    final_letter_geresh_pattern = ncg(cc(final_letters_allowing_geresh) + geresh + '|' + cc(final_letters))
    non_hebrew_letters_regex = re.compile(ncc(hebrew_letters) + '+')

    sentence_sep = '.?!'
    clause_sep_before_space = sentence_sep + ':;,)"'
    clause_sep_after_space = '("'
    clause_sep_between_spaces = '-'
    clause_sep_pattern = '\t|' + cc(clause_sep_before_space) + '\s|\s' + cc(clause_sep_after_space) + '|\s' + cc(clause_sep_between_spaces) + '\s'
    clause_sep_regex = re.compile(clause_sep_pattern)
    sentence_sep_regex = re.compile(cc(sentence_sep))

    mwe_words_sep = ' -'
    mwe_words_sep_regex = re.compile(cc(mwe_words_sep))

    mmm_pattern = '(?<!(?<!m)mmm)'.replace('m', 'מ')
    line_opening_hyphen_pattern = '((?:^|\n|\r)\s*-{1,2})(?=\w)'
    line_opening_hyphen_regex = re.compile(line_opening_hyphen_pattern, flags=re.MULTILINE)

    CLAUSE = 1
    SENTENCE = 2
    LINE = 3

    default_max_letter_repetition = 2
    default_max_end_of_word_letter_repetition = 2
    default_allow_mmm = True
    default_max_mwe_hyphens = 1
    default_allow_line_opening_hyphens = True
    default_strict = None


    def __init__(self, max_letter_repetition=default_max_letter_repetition, max_end_of_word_letter_repetition=default_max_end_of_word_letter_repetition, allow_mmm=default_allow_mmm, max_mwe_hyphens=default_max_mwe_hyphens, allow_line_opening_hyphens=default_allow_line_opening_hyphens):
        self.max_letter_repetition = max_letter_repetition
        self.max_end_of_word_letter_repetition = max_end_of_word_letter_repetition
        self.allow_mmm = allow_mmm
        self.max_mwe_hyphens = max_mwe_hyphens
        self.allow_line_opening_hyphens = allow_line_opening_hyphens

        mmm = ''
        neg_rep = ''
        neg_end_rep = ''
        if max_letter_repetition == 2 and allow_mmm:
            mmm = self.mmm_pattern
        if max_letter_repetition:
            neg_rep = nla('\\1{' + str(max_letter_repetition) + '}' + mmm)
        if max_end_of_word_letter_repetition:
            neg_end_rep = nla('\\1{' + str(max_end_of_word_letter_repetition) + ',}' + ncg('$|' + ncc(self.hebrew_letters)))
        self.word_pattern = '(?<!' + cc(self.hebrew_letters) + '[^\s-])\\b' + ncg('(' + self.nonfinal_letter_geresh_pattern + ')' + neg_rep + neg_end_rep) + '+' + self.final_letter_geresh_pattern + '(?!\w)'+nla('[^\s-]' + cc(self.hebrew_letters)) + nla('-' + ncg('$|' + ncc(self.hebrew_letters)))

        max_mwe_hyphens_pattern = ''
        if max_mwe_hyphens != 0:
            max_mwe_hyphens_str = ''
            if max_mwe_hyphens is not None:
                max_mwe_hyphens_str = str(max_mwe_hyphens)
            max_mwe_hyphens_pattern = '|' + ncg('-' + self.word_pattern.replace('\\1', '\\3')) + '{1,' + max_mwe_hyphens_str + '}'
        self.mwe_pattern = '(?<!-)' + self.word_pattern + ncg(ncg(' ' + self.word_pattern.replace('\\1', '\\2')) + '+' + max_mwe_hyphens_pattern) + '(?!-)'
        self.line_with_strict_mwe_pattern = '^' + ncc(self.hebrew_letters) + '*' + self.mwe_pattern + ncc(self.hebrew_letters) + '*$'

        self.word_regex = re.compile(self.word_pattern)
        self.mwe_regex = re.compile(self.mwe_pattern)
        self.line_with_strict_mwe_regex = re.compile(self.line_with_strict_mwe_pattern, flags=re.MULTILINE)

    def sanitize(self, text):
        text = self.hebrew_diacritics_regex.sub('', text)
        text = self.non_hebrew_letters_regex.sub(lambda x: unidecode(x.group()), text)
        if self.allow_line_opening_hyphens:
            text = self.line_opening_hyphen_regex.sub('\\1 ', text)
        return text

    def is_word(self, text):
        text = self.sanitize(text)
        return self.word_regex.fullmatch(text) is not None

    def get_words(self, text, generator=False):
        text = self.sanitize(text)
        result = (match.group() for match in self.word_regex.finditer(text))
        if not generator:
            result = list(result)
        return result

    def has_word(self, text):
        for word in self.get_words(text, generator=True):
            return True
        return False

    def is_mwe(self, text):
        text = self.sanitize(text)
        return self.mwe_regex.fullmatch(text) is not None

    def is_word_or_mwe(self, text):
        return self.is_word(text) or self.is_mwe(text)

    def get_mwe(self, text, strict=default_strict, generator=False):
        text = self.sanitize(text)
        if strict:
            if strict == self.CLAUSE:
                text = '\n'.join(self.clause_sep_regex.split(text))
            elif strict == self.SENTENCE:
                text = '\n'.join(self.sentence_sep_regex.split(text))
            else:
                assert strict == self.LINE, 'Unknown strict mode: %s'%strict
            result = (self.mwe_regex.search(match.group()).group() for match in
                    self.line_with_strict_mwe_regex.finditer(text))
        else:
            result = (match.group() for match in self.mwe_regex.finditer(text))
        if not generator:
            result = list(result)
        return result

    def get_mwe_words(self, text, strict=default_strict, generator=False):
        result = (self.mwe_words_sep_regex.split(mwe) for mwe in self.get_mwe(text, strict=strict))
        if not generator:
            result = list(result)
        return result


if __name__ == '__main__':
    text = 'א בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ. ב וְהָאָ֗רֶץ הָיְתָ֥ה תֹ֙הוּ֙ וָבֹ֔הוּ וְחֹ֖שֶׁךְ עַל־פְּנֵ֣י תְה֑וֹם, וְר֣וּחַ אֱלֹהִ֔ים מְרַחֶ֖פֶת עַל־פְּנֵ֥י הַמָּֽיִם. ג וַיֹּ֥אמֶר אֱלֹהִ֖ים: "יְהִ֣י א֑וֹר", וַֽיְהִי־אֽוֹר. ד וַיַּ֧רְא אֱלֹהִ֛ים אֶת־הָא֖וֹר כִּי־ט֑וֹב, וַיַּבְדֵּ֣ל אֱלֹהִ֔ים בֵּ֥ין הָא֖וֹר וּבֵ֥ין הַחֹֽשֶׁךְ. ה וַיִּקְרָ֨א אֱלֹהִ֤ים לָאוֹר֙ "י֔וֹם" וְלַחֹ֖שֶׁךְ קָ֣רָא "לָ֑יְלָה", וַֽיְהִי־עֶ֥רֶב וַֽיְהִי־בֹ֖קֶר י֥וֹם אֶחָֽד.'

    def print_with_len(lst):
        print(lst, len(lst))

    heb_tokenizer = HebTokenizer()
    print_with_len(text)
    # print_with_len(heb_tokenizer.sanitize(text))
    print_with_len(heb_tokenizer.get_words("test אֶחָֽד"))
    # print('has_word=%s'%heb_tokenizer.has_word(text))
    # print_with_len(heb_tokenizer.get_mwe(text))
    # print_with_len(heb_tokenizer.get_mwe_words(text))
