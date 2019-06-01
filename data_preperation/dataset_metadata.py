import pathlib


class OCRDataInfo:
    def __init__(self, dataset_name):
        if dataset_name.lower() == 'tibetan':
            self.illegal_chars = ['&', '?', 'B', 'C', 'E', 'G', 'H', 'M', 'd', 'g', 'n', '{', '}', '\x81',
                '\x87', '«', '¹', '»', 'Ä', 'á', '¡', 'Ã', '+']
            self.legal_line_end_chars = ['༎', '༑',  '།', '་']
            self.initial_sign = '༄༄་'
            self.initial_special_sign = '༄༄༄་་'
            self.legal_sep_strs = ['༎', '༑', '།', '།།']
        elif dataset_name.lower() == 'wiener':
            self.illegal_chars = ['&', '?', '{', '}', '\x81', '\x87', '«', '¹', '»', '+']
            self.legal_line_end_chars = ['\n']
            self.initial_sign = '1. '
            self.initial_special_sign = '2. '
            self.legal_sep_strs = [' ', '\t']


class SynthDataInfo(OCRDataInfo):
    def __init__(self, is_long, use_spacing, multi_line, dataset_name):
        super(SynthDataInfo, self).__init__(dataset_name)
        self.multi_line = multi_line
        if dataset_name.lower() == 'tibetan':
            self.font_names = ['Qomolangma-Drutsa', 'Qomolangma-Betsu', 'Shangshung Sgoba-KhraChen', 'Shangshung Sgoba-KhraChung']
            self.fonts_dir = str(pathlib.Path('extra/Fonts').absolute())
        elif dataset_name.lower() == 'wiener':
            self.font_names = ['F25 BlackletterTypewriter', 'Breitkopf Fraktur', 'Gabriele Black Ribbon FG',
                               'CMU Typewriter Text', 'zai Olivetti-Underwood Studio 21 Typewriter',
                               'TlwgTypewriter', 'CMU Typewriter Text Variable Width', 'Gabriele Dark Ribbon FG',
                               'Gabriele Light Ribbon FG', ]
            self.fonts_dir = str(pathlib.Path('../../wiener_fonts').absolute())

        if is_long:
            self.min_len = 130
            self.max_len = 180
            self.min_allowed_len = 80
            self.max_allowed_len = 200
        else:
            # short lines:
            self.min_len = 10
            self.max_len = 50
            self.min_allowed_len = 5
            self.max_allowed_len = 60

        if use_spacing:
            self.has_rand_spaces = True
            self.has_initial_signs = True
            self.has_initial_space = True
        else:
            self.has_rand_spaces = False
            self.has_initial_signs = False
            self.has_initial_space = False

        self.num_rand_spaces_range = 3
        self.rand_space_range = [2, 6]
        self.initial_sign_space_range = [1, 8]
        self.initial_sign_prob = 0.1
        self.initial_special_sign_prob = 0.01
        self.initial_space_range = [0, 4]