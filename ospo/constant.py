# model
IMG_SIZE = 384
PATCH_SIZE = 16
PARALLEL_SIZE = 1
IMAGE_TOKEN_NUM_PER_IMAGE = 576
VOCAB_SIZE = 16384


# step 1
CATEGORY_LIST = ["attribute", "layout", "non-spatial", "complex"]

ATTRIBUTE_MAX_LEN = 667
SPATIAL_MAX_LEN = 1000
NUMERACY_MAX_LEN = 4000
NON_SPATIAL_COMPLEX_MAX_LEN = 4000

INCLUDED_NUMBER_WORDS = ["one", "two", "three", "four"]
NON_INCLUDED_NUMBER_WORDS = [
    "five","six","seven","eight","nine","ten", "eleven","twelve","thirteen","fourteen",
    "fifteen","sixteen","seventeen","eighteen","nineteen","twenty","thirty","forty",
    "fifty","sixty","seventy","eighty","ninety","hundred","thousand","million","billion"
]

