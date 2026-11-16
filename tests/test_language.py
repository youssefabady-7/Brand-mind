"""Unit tests for hybrid language and Egyptian dialect detection.
"""

import pytest
from src.agents.language import detect_language


def test_detect_english():
    assert detect_language("What are your shipping rates for Cairo?") == "en"
    assert detect_language("Can I return unopened skincare products?") == "en"
    assert detect_language("Where is my order LUM-1001?") == "en"


def test_detect_standard_arabic():
    assert detect_language("ما هي سياسة الاسترجاع والاستبدال المعتمدة لديكم؟") == "ar"
    assert detect_language("هل المنتجات مناسبة للمرأة الحامل؟") == "ar"


def test_detect_egyptian_arabic():
    # Egyptian markers: عايز, فين, النهاردة, أوي, كوتش
    assert detect_language("عايز أعرف الأوردر بتاعي وصل لفين النهاردة؟") == "ar_eg"
    assert detect_language("الكوتش ده ضيق أوي وعايز أبدله بمقاس أكبر") == "ar_eg"
    assert detect_language("هو التوصيل لإسكندرية بياخد أد إيه؟") == "ar_eg"
    assert detect_language("وشي احمر خالص وحاسس بحرقان شديد") == "ar_eg"


def test_detect_arabizi():
    assert detect_language("3ayez a3raf el order bta3y feen") == "ar_eg"


def test_empty_string_defaults_english():
    assert detect_language("") == "en"
    assert detect_language("   ") == "en"
