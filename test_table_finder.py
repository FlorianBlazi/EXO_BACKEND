from PIL import Image
from table_finder import TableFinder
from PIL.Image import Image as Image_PIL


def test_table_finder_init():
    finder = TableFinder()
    assert finder is not None


def test_find_tables():
    finder = TableFinder()

    image = Image.new("RGB", (256, 256), "white")
    result = finder.find_tables(image)

    assert type(result) == dict
    assert "scores" in result
    assert "labels" in result
    assert "boxes" in result


def test_identify_tables():
    finder = TableFinder()

    image = Image.new("RGB", (256, 256), "white")

    results = finder.find_tables(image)

    boxed = finder.identify_tables(image, results)

    assert isinstance(boxed, Image_PIL)
