import pandas as pd
from pycylon.index import Index, RangeIndex, NumericIndex, CategoricalIndex, ColumnIndex, \
    range_calculator
from pycylon import Table
from pycylon import CylonContext
import pyarrow as pa


def test_with_pandas():
    pdf = pd.DataFrame([[1, 2, 3, 4, 5, 'a'], [6, 7, 8, 9, 10, 'b'], [11, 12, 13, 14, 15, 'c'],
                        [16, 17, 18, 19, 20, 'a'], [16, 17, 18, 19, 20, 'd'],
                        [111, 112, 113, 114, 5,
                         'a']])

    # print(pdf)
    pdf1 = pdf.set_index([1, 2])
    # print(pdf1)
    print(pdf1.index)


def test_numeric_index():
    rg = range(0, 10, 1)
    rg1 = range(0, 10, 2)
    r = NumericIndex(data=rg)

    assert r.index_values == rg
    assert r.index_values != rg1


def test_range_index():
    rg = range(0, 10, 1)
    rg1 = range(0, 10, 2)
    r = RangeIndex(start=rg.start, stop=rg.stop, step=rg.step)

    assert r.index_values == rg
    assert r.index_values != rg1

    r1 = RangeIndex(rg)
    r2 = RangeIndex(rg)

    assert r1.index_values == rg
    assert r2.index_values != rg1


def calculate_range_size_manual(rg: range):
    sum = 0
    for i in rg:
        sum += 1
    return sum


def test_range_count():
    rg_1 = range(0, 10)
    rg_2 = range(0, 10, 2)
    rg_3 = range(0, 10, 3)
    rg_4 = range(0, 11, 2)
    rg_5 = range(0, 14, 3)
    rgs = [rg_1, rg_2, rg_3, rg_4, rg_5]
    for rg in rgs:
        assert range_calculator(rg) == calculate_range_size_manual(rg)


def test_set_index():
    pdf = pd.DataFrame([[1, 2, 3, 4, 5, 'a'], [6, 7, 8, 9, 10, 'b'], [11, 12, 13, 14, 15, 'c'],
                        [16, 17, 18, 19, 20, 'a'], [16, 17, 18, 19, 20, 'd'],
                        [111, 112, 113, 114, 5,
                         'a']])
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf)
    # set index by row indices
    row_indices = ['a', 'b', 'c', 'd', 'e', 'f']
    cn_tb.set_index(row_indices)

    assert cn_tb.row_count == len(row_indices)
    assert isinstance(cn_tb.index, CategoricalIndex)
    assert cn_tb.index.index_values == row_indices

    print(cn_tb.column_names)
    # set index column name
    col_name = "0"
    cn_tb.set_index(col_name)

    col_index = cn_tb.index
    assert isinstance(col_index, ColumnIndex)
    col_index_values = col_index.index_values

    ar_0 = pa.chunked_array([pdf[int(col_name)].values])
    ar_1 = pa.chunked_array([pdf[1].values])
    assert ar_0 == col_index_values
    assert ar_1 != col_index_values

    # set index by column names

    col_names = ["0", "1"]
    ars = []
    for col_name in col_names:
        ars.append(pa.chunked_array([pdf[int(col_name)]]))
    cn_tb.set_index(col_names)

    col_index = cn_tb.index
    assert isinstance(col_index, ColumnIndex)
    col_index_values = col_index.index_values

    for col_index_value, ar in zip(col_index_values, ars):
        assert col_index_value == ar


def test_loc():
    df = pd.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                      columns=['max_speed', 'shield'])

    print(df)
    ld = df.loc['viper']

    print(type(ld))


test_loc()
